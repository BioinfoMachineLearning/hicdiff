import matplotlib.pyplot as plt
import os
import math
import subprocess
import glob
import pyrootutils
import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F

import gc
import cooler

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# below is used to add noise for regular deeplearning network and vaidation or test datasets for DDPM
def extract(a, t, x_shape):  # t has same dimensions as a, this in order to get the beta_t or alpha_t with corrresponding t
    b, *_ = t.shape   # a = lenth of numsteps
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) # this will make the a has the shape (b, 1, 1, 1)

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)   # here define the dtype is float64

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def q_sample(target, t, beta_schedule = 'linear', timesteps = 1000, schedule_fn_kwargs = dict()):  # here noise is gaussion  noise which is used for forward-adding noising sample
    # use the same beta_schedule as in network training to add the noise
    if beta_schedule == 'linear':
        beta_schedule_fn = linear_beta_schedule
    elif beta_schedule == 'cosine':
        beta_schedule_fn = cosine_beta_schedule
    elif beta_schedule == 'sigmoid':
        beta_schedule_fn = sigmoid_beta_schedule
    else:
        raise ValueError(f'unknown beta schedule {beta_schedule}')

    betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)  # in betas in the DDPM model
    alphas = 1. - betas  # alphas in the DDPM model
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    noise = torch.randn_like(target) # noise has the same shape as x_start

    return (
        extract(sqrt_alphas_cumprod, t, target.shape) * target +
        extract(sqrt_one_minus_alphas_cumprod, t, target.shape) * noise
    )

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        # the second method to add poisson noise
        noisy = image + np.random.poisson(image)
        return noisy
    elif noise_typ =="speckle":
        row,col = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)
        noisy = image + image * gauss
        return noisy

def splitPieces(fn, piece_size, step, resol):
    data   = np.load(fn)
    pieces = []
    bound  = data.shape[0]
    bound1 = data.shape[1]
    assert bound == bound1

    scal = int(40000/resol)
    rest = bound % piece_size
    if rest != 0:
        data = torch.from_numpy(data)  # convert to tensor
        pad_size  = piece_size - rest
        data = F.pad(data, (0, pad_size, 0, pad_size), value = 0.0)
    data = np.array(data)  # convert to numpy
    bound = data.shape[0]  #the data shape after padding
    for i in range(0, bound, step): # for half is enough because the entire map is symmetric
        for j in range(i, bound, step):
            if abs(i - j) <= int(piece_size * 4 * scal + 1) and i + step <= bound and j + step <= bound:
                pieces.append(data[i:i+piece_size, j:j+piece_size])
    pieces = np.asarray(pieces)
    pieces = np.expand_dims(pieces,1)
    return pieces

def loadBothConstraints(stria, strib, res):
    contact_mapa  = np.loadtxt(stria)  # high resolution with cooler balance
    contact_mapb  = np.loadtxt(strib)  # high resolution with raw count's number

    print("============raw contact mapb shape: {}  and data length is {}".format(contact_mapb.shape, len(contact_mapb)))

    # method  has similar function
    rowsa         = (contact_mapa[:,0]/res).astype(int)
    colsa         = (contact_mapa[:,1]/res).astype(int)
    valsa         = contact_mapa[:,2]
    rowsb         = (contact_mapb[:,0]/res).astype(int)
    colsb         = (contact_mapb[:,1]/res).astype(int)
    valsb         = contact_mapb[:,2].astype(int)
    bigbin        = np.max((np.max((rowsa, colsa)), np.max((rowsb, colsb))))
    smallbin      = np.min((np.min((rowsa, colsa)), np.min((rowsb, colsb))))
    mata          = np.zeros((bigbin-smallbin+1, bigbin-smallbin+1), dtype='float32')
    matb          = np.zeros((bigbin-smallbin+1, bigbin-smallbin+1), dtype= 'int')
    coordinates   = list(range(smallbin, bigbin))
    i=0
    for ra,ca,ia in zip(rowsa, colsa, valsa):
        i = i+1
        #print(str(i)+"/"+str(len(valsa)+len(valsb)))
        mata[ra-smallbin, ca-smallbin] = ia
        mata[ca-smallbin, ra-smallbin] = ia
    for rb,cb,ib in zip(rowsb, colsb, valsb):
        i = i+1
        #print(str(i)+"/"+str(len(valsa)+len(valsb)))
        matb[rb-smallbin, cb-smallbin] = ib
        matb[cb-smallbin, rb-smallbin] = ib
    diaga         = np.diag(mata)  # np.diag() will give a 1-D array
    diagb         = np.diag(matb)

    removeidx = np.unique(np.concatenate((np.argwhere(diaga == 0)[:, 0], np.argwhere(np.isnan(diaga))[:, 0])))
    print("\n the new removeidx shape: {} and its length: {}".format(removeidx.shape, len(removeidx)))
    #  input("Press Enter to continue...")  # input used to check some thing

    mata = np.delete(mata, removeidx, axis=0)
    mata = np.delete(mata, removeidx, axis=1)

    # normalize the data in range(-1, 1)
    per_a       = np.percentile(mata, 99.9)
    print(np.percentile(mata, 99.9), np.percentile(mata, 99.99), np.max(mata))
    mata        = np.clip(mata, 0, per_a)
    mata        = mata/per_a    # the range (0, 1)
    mata        = 2 * mata - 1.0   # the range (-1, 1)


    matb = np.delete(matb, removeidx, axis=0)
    matb = np.delete(matb, removeidx, axis=1)
    per_b       = np.percentile(matb, 99.9)
    print(np.percentile(matb, 99.9), np.percentile(matb, 99.99), np.max(matb))
    matb        = np.clip(matb, 0, per_b)
    matb        = matb/per_b   # the range (0, 1)
    matb        = 2 * matb - 1.0  # the range (-1, 1)

    return mata


class GSE130711Module(pl.LightningDataModule):
    def __init__(self,
                 batch_size = 64,
                 res = 40000,
                 piece_size = 64,
                 cell_line = 'Human',
                 cell_No = 1,
                 timesteps = 1000,
                 beta_schedule='linear',
                 ): #64 is used for unet_model
        super( ).__init__( )
        self.batch_size = batch_size
        self.res = res
        self.step = piece_size  # here the parameter should be modified
        self.piece_size = piece_size
        self.cellLine = cell_line
        self.cellNo = cell_No
        self.dirname = "DataFull_"+self.cellLine+"_cell"+str(self.cellNo)+"_"+str(self.res)
        self.timestep = timesteps
        self.beta_shedule = beta_schedule

    def extract_constraint_mats(self):
        if not os.path.exists(self.dirname+"/Constraints"):
            subprocess.run("mkdir -p "+self.dirname+"/Constraints", shell = True)

        outdir = self.dirname+"/Constraints"
        file_inter = glob.glob(str(root) + '/Datasets/Human/single/' + 'cell'+str(self.cellNo)+r'*.mcool')
        filepath = file_inter[0]
        AllRes = cooler.fileops.list_coolers(filepath)
        print(AllRes)

        c = cooler.Cooler(filepath + '::resolutions/' + str(self.res))
        c1 = c.chroms()[:]  # c1 chromesize information in the list format
        print(c1.loc[0, 'name'], c1.index)
        # print('\n')

        for i in c1.index:
            print(i, c1.loc[i, 'name'])
            chro = c1.loc[i, 'name']  # chro is str
            # print(type(chro))
            c2 = c.matrix(balance = True, as_pixels = True, join = True).fetch(chro)
            c3 = c2[['start1', 'start2', 'count']]
            # print(c2)
            c2 = c2[['start1', 'start2', 'balanced']]
            c2.fillna(0, inplace = True)
            # print(c2)
            if i >= 22:
                pass
            else:
                c2.to_csv(outdir+'/chrom_' + str(i + 1) + '_' + str(self.res) + '.txt', sep = '\t', index = False, header = False) # balanced
                c3.to_csv(outdir+'/chrom_' + str(i + 1) + '_' + 'count' + '.txt', sep = '\t', index = False, header = False)  # raw count

    def extract_create_numpy(self):
        if not os.path.exists(self.dirname+"/Full_Mats"):
            subprocess.run("mkdir -p "+self.dirname+"/Full_Mats", shell = True)

        globs = glob.glob(self.dirname+"/Constraints/chrom_1_" + str(self.res) + ".txt")
        if len(globs) == 0:
            print("wait.. first we need to extract mats and double check the mats")
            #  input("Press Enter to continue...")
            self.extract_constraint_mats()
        for i in range(1, 23):
            target = loadBothConstraints(
                self.dirname+"/Constraints/chrom_" + str(i) + "_" + str(self.res) + ".txt",
                self.dirname+"/Constraints/chrom_" + str(i) + "_" + "count" + ".txt",
                self.res)

            target = np.float32(target)

            print("the second time to convert float64 to float32")
            print(target.dtype)

            np.save(self.dirname+"/Full_Mats/GSE131811_mat_full_chr_" + str(i) + "_" + str(self.res), target)

    def split_numpy(self):
        if not os.path.exists(self.dirname+"/Splits"):
            subprocess.run("mkdir -p "+self.dirname+"/Splits", shell = True)

        globs = glob.glob(self.dirname+"/Full_Mats/GSE131811_mat_full_chr_1_" + str(self.res) + ".npy")
        if len(globs) == 0:
            self.extract_create_numpy()

        for i in range(1, 23):
            target = splitPieces(self.dirname+"/Full_Mats/GSE131811_mat_full_chr_" + str(i) + "_" + str(self.res) + ".npy",
                                    self.piece_size, self.step, resol = self.res)

            np.save(
                self.dirname+"/Splits/GSE131811_full_chr_" + str(i) + "_" + str(self.res) + "_piece_" + str(self.piece_size),
                target)
            # below to get the noisy data
            data_t = torch.from_numpy(target)
            b = target.shape[0]
            t = torch.randint(0, self.timestep, (b,), device=data_t.device).long()
            data = q_sample(data_t, t=t, beta_schedule = self.beta_shedule, timesteps = self.timestep)
            data = data.numpy()  # here is float64, i.e., long
            np.save(
                self.dirname + "/Splits/GSE131811_noisy_chr_" + str(i) + "_" + str(self.res) + "_piece_" + str(self.piece_size), data)


    def prepare_data(self):
        print("Preparing the Preparations ...")
        globs = glob.glob(
            self.dirname+"/Splits/GSE131811_full_chr_*_" + str(self.res) + "_piece_" + str(self.piece_size) + str(".npy"))
        if len(globs) > 20:
            print("Ready to go")
        else:
            print(".. wait, first we need to split the mats")
            self.split_numpy()

    class gse131811Dataset(Dataset):
        def __init__(self, full, tvt, res, piece_size, dir, time_step = 1000):
            self.piece_size = piece_size
            self.tvt = tvt
            self.res = res
            self.full = full
            self.dir = dir

            if full == True:
                if tvt in list(range(1, 23)):
                    self.chros = [tvt]
                if tvt == "train":
                    self.chros = [1, 3, 5, 7, 8, 9, 11, 13, 15, 16, 17, 19, 21, 22]
                elif tvt == "val":
                    self.chros = [4, 14, 18, 20] #
                elif tvt == "test":
                    self.chros = [2, 6, 10, 12] #

                self.target = np.load(
                    self.dir+"/Splits/GSE131811_full_chr_" + str(self.chros[0]) + "_" + str(self.res) + "_piece_" + str(
                        self.piece_size) + ".npy")

                self.data = np.load(
                    self.dir + "/Splits/GSE131811_noisy_chr_" + str(self.chros[0]) + "_" + str(self.res) + "_piece_" + str(
                        self.piece_size) + ".npy")

                self.info = np.repeat(self.chros[0], self.target.shape[0])

                for c, chro in enumerate(self.chros[1:]):
                    temp = np.load(
                        self.dir+"/Splits/GSE131811_full_chr_" + str(chro) + "_" + str(self.res) + "_piece_" + str(
                            self.piece_size) + ".npy")
                    print(self.target.shape, temp.shape, len(temp), chro)
                    # input("Press Enter to continue...")
                    if len(temp) == 0:
                        pass
                    else:
                        self.target = np.concatenate((self.target, temp))

                    temp = np.load(
                        self.dir + "/Splits/GSE131811_noisy_chr_" + str(chro) + "_" + str(self.res) + "_piece_" + str(
                            self.piece_size) + ".npy")
                    if len(temp) == 0:
                        pass
                    else:
                        self.data = np.concatenate((self.data, temp))
                        self.info = np.concatenate((self.info, np.repeat(chro, temp.shape[0])))

                self.target = torch.from_numpy(self.target)  # target is float
                self.data = torch.from_numpy(self.data)  # data is Double
                self.info = torch.from_numpy(self.info)

                print("========================= the stage of training =====================\n", tvt)
                print(self.target.shape)

            else:
                if tvt == "train":
                    self.chros = [15]
                elif tvt == "val":
                    self.chros = [16]
                elif tvt == "test":
                    self.chros = [17]
                self.target = np.load(
                    self.dir+"/Splits/GSE131811_full_chr_" + str(self.chros[0]) + "_" + str(self.res) + "_piece_" + str(
                        self.piece_size) + ".npy")
                self.data = np.load(
                    self.dir + "/Splits/GSE131811_noisy_chr_" + str(self.chros[0]) + "_" + str(self.res) + "_piece_" + str(
                        self.piece_size) + ".npy")
                self.info = np.repeat(self.chros[0], self.target.shape[0])

                self.target = torch.from_numpy(self.target)
                self.data = torch.from_numpy(self.data)
                self.info = torch.from_numpy(self.info)

                print("========================= the stage of training =====================\n", tvt)
                print(self.target.shape)

        def __len__(self):
            return self.target.shape[0]

        def __getitem__(self, idx):
            return self.data[idx], self.target[idx],  self.info[idx]

    def setup(self, stage = None):
        if stage in list(range(1, 23)):
            self.test_set = self.gse131811Dataset(full = True, tvt = stage, res = self.res, piece_size = self.piece_size,  dir = self.dirname)
        if stage == 'fit':
            self.train_set = self.gse131811Dataset(full = True, tvt = 'train', res = self.res, piece_size = self.piece_size,  dir = self.dirname)
            self.val_set = self.gse131811Dataset(full = True, tvt = 'val', res = self.res, piece_size = self.piece_size,  dir = self.dirname)
        if stage == 'test':
            self.test_set = self.gse131811Dataset(full = True, tvt = 'test', res = self.res, piece_size = self.piece_size, dir = self.dirname)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, num_workers = 12, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers = 12)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, num_workers = 12)


if __name__ == '__main__':
    obj = GSE130711Module(cell_No = 1)
    obj.prepare_data()
    obj.setup(stage = 'fit')
    obj.setup(stage = 'test')
    print("all thing is done!!!")

    aa = obj.test_dataloader().dataset.target
    print("\nTo check wether the data is tensor")
    print(type(aa))
    print(aa.shape)
    bb = obj.test_dataloader().dataset.data
    print(f'\nThe noisy data length is {bb.shape}')

    '''
    test_loader = obj.test_dataloader()
    i = 1
    for target, ind in test_loader:
        print(f'\nthe target batch id: {i} in test_loader')
        print(target.shape)
        print(f'the chrom of index shape is {ind.shape}')
        i = i+1
    '''

    ds_out = obj.test_dataloader().dataset.target[8:9]
    len1 = obj.test_dataloader().dataset.target.shape
    ds_out = ds_out[0][0][:, :]
    print("\nThe test target length is:{}".format(len1))
    print(ds_out)

    ds_out2 = obj.test_dataloader().dataset.data[8:9]
    len2 = obj.test_dataloader().dataset.data.shape
    ds_out2 = ds_out2[0][0][:, :]
    print("\nThe test data length is:{}".format(len2))
    print(ds_out2)

    fig, ax = plt.subplots(1, 2)  # just one row/colum this will think as one-dimensional
    '''for j in range(0, 2): # in order to set the x_ticks and y_ticks without any labels/digits
        ax[j].set_xticks([])
        ax[j].set_yticks([])'''

    show1 = ax[0].imshow(ds_out, cmap="Reds")
    ax[0].set_title("Target")
    fig.colorbar(show1, ax=ax[0], location = 'bottom', orientation = 'horizontal')

    show2 = ax[1].imshow(ds_out2, cmap="Reds")
    ax[1].set_title("Noisy data")
    fig.colorbar(show2, ax=ax[1], location='bottom', orientation='horizontal')

    plt.show()

