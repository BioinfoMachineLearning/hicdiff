#the implementation of SSIM in this file is pulled from DeepHiC https://github.com/omegahh/DeepHiC
import matplotlib.pyplot as plt

import torch.nn as nn
from math import log10
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
from tqdm import tqdm
import torch

import os
import subprocess

from processdata.PrepareData_linear_sing import GSE130711Module as sing
from processdata.PrepareData_linear_sing import GSE131811Module as sing_D

from processdata.PrepareData_linear import GSE130711Module as population
from processdata.PrepareData_linear import GSE131811Module as population_D
#from processdata.PrepareData_tensorH import GSE130711Module
from src.Utils.loss.SSIM import ssim
from src.Utils.GenomeDISCO import compute_reproducibility

from src.datasets import inverse_data_transform
from src.functions.denoising import efficient_generalized_steps
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        # the origal methods for sigmoid shedular
        # betas = np.linspace(-6, 6, num_diffusion_timesteps)
        # betas = sigmoid(betas) * (beta_end - beta_start) + beta_start

        timesteps = num_diffusion_timesteps
        start = -3
        end = 3
        tau = 1
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
        v_start = torch.tensor(start / tau).sigmoid( )
        v_end = torch.tensor(end / tau).sigmoid( )
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid( ) + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999).float()

    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class VisionMetrics:
    def __init__(self, image_channel = 1, image_size = 64, sehedule = 'linear', timestep = 20):
        self.ssim         = ssim
        self.metric_logs = {
            #"pre_pcc":[],
            "pas_pcc":[],
            #"pre_spc":[],
            "pas_spc":[],
            #"pre_psnr":[],
            "pas_psnr":[],
            #"pre_ssim":[],
            "pas_ssim":[],
            #"pre_mse":[],
            "pas_mse":[],
            #"pre_snr":[],
            "pas_snr":[],
            "pas_gds":[]
            }
        betas = get_beta_schedule(
            beta_schedule=sehedule,
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000,
        )

        if sehedule == 'linear':
            self.betas = torch.from_numpy(betas).float()
        else:
            self.betas = betas
        self.num_timesteps = betas.shape[0]
        self.image_channel = image_channel
        self.image_size = image_size
        self.timestep = timestep

    def log_means(self, name):
        return (name, np.mean(self.metric_logs[name]))

    def getMetrics(self, model, model_name = 'HiCdiff', device = None, chro = "test", deg = 'deno', sigma = 0.1, cellN = 21,  cell_line="Dros_cell", res = None):

        #for e, epoch in enumerate(self.test_loader):
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # prepare the test datasets
        dm_test = None
        if cellN == 1 or cellN == 22:
            if cell_line == "Dros":
                dm_test = population_D(batch_size = 64, deg = deg, sigma_0 = sigma, cell_No = cellN)
            elif cell_line == "Human":
                dm_test = population(batch_size = 64, deg = deg, sigma_0 = sigma, cell_No = cellN)
        elif cellN in [2, 3, 4, 5, 6]:
            if cell_line == "Dros":
                dm_test = sing_D(batch_size = 64, deg = deg, sigma_0 = sigma, cell_No = cellN)
            elif cell_line == "Human":
                dm_test = sing(batch_size = 64, deg = deg, sigma_0 = sigma, cell_No = cellN)
        dm_test.prepare_data()
        dm_test.setup(stage=chro)
        test_loader = dm_test.test_dataloader()

        # prepare the degradation matrix
        from src.functions.H_func import MakeFunc
        H_funcs = MakeFunc(deg=deg, image_channel=self.image_channel, image_size=self.image_size, device=device)
        sigma_0 = sigma

        # below is some variables to store the modeling results
        result_pr = None
        result_hr = None
        result_lr = None
        result_inds = None

        # below is to build the directory  to store the modeling results
        Outdir = str(root) + '/Outputs_diff'
        ModelResult = model_name + cell_line + str(cellN) + "_" + deg + "_" +str(sigma) + "_trans2_" + str(self.timestep)
        if not os.path.exists(Outdir+"/"+ModelResult):
            subprocess.run("mkdir -p "+Outdir+"/"+ModelResult, shell = True)


        test_result = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0, 'pccs':0, 'pcc':0, 'spcs':0, 'spc':0, 'snrs':0, 'snr':0}

        BatchNum = 0
        test_bar = tqdm(test_loader)
        with torch.no_grad():
            for lr, hr, sp, inds in test_bar:
                batch_size = lr.size(0)
                test_result['nsamples'] += batch_size

                sp = sp.to(device)
                hr = hr.to(device)
                # data, full_target, sample, info = epoch

                # Begin DDIM
                x = torch.randn(
                    sp.shape[0],
                    self.image_channel,
                    self.image_size,
                    self.image_size,
                    device= device,
                )

                out, _ = self.sample_image(x, model, H_funcs, sp, sigma_0, device = device, last=False, cls_fn=None, classes=None)
                out = out[-1]

                # to store the result in range(-1, 1) and pay attention that hr and out are on GPU
                if BatchNum == 0:
                    result_pr = out.cpu()
                    result_hr = hr.cpu()
                    result_lr = lr
                    result_inds = inds
                else:
                    result_pr = torch.cat((result_pr, out.cpu()))
                    result_hr = torch.cat((result_hr, hr.cpu()))
                    result_lr = torch.cat((result_lr, lr))
                    result_inds = torch.cat((result_inds, inds))

                BatchNum = BatchNum + 1
                # convert the pixels in x to range(0, 1) to measure them
                out = inverse_data_transform('rescaled', out)   #out = torch.stack(out)  # should first convert a list of tensor to one tensor
                hr = inverse_data_transform('rescaled', hr)   #hr = torch.stack(hr)   #should first convert a list of tensor to one tensor

            # below is to store the modeling in a fold as numpy data structure
            predict = result_pr.numpy()
            target = result_hr.numpy()
            low = result_lr.numpy()
            index = result_inds.numpy()
            np.save(Outdir + "/" + ModelResult + "/" + "target", target)
            np.save(Outdir + "/" + ModelResult + "/" + "noisy", low)
            np.save(Outdir + "/" + ModelResult + "/" + "predict", predict)
            np.save(Outdir + "/" + ModelResult + "/" + "inds", index)


        return predict

    def sample_image(self, x, model, H_funcs, y_0, sigma_0,  device = None, last=False, cls_fn=None, classes=None): #used to sample the data
        skip = self.num_timesteps // self.timestep  # timesteps  controls how many timesteps used in the process
        seq = range(0, self.num_timesteps, skip)
        # note that,  the sigma_0 controls the addictive gausissn  noise level in y-equations, H_funcs controls the degradation matrix in y-equations
        # self.betas is result from the linear betas-schedules
        self.betas = self.betas.to(device)
        x = efficient_generalized_steps(x, seq, model, self.betas, H_funcs, y_0, sigma_0, etaB=1.0, etaA=0.85, etaC=0.85, cls_fn=cls_fn, classes=classes, device = device)
        if last:
            x = x[0][-1]
        return x

###
if __name__=='__main__':
    print("\nTest for the stardrd metrics\n")
    '''device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    visionMetrics = VisionMetrics()
    visionMetrics.setDataset(20, cell_line="GSE131811")
    WEIGHT_PATH   = "deepchromap_weights.ckpt"
    model         =     Generator().to(device)
    pretrained_model = model.load_from_checkpoint(WEIGHT_PATH)
    pretrained_model.freeze()
    visionMetrics.getMetrics(model=pretrained_model, spliter=False)'''
