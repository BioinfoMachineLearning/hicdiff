#the implementation of SSIM in this file is pulled from DeepHiC https://github.com/omegahh/DeepHiC
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from math import log10
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
from tqdm import tqdm
import torch

from processdata.PrepareData_linear_sing import GSE130711Module as sing
from processdata.PrepareData_linear_sing import GSE131811Module as sing_D

from processdata.PrepareData_linear import GSE130711Module as population
from processdata.PrepareData_linear import GSE131811Module as population_D

#from processdata.PrepareData_tensorH import GSE130711Module
from src.Utils.loss.SSIM import ssim
from src.Utils.GenomeDISCO import compute_reproducibility
from src.datasets import inverse_data_transform

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

class VisionMetrics:
    def __init__(self):
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

    def log_means(self, name):
        return (name, np.mean(self.metric_logs[name]))

    def getMetrics(self, model, model_name = 'HiCdiff', device = None, chro = "test", deg = 'deno', sigma = 0.1, cellN = 21,  cell_line="Dros_cell"):
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

        #for e, epoch in enumerate(self.test_loader):
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # prepare the test datasets
        dm_test = None
        if cellN == 1 or cellN == 22:
            if cell_line == "Dros":
                dm_test = population_D(batch_size=64, deg=deg, sigma_0=sigma, cell_No=cellN)
            elif cell_line == "Human":
                dm_test = population(batch_size=64, deg=deg, sigma_0=sigma, cell_No=cellN)
        elif cellN in [2, 3, 4, 5, 6]:
            if cell_line == "Dros":
                dm_test = sing_D(batch_size=64, deg=deg, sigma_0=sigma, cell_No=cellN)
            elif cell_line == "Human":
                dm_test = sing(batch_size=64, deg=deg, sigma_0=sigma, cell_No=cellN)
        dm_test.prepare_data()
        dm_test.setup(stage=chro)
        test_loader = dm_test.test_dataloader()

        # below is some variables to store the modeling results
        result_pr = None
        result_hr = None
        result_lr = None
        result_inds = None

        batch_ssims = []
        batch_mses = []
        batch_psnrs = []
        batch_snrs = []
        batch_spcs = []
        batch_pccs = []
        
        batch_gds = []


        test_result = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0, 'pccs':0, 'pcc':0, 'spcs':0, 'spc':0, 'snrs':0, 'snr':0}

        BatchNum = 0
        test_bar = tqdm(test_loader)
        with torch.no_grad():
            for lr, hr, _, inds in test_bar:
                batch_size = lr.size(0)
                test_result['nsamples'] += batch_size

                lr = lr.to(device)
                hr = hr.to(device)
                # data, full_target, info = epoch

                if model_name == "hiedsr" or model_name == "hiedsrgan":  #no need padding the input data
                    out = model(lr)

                if model_name == "hicplus" or model_name == "hicsr":  #need padding the input data
                    temp = F.pad(lr, (6, 6, 6, 6), mode='constant')
                    out = model(temp)

                if model_name == "deephic" or model_name=='unet' or model_name=='hicarn': # no need padding the data
                    out = model(lr)

                # to store the result in range(-1, 1) and pay attention that hr and out are on GPU
                if BatchNum == 0:
                    result_pr = out.cpu()
                    result_hr = hr.cpu()
                    result_lr = lr.cpu()
                    result_inds = inds
                else:
                    result_pr = torch.cat((result_pr, out.cpu()))
                    result_hr = torch.cat((result_hr, hr.cpu()))
                    result_lr = torch.cat((result_lr, lr.cpu()))
                    result_inds = torch.cat((result_inds, inds))

                BatchNum = BatchNum + 1
                # convert the pixels in x to range(0, 1) to measure them
                out = inverse_data_transform('rescaled', out)  # out = torch.stack(out)  # should first convert a list of tensor to one tensor
                hr = inverse_data_transform('rescaled', hr)  # hr = torch.stack(hr)   #should first convert a list of tensor to one tensor

                #print(out)
                #input("press enter to contnue....")
                batch_mse = ((out - hr) ** 2).mean()
                test_result['mse'] += batch_mse * batch_size
                batch_ssim = self.ssim(out, hr)
                test_result['ssims'] += batch_ssim * batch_size
                test_result['psnr'] = 10 * log10(1 / (test_result['mse'] / test_result['nsamples']))
                test_result['ssim'] = test_result['ssims'] / test_result['nsamples']

                batch_snr = (hr.sum() / ((hr - out) ** 2).sum().sqrt())
                if ((hr - out) ** 2).sum().sqrt() == 0 and hr.sum() == 0:
                    batch_snr = torch.tensor(0.0)
                test_result['snrs'] += batch_snr * batch_size
                test_result['snr'] = test_result['snrs']
                batch_pcc = pearsonr(out.cpu().flatten(), hr.cpu().flatten())[0]
                batch_spc = spearmanr(out.cpu().flatten(), hr.cpu().flatten())[0]
                test_result['pccs'] += batch_pcc * batch_size
                test_result['spcs'] += batch_spc * batch_size
                test_result['pcc'] = test_result['pccs']/test_result['nsamples']
                test_result['spc'] = test_result['spcs']/test_result['nsamples']

                batch_ssims.append(test_result['ssim'])
                batch_psnrs.append(test_result['psnr'])
                batch_mses.append(batch_mse)
                batch_snrs.append(test_result['snr'])
                batch_pccs.append(test_result['pcc'])
                batch_spcs.append(test_result['spc'])
                
                for i, j in zip(hr, out):
                    if hr.sum() == 0:
                        continue
                    out1 = torch.squeeze(j, dim = 0)
                    hr1 = torch.squeeze(i, dim = 0)
                    out2 = out1.cpu().detach().numpy()
                    hr2 = hr1.cpu().detach().numpy()
                    genomeDISCO = compute_reproducibility(out2, hr2, transition = True)
                    batch_gds.append(genomeDISCO)

            # below is to get the vision comparison with low, target and predict
            fig, ax = plt.subplots(1, 4)  # just one row/colum this will think as one-dimensional
            '''for j in range(0, 2): # in order to set the x_ticks and y_ticks without any labels/digits
                ax[j].set_xticks([])
                ax[j].set_yticks([])'''

            ds_out = result_lr[7][0][:, :]
            show1 = ax[0].imshow(ds_out, cmap="Reds")
            ax[0].set_title("Noisy")
            fig.colorbar(show1, ax=ax[0], location='bottom', orientation='horizontal')

            ds_out1 = result_hr[7][0][:, :]
            show2 = ax[1].imshow(ds_out1, cmap="Reds")
            ax[1].set_title("Target")
            fig.colorbar(show2, ax=ax[1], location='bottom', orientation='horizontal')

            ds_out2 = result_pr[7][0][:, :]
            show3 = ax[2].imshow(ds_out2, cmap="Reds")
            ax[2].set_title("predict")
            fig.colorbar(show3, ax=ax[2], location='bottom', orientation='horizontal')

            ds_out3 = torch.clamp(result_pr[7][0][:, :], -1, 1)
            show4 = ax[3].imshow(ds_out3, cmap = "Reds")
            ax[3].set_title("predict")
            fig.colorbar(show4, ax = ax[3], location = 'bottom', orientation = 'horizontal')

            plt.show()

        Nssim = sum(batch_ssims) / len(batch_ssims)
        Npsnr = sum(batch_psnrs) / len(batch_psnrs)
        Nmse = sum(batch_mses) / len(batch_mses)
        Nsnr = sum(batch_snrs) / len(batch_snrs)
        Npcc = sum(batch_pccs) / len(batch_pccs)
        Nspc = sum(batch_spcs) / len(batch_spcs)
        Ngds = sum(batch_gds) / len(batch_gds)

        self.metric_logs['pas_ssim'].append(Nssim.cpu())
        self.metric_logs['pas_psnr'].append(Npsnr)
        self.metric_logs['pas_mse'].append(Nmse.cpu())
        self.metric_logs['pas_snr'].append(Nsnr.cpu())
        self.metric_logs['pas_pcc'].append(Npcc)
        self.metric_logs['pas_spc'].append(Nspc)
        self.metric_logs['pas_gds'].append(Ngds)

            # self._logPCC(data=data, target=full_target, output=output)
            # self._logSPC(data=data, target=full_target, output=output)
            # self._logMSE(data=data, target=full_target, output=output)
            # self._logPSNR(data=data, target=full_target, output=output)
            # self._logSNR(data=data, target=full_target, output=output)
            # self._logSSIM(data=data, target=full_target, output=output)
        print(list(map(self.log_means, self.metric_logs.keys())))
        return self.metric_logs

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
