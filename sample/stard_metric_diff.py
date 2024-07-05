import sys
#sys.path.append(".")
sys.path.append("../")
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import pyrootutils

#Diffusion  models
#import src.hicdiff as unet
from src.model.hicedrn_Diff import hicedrn_Diff

from src.hicdiff import  GaussianDiffusion


from src.Utils import metrics_diff as vm
from processdata.PrepareData_linear import GSE130711Module

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#below is the cell information for test
cell_lin ="yeast"
cell_no = 2
deg = 'deno'
sigma = 0.1
image_channel = 1
image_size = 64
timestep = 1000
shedular = 'linear'  # 'linear' or 'sigmoid'
res = 10000

# below is used to load the diffusion model
model_h = hicedrn_Diff(
    self_condition = False
)
diffusion_h = GaussianDiffusion(
    model_h,
    image_size=64,
    timesteps=timestep,  # number of steps
    loss_type='l2',  # L1 or L2
    beta_schedule = shedular,
    auto_normalize=False
).to(device)

#below two is used for pretrained models' paths
cell_not = 2
cell_lint = "Human"

file_inter = cell_lint+str(cell_not)+'_'+deg+'_'+str(sigma)+"/"
#Load Our diffusion models' weight

model_hicEdrn = diffusion_h.to(device)
file_path1 = str(root)+"/Model_Weights/"+"bestg_40000_c64_s64_"+cell_lint+str(cell_not)+"_HiCedrn_l2_"+shedular[:3]+"_trans.pytorch"
model_hicEdrn.load_state_dict(torch.load(file_path1))
model_hicEdrn.eval()

#pass through models
chro =  'test'
print("hicdiff")
visionMetrics = vm.VisionMetrics(image_channel = image_channel, image_size = image_size, sehedule = shedular, timestep = timestep)
predict = visionMetrics.getMetrics(model=model_hicEdrn.model, model_name = 'hicedrn_l2_' + shedular[:3], device = device, chro = chro, deg=deg, sigma=sigma, cellN=cell_no, cell_line=cell_lin, res = res)



