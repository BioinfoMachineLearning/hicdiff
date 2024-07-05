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
from src.hicdiff_condition import GaussianDiffusion

from src.Utils import metrics_cond as vm
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#below is the cell information for test
cell_lin ="Human"  # 'Human' or 'Dros'
cell_no = 1
deg = 'deno'
sigma = 0.1
image_channel = 1
image_size = 64
shedular = 'sigmoid'   # 'linear' or 'sigmoid'

#below information is used for loading the pretrained diffusion models' paths
cell_not = 1
cell_lint = "Human"
sigma_0 = 0.1
model_type = 'condition'
timestep = 1000 if model_type == 'condition' else 2000


# below is used to load the diffusion model
model_h = hicedrn_Diff(
    self_condition = True
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
file_inter = cell_lint+str(cell_not)+'_'+deg+'_'+str(sigma)+"/"


#Load Our diffusion models' weight

model_hicEdrn = diffusion_h.to(device)
# for human1_0.1 population train
file_path1 = str(root)+"/Model_Weights/"+"bestg_40000_c64_s64_"+cell_lint+str(cell_not)+"_HiCedrn_cond_l2_"+shedular[:3]+".pytorch"
model_hicEdrn.load_state_dict(torch.load(file_path1))
model_hicEdrn.eval()


#pass through models
chro =  "test"  # if cell_lin == 'Dros' else 2   # 2
print("hicdiff")
visionMetrics = vm.VisionMetrics(image_channel = image_channel, image_size = image_size, timestep = timestep, type = model_type)
predict = visionMetrics.getMetrics(model=model_hicEdrn.super_resolution, model_name = 'hicedrn_l2_'+ shedular[:3], device = device, chro = chro, deg=deg, sigma=sigma, cellN=cell_no, cell_line=cell_lin)




