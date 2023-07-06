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

from src.hicdiff import Unet, GaussianDiffusion


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
cell_lin ="Human"
cell_no = 2
deg = 'deno'
sigma = 0.1
image_channel = 1
image_size = 64
timestep = 1000
shedular = 'linear'  # 'linear' or 'sigmoid'

# below is used to load the diffusion model
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    self_condition = False
)
diffusion = GaussianDiffusion(
    model,
    image_size=64,
    timesteps=timestep,  # number of steps
    loss_type='l2',  # L1 or L2
    beta_schedule = shedular,
    auto_normalize=False
).to(device)


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

'''
#load data
if cell_lin == "Dros_cell":
    dm_test = GSE130711Module(batch_size = 64, deg = 'deno', cell_No=cell_no)
if cell_lin == "Human":
    dm_test = GSE130711Module(batch_size=64, deg = 'deno', cell_No=cell_no)
dm_test.prepare_data()
dm_test.setup(stage='test')

ds     = dm_test.test_dataloader().dataset.data[65:66]
target = dm_test.test_dataloader().dataset.target[65:66]
'''

#Our diffusion models' weight

model_hicEdrn = diffusion_h.to(device)
file_path1 = str(root)+"/Model_Weights/"+"bestg_40000_c64_s64_"+cell_lint+str(cell_not)+"_HiCedrn_l2_"+shedular[:3]+"_trans.pytorch"
model_hicEdrn.load_state_dict(torch.load(file_path1))
model_hicEdrn.eval()

model_unet = diffusion.to(device)
file_path1 = str(root)+"/Model_Weights/"+"bestg_40000_c64_s64_"+cell_lint+str(cell_not)+"_unet_l2_"+shedular[:3]+"_trans.pytorch"  # trained on multicom.rnet
model_unet.load_state_dict(torch.load(file_path1))
model_unet.eval()

#pass through models

v_m ={}
chro =  2  if cell_lin == 'Dros' else 2  # 2
chro1 = 1

#compute vision metrics
print("hicedrn")
visionMetrics = vm.VisionMetrics(image_channel = image_channel, image_size = image_size, sehedule = shedular, timestep = timestep)
v_m[chro1, 'hicedrn']=visionMetrics.getMetrics(model=model_hicEdrn.model, model_name = 'hicedrn_l2_' + shedular[:3], device = device, chro = chro, deg=deg, sigma=sigma, cellN=cell_no, cell_line=cell_lin)

print("unet")
visionMetrics = vm.VisionMetrics(image_channel = image_channel, image_size = image_size, sehedule = shedular, timestep = timestep)
v_m[chro1, 'unet']=visionMetrics.getMetrics(model=model_unet.model, model_name = 'unet_l2_' + shedular[:3], device = device, chro = chro, deg=deg, sigma=sigma, cellN=cell_no, cell_line=cell_lin)


model_names = ['hicedrn', 'unet']
# model_names = ['hicedrn']
metric_names = ['ssim', 'psnr', 'mse', 'snr', 'pcc', 'spc', 'gds']


# below is to record the gds values for each cell
gds_path = cell_lin+str(cell_no)+"_"+deg+"_"+str(sigma)
gds_dir = str(root) + "/Metrics"
if not os.path.exists(gds_dir):
    os.makedirs(gds_dir, exist_ok = True)
record_gds = open(gds_dir+"/"+gds_path+"_diff_uncond_trans2_"+shedular[:3]+".txt", "w")
#record_gds.write("\n"+gds_path+":\n")

cell_text = []
for mod_nm in model_names:
    met_list = []
    record_gds.write("\n"+mod_nm + "\n")
    for met_nm in metric_names:
        met_list.append("{:.4f}".format(np.mean(v_m[chro1, mod_nm]['pas_'+str(met_nm)])))
        #if met_nm == 'gds':
        record_gds.write(met_nm + ":\t" + str(np.mean(v_m[chro1, mod_nm]['pas_' + str(met_nm)])) + "\n")
    cell_text.append(met_list)
record_gds.close()

plt.subplots_adjust(left=0.2, top=0.8)
plt.table(cellText=cell_text, rowLabels=model_names, colLabels=metric_names, loc='top')
#plt.title(chro)
plt.show()



