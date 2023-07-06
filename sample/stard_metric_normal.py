import sys
#sys.path.append(".")
sys.path.append("../")
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import pyrootutils

#Our models
import src.model.schicedrn_gan as hiedsr
#other models
import src.model.hicsr   as hicsr
import src.model.deephic as deephic
import src.model.hicplus as hicplus
import src.model.Unet_parts1 as unet
import src.model.Hicarn as hicarn
from src.Utils import stard_metrics as vm
from processdata.PrepareData_linear import GSE130711Module

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

#below is the cell information for test
cell_lin ="Dros"
cell_no = 2
deg = 'deno'
sigma = 0.1
image_channel = 1
image_size = 64
timestep = 20

#below two is used for pretrained models' paths
cell_not = 2
cell_lint = "Human"
sigma_0 = 0.1

file_inter = cell_lint+str(cell_not)+'_'+deg+'_'+str(sigma)+"/"
#file_inter = "Downsample_"+str(percentage)+"/"

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


#Our models on multicom
hiedsrMod  = hiedsr.Generator().to(device)
file_path1 = str(root)+"/Model_Weights/"+"finalg_40000_c64_s64_"+cell_lint+str(cell_not)+"_"+deg+"_"+str(sigma_0)+"_schicedrn_const.pytorch"
hiedsrMod.load_state_dict(torch.load(file_path1))
hiedsrMod.eval()

model_hicarn = hicarn.Generator(num_channels=64).to(device)
file_path1 = str(root)+"/Model_Weights/"+"finalg_40000_c64_s64_"+cell_lint+str(cell_not)+"_"+deg+"_"+str(sigma_0)+"_hicarn.pytorch"
model_hicarn.load_state_dict(torch.load(file_path1))
model_hicarn.eval()
'''
hiedsrganMod = hiedsr.Generator().to(device)
file_path1 = str(root)+"/Model_Weights/"+file_inter+"bestg_40000_c64_s64_"+cell_lint+str(cell_not)+"_hiedsrgan.pytorch"
hiedsrganMod.load_state_dict(torch.load(file_path1))
hiedsrganMod.eval()
'''

# otheir models on muticom  (hicsr and nicplus these two will lead Hout = Hin - 12, the rest models will have Hout = Hin)
model_hicsr   = hicsr.Generator(num_res_blocks=15).to(device)
file_path1  = str(root)+"/Model_Weights/"+"finalg_40000_c64_s64_"+cell_lint+str(cell_not)+"_"+deg+"_"+str(sigma_0)+"_hicsr.pytorch"
model_hicsr.load_state_dict(torch.load(file_path1))
model_hicsr.eval()

model_deephic = deephic.Generator(scale_factor=1, in_channel=1, resblock_num=5).to(device)
file_path1 = str(root)+"/Model_Weights/"+"finalg_40000_c64_s64_"+cell_lint+str(cell_not)+"_"+deg+"_"+str(sigma_0)+"_deephic.pytorch"
model_deephic.load_state_dict(torch.load(file_path1))
model_deephic.eval()

model_hicplus = hicplus.Net(40,28).to(device)
file_path1 = str(root)+"/Model_Weights/"+"finalg_40000_c64_s64_"+cell_lint+str(cell_not)+"_"+deg+"_"+str(sigma_0)+"_hicplus.pytorch"
model_hicplus.load_state_dict(torch.load(file_path1))
model_hicplus.eval()

model_unet = unet.unet_2D().to(device)
file_path1 = str(root)+"/Model_Weights/"+"finalg_40000_c64_s64_"+cell_lint+str(cell_not)+"_"+deg+"_"+str(sigma_0)+"_LoopenhanceN.pytorch"
model_unet.load_state_dict(torch.load(file_path1))
model_unet.eval()


#pass through models

v_m ={}
chro =  "test"  if cell_lin == 'Dros' else 2   # 2s
chro1 = 1
#compute vision metrics

print("hiedsr")
visionMetrics = vm.VisionMetrics()
v_m[chro1, 'hiedsr']=visionMetrics.getMetrics(model=hiedsrMod, model_name="hiedsr",  device = device, chro = chro, deg=deg, sigma=sigma, cellN=cell_no, cell_line=cell_lin)

print("deephic")
visionMetrics = vm.VisionMetrics()
v_m[chro1, 'deephic']=visionMetrics.getMetrics(model=model_deephic, model_name="deephic",  device = device, chro = chro, deg=deg, sigma=sigma, cellN=cell_no, cell_line=cell_lin)

print("hicarn")
visionMetrics = vm.VisionMetrics()
v_m[chro1, 'hicarn']=visionMetrics.getMetrics(model=model_hicarn, model_name="hicarn",  device = device, chro = chro, deg=deg, sigma=sigma, cellN=cell_no, cell_line=cell_lin)


'''
print("hiedsrgan")
visionMetrics = vm.VisionMetrics()
v_m[chro1, 'hiedsrgan']=visionMetrics.getMetrics(model=hiedsrganMod, model_name="hiedsrgan",  device = device, chro = chro, deg=deg, sigma=sigma, cellN=cell_no, cell_line=cell_lin)
'''

print("HiCSR")
visionMetrics = vm.VisionMetrics()
v_m[chro1, 'hicsr']=visionMetrics.getMetrics(model=model_hicsr, model_name="hicsr",  device = device, chro = chro, deg=deg, sigma=sigma, cellN=cell_no, cell_line=cell_lin)

print("unet")
visionMetrics = vm.VisionMetrics()
v_m[chro1, 'unet']=visionMetrics.getMetrics(model=model_unet, model_name="unet", device = device, chro = chro, deg=deg, sigma=sigma, cellN=cell_no, cell_line=cell_lin)

print("hicplus")
visionMetrics = vm.VisionMetrics()
v_m[chro1, 'hicplus']=visionMetrics.getMetrics(model=model_hicplus, model_name="hicplus", device = device, chro = chro, deg=deg, sigma=sigma, cellN=cell_no, cell_line=cell_lin)


model_names  = ['hiedsr', 'hicarn',  'deephic', 'hicsr', 'unet', 'hicplus']
# model_names  = ['hiedsr', 'deephic']
metric_names = ['ssim', 'psnr', 'mse', 'snr', 'pcc', 'spc', 'gds']

# below is to record the gds values for each cell
gds_path = cell_lin+str(cell_no)+"_"+deg+"_"+str(sigma)+'_'+'normal'
gds_dir = str(root) + "/Metrics"
if not os.path.exists(gds_dir):
    os.makedirs(gds_dir, exist_ok = True)
record_gds = open(gds_dir+"/"+gds_path+".txt", "a")  # over-write the files
#record_gds.write("\n"+gds_path+":\n")

cell_text = []
for mod_nm in model_names:
    met_list = []
    record_gds.write("\n" + mod_nm + "\n")
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



