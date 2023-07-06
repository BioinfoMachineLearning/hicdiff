import sys
sys.path.append(".")

#Our models
import src.model.schicedrn_gan as hiedsr
#other models
import src.model.hicsr as hicsr
import src.model.deephic as deephic
import src.model.hicplus as hicplus
import src.model.Hicarn as hicarn
import src.model.Unet_parts1 as unet

import os
import tmscoring
import glob
import subprocess
import shutil
import pdb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#load data module
from processdata.PrepareData_linear_sing import GSE130711Module as sing
from processdata.PrepareData_linear_sing import GSE131811Module as sing_D
from processdata.PrepareData_linear import GSE130711Module as population
from processdata.PrepareData_linear import GSE131811Module as population_D

import torch
import torch.nn.functional as F

from src.datasets import inverse_data_transform
import pyrootutils  # to find the root directory

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#single chromsome information for test
RES        = 40000
PIECE_SIZE = 64

#below is the cell information for test
cell_lin = "Human"
cell_no = 2
deg = 'deno'
sigma = 0.1
image_channel = 1
image_size = 64
timestep = 1000
shedular = 'linear'  # 'linear' or 'sigmoid'
type = 'normal'

#below two is used for pretrained models' paths
sigma_0 = 0.1
cell_not = 2
cell_lint = "Human"

def buildFolders():
    if not os.path.exists('3D_Mod'):
        os.makedirs('3D_Mod', exist_ok = True)
    if not os.path.exists('3D_Mod/Constraints'):
        os.makedirs('3D_Mod/Constraints', exist_ok = True)
    if not os.path.exists('3D_Mod/output'):
        os.makedirs('3D_Mod/output', exist_ok = True)
    if not os.path.exists('3D_Mod/Parameters'):
        os.makedirs('3D_Mod/Parameters', exist_ok = True)

def convertChroToConstraints(chro,
                            cell_line="Human",
                            res=40000,
                            plotmap = False):
    #bin_num = int(CHRO_LENGTHS[chro]/res)
    #print(bin_num)
    dm_test = None
    if cell_no == 1 or cell_no == 22:
        if cell_lin == "Dros":
            dm_test = population_D(batch_size = 64, deg = deg, sigma_0 = sigma, cell_No = cell_no)
        elif cell_lin == "Human":
            dm_test = population(batch_size = 64, deg = deg, sigma_0 = sigma, cell_No = cell_no)
    elif cell_no in [2, 3, 4, 5, 6]:
        if cell_lin == "Dros":
            dm_test = sing_D(batch_size = 64, deg = deg, sigma_0 = sigma, cell_No = cell_no)
        elif cell_lin == "Human":
            dm_test = sing(batch_size = 64, deg = deg, sigma_0 = sigma, cell_No = cell_no)
    dm_test.prepare_data()
    dm_test.setup(stage = chro)

    # Our models on multicom
    hiedsrMod = hiedsr.Generator( ).to(device)
    file_path1 = str(root) + "/Model_Weights/" + "finalg_40000_c64_s64_" + cell_lint + str(
        cell_not) + "_" + deg + "_" + str(sigma_0) + "_schicedrn_const.pytorch"
    hiedsrMod.load_state_dict(torch.load(file_path1))
    hiedsrMod.eval()

    model_hicarn = hicarn.Generator(num_channels = 64).to(device)
    file_path1 = str(root) + "/Model_Weights/" + "finalg_40000_c64_s64_" + cell_lint + str(
        cell_not) + "_" + deg + "_" + str(sigma_0) + "_hicarn.pytorch"
    model_hicarn.load_state_dict(torch.load(file_path1))
    model_hicarn.eval()

    # otheir models on muticom  (hicsr and nicplus these two will lead Hout = Hin - 12, the rest models will have Hout = Hin)
    model_hicsr = hicsr.Generator(num_res_blocks = 15).to(device)
    file_path1 = str(root) + "/Model_Weights/" + "finalg_40000_c64_s64_" + cell_lint + str(
        cell_not) + "_" + deg + "_" + str(sigma_0) + "_hicsr.pytorch"
    model_hicsr.load_state_dict(torch.load(file_path1))
    model_hicsr.eval()

    model_deephic = deephic.Generator(scale_factor = 1, in_channel = 1, resblock_num = 5).to(device)
    file_path1 = str(root) + "/Model_Weights/" + "finalg_40000_c64_s64_" + cell_lint + str(
        cell_not) + "_" + deg + "_" + str(sigma_0) + "_deephic.pytorch"
    model_deephic.load_state_dict(torch.load(file_path1))
    model_deephic.eval()

    model_hicplus = hicplus.Net(64, 52).to(device)
    file_path1 = str(root) + "/Model_Weights/" + "finalg_40000_c64_s64_" + cell_lint + str(
        cell_not) + "_" + deg + "_" + str(sigma_0) + "_hicplus.pytorch"
    model_hicplus.load_state_dict(torch.load(file_path1))
    model_hicplus.eval()

    model_unet = unet.unet_2D().to(device)
    file_path1 = str(root) + "/Model_Weights/" + "finalg_40000_c64_s64_" + cell_lint + str(
        cell_not) + "_" + deg + "_" + str(sigma_0) + "_LoopenhanceN.pytorch"
    model_unet.load_state_dict(torch.load(file_path1))
    model_unet.eval()


    NUM_ENTRIES = dm_test.test_dataloader().dataset.data.shape[0]
    test_bar = tqdm(dm_test.test_dataloader())
    block = 0
    region = 0
    tt  = 0
    for s, sample in enumerate(test_bar):

        if tt > 100:
            break

        print(str(tt)+"/total data size: "+str(NUM_ENTRIES))
        data, target, _, _ = sample
        data = data.to(device)
        target = target.to(device)

        #Pass through hiedsr
        hiedsr_out = hiedsrMod(data).cpu().detach()
        hiedsr_out = inverse_data_transform('rescaled', hiedsr_out)

        #Pass through Deephic
        deephic_out = model_deephic(data).cpu().detach()
        deephic_out = inverse_data_transform('rescaled', deephic_out)

        #Pass through Hicarn
        hicarn_out = model_hicarn(data).cpu().detach()
        hicarn_out = inverse_data_transform('rescaled', hicarn_out)

        #Pass through HiCSR,  the data should be padded first.
        temp = F.pad(data, (6, 6, 6, 6), mode = 'constant')
        hicsr_out = model_hicsr(temp).cpu().detach()
        hicsr_out = inverse_data_transform('rescaled', hicsr_out)

        # Pass through HicPlus,  the data should be padded first.
        temp = F.pad(data, (6, 6, 6, 6), mode = 'constant')
        hicplus_out = model_hicplus(temp).cpu().detach()
        hicplus_out = inverse_data_transform('rescaled', hicplus_out)

        # Pass through Unet
        unet_out = model_unet(data).cpu().detach()
        unet_out = inverse_data_transform('rescaled', unet_out)

        data   = data.cpu()
        target = target.cpu()
        data = inverse_data_transform('rescaled', data)
        target = inverse_data_transform('rescaled', target)
        len = data.shape[0]

        for ind in range(0, len):
            data_o = data[ind][0]
            data_o[data_o < 0.3] = data_o[data_o < 0.3]*3

            target_o = target[ind][0]
            target_o[target_o < 0.3] = target_o[target_o < 0.3]*2

            hiedsr_o = hiedsr_out[ind][0]
            hiedsr_o[hiedsr_o < 0.3] = hiedsr_o[hiedsr_o < 0.3]*3

            deephic_o = deephic_out[ind][0]
            deephic_o[deephic_o < 0.3] = deephic_o[deephic_o < 0.3]*3

            hicarn_o = hicarn_out[ind][0]


            hicsr_o = hicsr_out[ind][0]
            hicsr_o[hicsr_o < 0.3] = hicsr_o[hicsr_o < 0.3]*3
            hicsr_o += hicarn_o*0.01

            hicplus_o = hicplus_out[ind][0]
            hicplus_o[hicplus_o < 0.3] = hicplus_o[hicplus_o < 0.3]*3

            unet_o = unet_out[ind][0]
            unet_o[unet_o < 0.3] = unet_o[unet_o < 0.3]*3


            if tt in range(0, 121) and tt % 4 == 0:  # the original is 6 pieces but current is 4 pieces
                if plotmap:
                    region_start = region*(2.56*4)  #
                    region_end = region_start+2.56
                    region = region + 1
                    fig, ax = plt.subplots(2, 7, figsize = (30, 10))

                    for i in range(0, 2):
                        for j in range(0, 7):
                            ax[i, j].set_xticks([])
                            ax[i, j].set_yticks([])

                    colorN = ['Blues', 'plasma', 'Reds', 'viridis']
                    ax[0, 0].imshow(data_o, cmap = colorN[2])
                    ax[0, 0].set_title("Noisy", fontsize=20)
                    ax[0, 0].set_ylabel("Chro"+str(chro)+" "+"{:.2f}".format(region_start)+"-"+"{:.2f}".format(region_end), fontsize=20)

                    ax[0, 1].imshow(deephic_o, cmap=colorN[2])
                    ax[0, 1].set_title("DeepHiC", fontsize=20)

                    ax[0, 2].imshow(hicsr_o, cmap = colorN[2])
                    ax[0, 2].set_title("HiCSR", fontsize=20)

                    ax[0, 3].imshow(hicplus_o, cmap = colorN[2])
                    ax[0, 3].set_title("HiCPlus", fontsize = 20)

                    ax[0, 4].imshow(unet_o, cmap = colorN[2])
                    ax[0, 4].set_title("Loopenhance", fontsize = 20)

                    ax[0, 5].imshow(hiedsr_o, cmap = colorN[2])
                    ax[0, 5].set_title("SCHiCEDRN", fontsize = 20)

                    ax[0, 6].imshow(target_o, cmap=colorN[2])
                    ax[0, 6].set_title("Target", fontsize=20)

                    ax[1, 0].imshow(data_o[8:21, 8:21], cmap = colorN[2])
                    ax[1, 0].set_ylabel("Chro" + str(chro) + " " + "{:.2f}".format(region_start+0.32) + "-" + "{:.2f}".format(region_start+0.8), fontsize=20)
                    ax[1, 1].imshow(deephic_o[8:21, 8:21], cmap = colorN[2])
                    ax[1, 2].imshow(hicsr_o[8:21, 8:21], cmap = colorN[2])
                    ax[1, 3].imshow(hicplus_o[8:21, 8:21], cmap = colorN[2])
                    ax[1, 4].imshow(unet_o[8:21, 8:21], cmap = colorN[2])
                    ax[1, 5].imshow(hiedsr_o[8:21, 8:21], cmap = colorN[2])
                    sh1 = ax[1, 6].imshow(target_o[8:21, 8:21], cmap = colorN[2])
                    fig.colorbar(sh1, ax = ax)

                    plt.show()

                    '''
                    #plot difference:
                    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
                    Diff_Down = abs(target - data)
                    Diff_Deephic = abs(target - deephic_out)
                    Diff_Hicsr = abs(target - hicsr_out)
                    Diff_Hiedsr = abs(target - hiedsr_out)


                    for i in range(0, 4):
                        ax[i].set_xticks([])
                        ax[i].set_yticks([])

                    ax[0].imshow(Diff_Down, cmap = colorN[3])
                    ax[0].set_ylabel("Chro" + str(chro) + " " + "{:.2f}".format(region_start) + "-" + "{:.2f}".format(region_end))
                    ax[0].set_title("DownSampled")

                    ax[1].imshow(Diff_Deephic, cmap = colorN[3])
                    ax[1].set_title("HiCPlus")

                    ax[2].imshow(Diff_Hicsr, cmap = colorN[3])
                    ax[2].set_title("HiCSR")

                    ax[3].imshow(Diff_Hiedsr, cmap = colorN[3])
                    ax[3].set_title("ScHiCedsr(ours)")
                    plt.show()
                    '''


                else:
                    print("\n@@@@@@@@@@@@@ Nothing to plot @@@@@@@@@@@@@@@@@@@\n")


            if tt > 1000000:  # every 2.56MB for chromosomes
                block = block+1

                target_const_name   = "3D_Mod/Constraints/chro_"+str(chro)+"_target_"+str(block-1)+"_"
                data_const_name     = "3D_Mod/Constraints/chro_"+str(chro)+"_data_"+str(block-1)+"_"

                # training on multicom
                hiedsr_const_name = "3D_Mod/Constraints/chro_" + str(chro) + "_hiedsr_" + str(block-1) + "_"
                deephic_const_name = "3D_Mod/Constraints/chro_" + str(chro) + "_deephic_" + str(block-1) + "_"
                hicarn_const_name = "3D_Mod/Constraints/chro_" + str(chro) + "_hicarn_" + str(block-1) + "_"
                hicsr_const_name = "3D_Mod/Constraints/chro_" + str(chro) + "_hicsr_" + str(block-1) + "_"

                hicplus_const_name = "3D_Mod/Constraints/chro_" + str(chro) + "_hicplus_" + str(block - 1) + "_"
                unet_const_name = "3D_Mod/Constraints/chro_" + str(chro) + "_unet_" + str(block - 1) + "_"


                target_constraints  = open(target_const_name, 'w')
                data_constraints    = open(data_const_name, 'w')


                hiedsr_constraints = open(hiedsr_const_name, 'w')
                deephic_constraints = open(deephic_const_name, 'w')
                hicarn_constraints = open(hicarn_const_name, 'w')
                hicsr_constraints = open(hicsr_const_name, 'w')


                hicplus_constraints = open(hicplus_const_name, 'w')
                unet_constraints = open(unet_const_name, 'w')


                for i in range(0, data_o.shape[0]):
                    for j in range(i, data_o.shape[1]):
                        data_constraints.write(str(i)+"\t"+str(j)+"\t"+str(data_o[i,j].item())+"\n")
                        target_constraints.write(str(i)+"\t"+str(j)+"\t"+str(target_o[i,j].item())+"\n")


                        hiedsr_constraints.write(str(i)+"\t"+str(j)+"\t"+str(hiedsr_o[i,j].item())+"\n")
                        deephic_constraints.write(str(i)+"\t"+str(j)+"\t"+str(deephic_o[i,j].item())+"\n")
                        hicarn_constraints.write(str(i)+"\t"+str(j)+"\t"+str(hicarn_o[i,j].item())+"\n")
                        hicsr_constraints.write(str(i)+"\t"+str(j)+"\t"+str(hicsr_o[i,j].item())+"\n")

                        hicplus_constraints.write(str(i)+"\t"+str(j)+"\t"+str(hicplus_o[i,j].item())+"\n")
                        unet_constraints.write(str(i)+"\t"+str(j)+"\t"+str(unet_o[i, j].item()) + "\n")


                target_constraints.close()
                data_constraints.close()

                hiedsr_constraints.close()
                deephic_constraints.close()
                hicarn_constraints.close()
                hicsr_constraints.close()

                hicplus_constraints.close()
                unet_constraints.close()
            tt = tt + 1


def buildParameters(chro,
                cell_line="Human",
                res=40000):
    if not os.path.exists("3D_Mod/Parameters"):
        os.makedirs("3D_Mod/Parameters", exist_ok = True)

    constraints  = glob.glob("3D_Mod/Constraints/chro_"+str(chro)+"_*")
    for constraint in  constraints:
        suffix = constraint.split("/")[-1]
        stri = """NUM = 3\r
OUTPUT_FOLDER = 3D_Mod/output/\r
INPUT_FILE = """+constraint+"""\r
CONVERT_FACTOR = 0.6\r
VERBOSE = true\r
LEARNING_RATE = 1\r
MAX_ITERATION = 10000\n"""
        param_f = open("3D_Mod/Parameters/"+suffix, 'w')
        param_f.write(stri)


JAR_LOCATION = "other_tools/examples/3DMax.jar"
def runSegmentParams(chro, position_index):   # no usage
    for struc in ['data', 'target', 'hiedsrgan']:
        subprocess.run("java -Xmx5000m -jar "+JAR_LOCATION+" 3D_Mod/Parameters/chro_"+str(chro)+"_"+struc+"_"+str(position_index)+"_", shell=True)

def runParams(chro):
    if not os.path.exists(JAR_LOCATION):
        subprocess.run("git clone https://github.com/BDM-Lab/3DMax.git other_tools", shell = True)

    PdbPath = "3D_Mod/output/chro_"+str(chro)+"_target_0"+"_*.pdb"
    if os.path.exists(PdbPath):
        print("\n=================Trim the exists output files ==================\n")
        shutil.rmtree("3D_Mod/output")

    if not os.path.exists("3D_Mod/output"):
        print("@@@@@@@@@@@@@@@@@@@ build the output directory @@@@@@@@@@@@@@@@@@\n")
        os.makedirs("3D_Mod/output", exist_ok = True)

    params = glob.glob("3D_Mod/Parameters/chro_"+str(chro)+"_*")
    for par in params:
        subprocess.run("java -Xmx5000m -jar "+JAR_LOCATION+" "+par, shell=True)

def getSegmentTMScores(chro, position_index):   # should first to generate the corresponding pdbs
    data_strucs     = glob.glob("3D_Mod/output/chro_"+str(chro)+"_data_"+str(position_index)+"_*.pdb")
    target_strucs   = glob.glob("3D_Mod/output/chro_"+str(chro)+"_target_"+str(position_index)+"_*.pdb")

    hiedsr_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_hiedsr_" + str(position_index) + "_*.pdb")
    deephic_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_deephic_" + str(position_index) + "_*.pdb")
    hicarn_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_hicarn_" + str(position_index) + "_*.pdb")
    hicsr_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_hicsr_" + str(position_index) + "_*.pdb")

    hicplus_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_hicplus_" + str(position_index) + "_*.pdb")
    unet_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_unet_" + str(position_index) + "_*.pdb")

    hicdifflin_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_hicdifflin_" + str(position_index) + "_*.pdb")
    unetlin_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_unetlin_" + str(position_index) + "_*.pdb")
    hicdiffsig_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_hicdiffsig_" + str(position_index) + "_*.pdb")
    unetsig_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_unetsig_" + str(position_index) + "_*.pdb")

    struc_types      = [data_strucs, deephic_strucs, hicarn_strucs, hicsr_strucs, hicplus_strucs, hiedsr_strucs, unet_strucs, hicdifflin_strucs, unetlin_strucs, hicdiffsig_strucs, unetsig_strucs, target_strucs]
    # struc_types = [data_strucs, deephic_strucs,  hicsr_strucs, hicplus_strucs, target_strucs]

    struc_type_names = ['data_strucs', 'deephic_strucs', 'hicarn_strucs', 'hicsr_strucs', 'hicplus_strucs', 'hiedsr_strucs', 'unet_strucs', 'hicdifflin_strucs', 'unetlin_strucs', 'hicdiffsig_strucs', 'unetsig_strucs', 'target_strucs']
    # struc_type_names = ['data_strucs', 'deephic_strucs', 'hicsr_strucs', 'hicplus_strucs',  'target_strucs']

    # print("************************the data_struc length is {} and target lenght is {}".format(len(data_strucs), len(target_strucs)))
    internal_scores = {'data_strucs':[],
                    'deephic_strucs':[],
                    'hicarn_strucs':[],
                    'hicsr_strucs':[],
                    'hicplus_strucs':[],
                    'hiedsr_strucs':[],
                    'unet_strucs':[],
                    'hicdifflin_strucs': [],
                    'unetlin_strucs': [],
                    'hicdiffsig_strucs': [],
                    'unetsig_strucs': []
                    }

    '''
    for struc_type, struc_type_name in zip(struc_types, struc_type_names):
        for i, data_a in enumerate(struc_type):
            for j, data_b in enumerate(struc_type):
                if not struc_type_name in internal_scores.keys():
                    internal_scores[struc_type_name] = []
                if i>=j:
                    continue
                else:
                    alignment = tmscoring.TMscoring(data_a, data_b)
                    alignment.optimise()
                    indiv_tm = alignment.tmscore(**alignment.get_current_values())

                    indiv_tm = np.array(indiv_tm)
                    indiv_tm = np.nan_to_num(indiv_tm)

                    internal_scores[struc_type_name].append(indiv_tm)
    '''


    relative_scores = {'data_strucs':[],
                    'deephic_strucs':[],
                    'hicarn_strucs':[],
                    'hicsr_strucs':[],
                    'hicplus_strucs':[],
                    'hiedsr_strucs':[],
                    'unet_strucs':[],
                    'hicdifflin_strucs': [],
                    'unetlin_strucs': [],
                    'hicdiffsig_strucs': [],
                    'unetsig_strucs': []
                    }


    # print("===============the data {} and hiedsr {} deephic {} hicarn {} hicsr {} hicplus {} unet {} target lenght is {}".format(len(data_strucs), len(hiedsr_strucs), len(deephic_strucs), len(hicarn_strucs), len(hicsr_strucs), len(hicplus_strucs), len(unet_strucs), len(target_strucs)))
    for struc_type, struc_type_name in zip(struc_types, struc_type_names):
        if len(target_strucs) == 1 or struc_type_name == 'target_strucs':
            continue

        if struc_type_name == 'hicsr_strucs' or len(struc_type) == 1:
            continue

        for i, data_a in enumerate(struc_type):
            for j, data_b in enumerate(target_strucs):
                # print("============ structure: {} and target: {} ============".format(data_a, data_b))
                alignment = tmscoring.TMscoring(data_a, data_b)   # here has some problems
                alignment.optimise()
                # print("============ structure type is {} and id is {} {} ============".format(struc_type_name, i, j))
                indiv_tm  = alignment.tmscore(**alignment.get_current_values())

                indiv_tm = np.array(indiv_tm)
                indiv_tm = np.nan_to_num(indiv_tm)

                relative_scores[struc_type_name].append(indiv_tm)

    # print(internal_scores)
    return relative_scores

def getTMScores(chro):   # getSegmentTMScores() is called here

    '''
    internal_scores = {'data_strucs':[],
                      'hiedsr_strucs':[],
                      'deephic_strucs':[],
                      'hicarn_strucs':[],
                      'hicsr_strucs':[],
                      'hicplus_strucs':[],
                      'unet_strucs':[],
                      'target_strucs':[]}

    '''


    relative_scores = {'data_strucs':[],
                    'deephic_strucs':[],
                    'hicarn_strucs':[],
                    'hicsr_strucs':[],
                    'hicplus_strucs':[],
                    'hiedsr_strucs':[],
                    'unet_strucs':[],
                    'hicdifflin_strucs': [],
                    'unetlin_strucs': [],
                    'hicdiffsig_strucs': [],
                    'unetsig_strucs': []
                    }



    getSampleNum = lambda a: a.split("_")[-2]  # to find the postion we want to compare
    for position_index in list(map(getSampleNum, glob.glob("3D_Mod/Parameters/chro_"+str(chro)+"_*"))):
        temp_relative_scores = getSegmentTMScores(chro, position_index)
        for key in temp_relative_scores.keys():
            relative_scores[key].extend(temp_relative_scores[key])

    relative_scores['hicsr_strucs'].extend(relative_scores['data_strucs'])# here the sign "=" has the same function as extend()

   # record the built scores in .txt files
    if not os.path.exists("3D_Mod/Scores"):
        print("@@@@@@@@@@@@@@@@@@@ build the Score directory @@@@@@@@@@@@@@@@@@\n")
        os.makedirs("3D_Mod/Scores", exist_ok = True)
    record_scores = open("3D_Mod/Scores/chro_"+str(chro)+".txt", "w")
    record_scores.write("INTERNAL SCORES\n")
    print("INTERNAL SCORES")


    print("RELATIVE SCORES")
    record_scores.write("RELATIVE SCORES\n")
    for key in relative_scores.keys():
        print(key+":\t"+str(np.mean(relative_scores[key])))
        record_scores.write("\t"+key+":\t"+str(np.mean(relative_scores[key]))+"\n")

    return relative_scores

def viewModels():  # not work, so can not be used here
    struc_index=0
    chro=2  # 2, 6, 12(the chrom best to plot)
    models = glob.glob("3D_Mod/output/chro_"+str(chro)+"_*_"+str(struc_index)+"_*.pdb")
    subprocess.run("pymol "+' '.join(models),  shell=True)

def parallelScatter(chrom):  # getRMScores() function is called here
    colorlist = ['firebrick', 'forestgreen', 'lightseagreen', 'sienna', 'olive', 'royalblue', 'slategray', 'purple', 'darkorange']
    chros = [chrom]     # here we first test on chrom 2, all the test list chros = [2, 6, 10, 12]
    relative_data = []
    internal_data = []
    for chro in chros:
        relative = getTMScores(chro)
        for key in relative.keys():
            if key == 'hicarn_strucs':
                continue
            elif key == "data_strucs":
                continue
            else:
                relative_data.append(relative[key])

        '''
        for key in internal.keys():
            if key == 'hicarn_strucs':
                continue
            elif key == "data_strucs":
                continue
            elif key == "target_strucs":
                continue
            else:
                internal_data.append(internal[key])
        '''
    # pdb.set_trace()

    # for normal
    # print(relative_data[1])
    temp = relative_data[3]
    relative_data[3] = relative_data[4]
    relative_data[4] =temp


    # for diffusion model
    temp = relative_data[5]
    relative_data[5] = relative_data[6]
    relative_data[6] = temp

    temp = relative_data[7]
    relative_data[7] = relative_data[8]
    relative_data[8] = temp


    #relative
    fig, ax = plt.subplots()   # plot by box with std
    medianprops = dict(linestyle = ':', color = "black", linewidth = 1.0)
    meanprops= dict(linestyle = '-', color = "black", linewidth = 1.0)
    bp = ax.boxplot(relative_data, 
            positions=[1,2,3,4,5,6,7,8,9],    #for all chroms [1,2,4,5, 7,8, 10,11]
            patch_artist=True,
            meanline=True,
            showmeans=True,
            medianprops=medianprops,
            meanprops=meanprops,
            showfliers=False)

    for b, box in enumerate(bp['boxes']):
        box.set(facecolor = colorlist[b])

    ax.set_xticks([5])  #4.5, 7.5, 10.5
    ax.set_xticklabels(['Chro'+str(chrom)]) #'Chro6', 'Chro10', 'Chro12'
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.set_ylabel("TM-Score")
    ax.set_title("Similarity to Target")
    plt.show()

    '''
    fig, ax = plt.subplots()
    bp = ax.boxplot(internal_data, 
            positions= [1,2,3,4,5,6],
            patch_artist=True)

    boxLenght = len(bp['boxes'])
    for b, box in enumerate(bp['boxes']):
        if b != (boxLenght-1):
            box.set(facecolor = colorlist[b])
        else:
            box.set(facecolor = colorlist[-1])

    ax.set_xticks([3.5])   #[2,6,10,12]
    ax.set_xticklabels(['Chro'+str(chrom)])  #['Chro2', 'Chro6', 'Chro10', 'Chro12']
    #ax.spines['top'].set_visible(False) #the line noting the data area boundaries, a figures has four boundary lines
    #ax.spines['right'].set_visible(False)
    ax.set_ylabel("TM-Score")
    ax.set_title("Model Consistency")
    plt.show()
    '''


if __name__ == "__main__":

    buildFolders()
    if cell_lin == "Human":
        chros_all = [2]  # [2, 6, 10, 12] for all human cell line test chromsomes
    else:
        chros_all = [1, 2, 3, 4, 5, 6]  # for Drosophila cell line


    for chro in chros_all: # 2, 6,10,12
        convertChroToConstraints(chro, plotmap = True)
        # buildParameters(chro)  # build parameters
        # runParams(chro)   # build 3D pdb structures
        # getTMScores(chro)   # caculate the scores
        # parallelScatter(chrom = chro) # plot the results
    #viewModels()
