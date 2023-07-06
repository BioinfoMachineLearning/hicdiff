import sys
sys.path.append(".")

import os
import tmscoring
import glob
import subprocess
import shutil
import pdb
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
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
batch = 1

#below is the cell information for test
cell_lin = "Human"
cell_no = 2
deg = 'deno'
sigma = 0.1
image_channel = 1
image_size = 64
shedular = 'linear'  # 'linear' or 'sigmoid'
type = 1000   # 'condition' or 1000


# model = "hicedrn"
# dataroot = str(root)+"/Outputs_diff/"+model+"_l2_"+shedular[:3]+cell_lin+str(cell_no)+"_"+deg+"_"+str(sigma)+"_"+str(1000)
class Creatdataset(Dataset):
    def __init__(self, dataroot = None):
        self.target = np.load(dataroot+"/target.npy")
        self.data = np.load(dataroot+"/noisy.npy")
        self.samp = np.load(dataroot+"/predict.npy")
        self.info = np.load(dataroot+"/inds.npy")

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx], self.samp[idx], self.info[idx]

def create_dataLoader(model = "hicedrn", batch_s = 1, shedular = None, type = None):
    type = str(type) if type == 1000 else 'condition'
    #for chrom2
    # datadir = str(root)+"/Outputs_diff/"+model+"_l2_"+shedular[:3]+cell_lin+str(cell_no)+"_"+deg+"_"+str(sigma)+"_"+type
    #for chrom 6, 10, 12
    datadir = str(root)+"/Outputs_diff/"+model+"_l2_"+shedular[:3]+cell_lin+str(cell_no)+"_"+deg+"_"+str(sigma)+"_test_"+type
    dataset = Creatdataset(dataroot = datadir)
    Loader = DataLoader(dataset, batch_size = batch_s)
    return Loader


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

    # prepare the test datasets
    hicedrn_loader = create_dataLoader(model = "hicedrn", batch_s = batch, shedular = 'linear', type = 1000)
    hicedrn_bar = tqdm(hicedrn_loader)

    unet_loader = create_dataLoader(model = "unet", batch_s = batch, shedular = 'linear', type = 1000)
    unet_bar = tqdm(unet_loader)

    hicedrn_loader = create_dataLoader(model = "hicedrn", batch_s = batch, shedular = 'sigmoid', type = 'condition')
    hicedrn_bar_condi = tqdm(hicedrn_loader)

    unet_loader = create_dataLoader(model = "unet", batch_s = batch, shedular = 'sigmoid', type = 'condition')
    unet_bar_condi = tqdm(unet_loader)

    block = 0
    region = 0
    tt  = 0
    for s1, s2, s3, s4 in zip(hicedrn_bar, unet_bar, hicedrn_bar_condi, unet_bar_condi):
        if tt > 121:
            break

        data, target, hicdifflin_out, index = s1
        data, target, unetlin_out, _ = s2
        data, target, hicdiffsig_out, _ = s3
        data, target, unetsig_out, _ = s4

        data = inverse_data_transform('rescaled', data)
        target = inverse_data_transform('rescaled', target)
        hicdifflin_out = inverse_data_transform('rescaled', hicdifflin_out)
        unetlin_out = inverse_data_transform('rescaled', unetlin_out)
        hicdiffsig_out = inverse_data_transform('rescaled', hicdiffsig_out)
        unetsig_out = inverse_data_transform('rescaled', unetsig_out)

        len = data.shape[0]
        if index != chro:
            continue

        for ind in range(0, len):

            hicdifflin_o = hicdifflin_out[ind][0]
            hicdifflin_o[hicdifflin_o < 0.3] = hicdifflin_o[hicdifflin_o < 0.3]*3

            unetlin_o = unetlin_out[ind][0]
            unetlin_o[unetlin_o < 0.3] = unetlin_o[unetlin_o < 0.3]*3

            hicdiffsig_o = hicdiffsig_out[ind][0]
            hicdiffsig_o[hicdiffsig_o < 0.3] = hicdiffsig_o[hicdiffsig_o < 0.3]*3

            unetsig_o = unetsig_out[ind][0]
            unetsig_o[unetsig_o < 0.3] = unetsig_o[unetsig_o < 0.3]*3

            target_o = target[ind][0]
            target_o[target_o < 0.3] = target_o[target_o < 0.3] * 2


            if tt in range(0, 121) and tt % 4 == 0:  # the original is 6 pieces but current is 4 pieces
                if plotmap:
                    region_start = region*(2.56*4)  #
                    region_end = region_start+2.56
                    region = region + 1
                    fig, ax = plt.subplots(2, 5, figsize = (30, 10))

                    for i in range(0, 2):
                        for j in range(0, 5):
                            ax[i, j].set_xticks([])
                            ax[i, j].set_yticks([])

                    colorN = ['Blues', 'plasma', 'Reds', 'viridis']
                    ax[0, 0].imshow(hicdifflin_o, cmap = colorN[2])
                    ax[0, 0].set_title("HiCDiff_Lin", fontsize=20)
                    ax[0, 0].set_ylabel("Chro"+str(chro)+" "+"{:.2f}".format(region_start)+"-"+"{:.2f}".format(region_end), fontsize=20)

                    ax[0, 1].imshow(unetlin_o, cmap = colorN[2])
                    ax[0, 1].set_title("DDPM_Lin", fontsize = 20)

                    ax[0, 2].imshow(hicdiffsig_o, cmap = colorN[2])
                    ax[0, 2].set_title("HiCDiff_Sig", fontsize = 20)

                    ax[0, 3].imshow(unetsig_o, cmap = colorN[2])
                    ax[0, 3].set_title("DDPM_Sig", fontsize = 20)

                    ax[0, 4].imshow(target_o, cmap=colorN[2])
                    ax[0, 4].set_title("Target", fontsize=20)

                    ax[1, 0].imshow(hicdifflin_o[8:21, 8:21], cmap = colorN[2])
                    ax[1, 0].set_ylabel("Chro" + str(chro) + " " + "{:.2f}".format(region_start+0.32) + "-" + "{:.2f}".format(region_start+0.8), fontsize=20)
                    ax[1, 1].imshow(unetlin_o[8:21, 8:21], cmap = colorN[2])
                    ax[1, 2].imshow(hicdiffsig_o[8:21, 8:21], cmap = colorN[2])
                    ax[1, 3].imshow(unetsig_o[8:21, 8:21], cmap = colorN[2])
                    ax[1, 4].imshow(target_o[8:21, 8:21], cmap = colorN[2])

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


            if tt < 121:  # every 2.56MB for chromosomes
                block = block+1

                # training on multicom
                hicdifflin_const_name = "3D_Mod/Constraints/chro_" + str(chro) + "_hicdifflin_" + str(block-1) + "_"
                unetlin_const_name = "3D_Mod/Constraints/chro_" + str(chro) + "_unetlin_" + str(block-1) + "_"
                hicdiffsig_const_name = "3D_Mod/Constraints/chro_" + str(chro) + "_hicdiffsig_" + str(block-1) + "_"
                unetsig_const_name = "3D_Mod/Constraints/chro_" + str(chro) + "_unetsig_" + str(block-1) + "_"


                hicdifflin_constraints = open(hicdifflin_const_name, 'w')
                unetlin_constraints = open(unetlin_const_name, 'w')
                hicdiffsig_constraints = open(hicdiffsig_const_name, 'w')
                unetsig_constraints = open(unetsig_const_name, 'w')


                for i in range(0, target_o.shape[0]):
                    for j in range(i, target_o.shape[1]):

                        hicdifflin_constraints.write(str(i)+"\t"+str(j)+"\t"+str(hicdifflin_o[i,j].item())+"\n")
                        unetlin_constraints.write(str(i)+"\t"+str(j)+"\t"+str(unetlin_o[i,j].item())+"\n")
                        hicdiffsig_constraints.write(str(i)+"\t"+str(j)+"\t"+str(hicdiffsig_o[i,j].item())+"\n")
                        unetsig_constraints.write(str(i)+"\t"+str(j)+"\t"+str(unetsig_o[i,j].item())+"\n")


                hicdifflin_constraints.close()
                unetlin_constraints.close()
                hicdiffsig_constraints.close()
                unetsig_constraints.close()

            tt = tt + 1


# bellow should be modified
def buildParameters(chro,
                cell_line="Human",
                res=40000):
    if not os.path.exists("3D_Mod/Parameters"):
        os.makedirs("3D_Mod/Parameters", exist_ok = True)
    name = ["hicdifflin", "unetlin", "hicdiffsig", "unetsig"]
    constraints = []
    for na in name:
        cons  = glob.glob("3D_Mod/Constraints/chro_"+str(chro)+"_"+na+"_*")
        constraints.extend(cons)

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

# bellow should be modified
def runParams(chro):
    if not os.path.exists(JAR_LOCATION):
        subprocess.run("git clone https://github.com/BDM-Lab/3DMax.git other_tools", shell = True)

    PdbPath = "3D_Mod/output/chro_"+str(chro)+"_hicdifflin_0"+"_*.pdb"
    if not os.path.exists(PdbPath):
        print("\n=================generate the required pdb files ==================\n")
        name = ["hicdifflin", "unetlin", "hicdiffsig", "unetsig"]
        params = []
        for na in name:
            para = glob.glob("3D_Mod/Parameters/chro_"+str(chro)+"_"+na+"_*")
            params.extend(para)

        for par in params:
            subprocess.run("java -Xmx5000m -jar "+JAR_LOCATION+" "+par, shell=True)

def getSegmentTMScores(chro, position_index):   # should first to generate the corresponding pdbs
    target_strucs   = glob.glob("3D_Mod/output/chro_"+str(chro)+"_target_"+str(position_index)+"_*.pdb")

    hicdifflin_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_hicdifflin_" + str(position_index) + "_*.pdb")
    unetlin_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_unetlin_" + str(position_index) + "_*.pdb")
    hicdiffsig_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_hicdiffsig_" + str(position_index) + "_*.pdb")
    unetsig_strucs = glob.glob("3D_Mod/output/chro_" + str(chro) + "_unetsig_" + str(position_index) + "_*.pdb")


    struc_types      = [hicdifflin_strucs, unetlin_strucs, hicdiffsig_strucs, unetsig_strucs, target_strucs]

    struc_type_names = ['hicdifflin_strucs', 'unetlin_strucs', 'hicdiffsig_strucs', 'unetsig_strucs', 'target_strucs']

    # print("************************the data_struc length is {} and target lenght is {}".format(len(data_strucs), len(target_strucs)))
    internal_scores = {
                    'hicdifflin_strucs':[],
                    'unetlin_strucs':[],
                    'hicdiffsig_strucs':[],
                    'unetsig_strucs':[],
                    'target_strucs':[]
                    }


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


    relative_scores = {
                    'hicdifflin_strucs':[],
                    'unetlin_strucs':[],
                    'hicdiffsig_strucs':[],
                    'unetsig_strucs':[],
                    }


    # print("===============the data {} and hiedsr {} deephic {} hicarn {} hicsr {} hicplus {} unet {} target lenght is {}".format(len(data_strucs), len(hiedsr_strucs), len(deephic_strucs), len(hicarn_strucs), len(hicsr_strucs), len(hicplus_strucs), len(unet_strucs), len(target_strucs)))
    for struc_type, struc_type_name in zip(struc_types, struc_type_names):
        if len(target_strucs) == 1 or struc_type_name == 'target_strucs':
            continue

        if len(struc_type) == 1:
            continue

        for i, data_a in enumerate(struc_type):
            for j, data_b in enumerate(target_strucs):
                alignment = tmscoring.TMscoring(data_a, data_b)   # here has some problems
                alignment.optimise()
                # print("============ structure type is {} and id is {} {} ============".format(struc_type_name, i, j))
                indiv_tm  = alignment.tmscore(**alignment.get_current_values())

                indiv_tm = np.array(indiv_tm)
                indiv_tm = np.nan_to_num(indiv_tm)

                relative_scores[struc_type_name].append(indiv_tm)

    # print(internal_scores)
    return relative_scores, internal_scores

def getTMScores(chro):   # getSegmentTMScores() is called here

    internal_scores = {
        'hicdifflin_strucs': [],
        'unetlin_strucs': [],
        'hicdiffsig_strucs': [],
        'unetsig_strucs': [],
        'target_strucs': []
    }

    relative_scores = {
        'hicdifflin_strucs': [],
        'unetlin_strucs': [],
        'hicdiffsig_strucs': [],
        'unetsig_strucs': [],
    }


    getSampleNum = lambda a: a.split("_")[-2]  # to find the postion we want to compare

    name = ["hicdifflin", "unetlin", "hicdiffsig", "unetsig"]
    params = []
    for na in name:
        para = glob.glob("3D_Mod/Parameters/chro_" + str(chro) + "_" + na + "_*")
        params.extend(para)

    for position_index in list(map(getSampleNum, params)):
        temp_relative_scores, temp_internal_scores = getSegmentTMScores(chro, position_index)
        for key in temp_relative_scores.keys():
            relative_scores[key].extend(temp_relative_scores[key])
        for key in temp_internal_scores.keys():
            internal_scores[key].extend(temp_internal_scores[key])

   # record the built scores in .txt files
    if not os.path.exists("3D_Mod/Scores"):
        print("@@@@@@@@@@@@@@@@@@@ build the Score directory @@@@@@@@@@@@@@@@@@\n")
        os.makedirs("3D_Mod/Scores", exist_ok = True)
    record_scores = open("3D_Mod/Scores/chro_"+str(chro)+"diff.txt", "w")  # make the file name different from traditional one
    record_scores.write("INTERNAL SCORES\n")
    print("INTERNAL SCORES")
    for key in internal_scores.keys():
        print(key+":\t"+str(np.mean(internal_scores[key])))
        record_scores.write("\t"+key+":\t"+str(np.mean(internal_scores[key]))+"\n")

    print("RELATIVE SCORES")
    record_scores.write("RELATIVE SCORES\n")
    for key in relative_scores.keys():
        print(key+":\t"+str(np.mean(relative_scores[key])))
        record_scores.write("\t"+key+":\t"+str(np.mean(relative_scores[key]))+"\n")

    return relative_scores, internal_scores

def viewModels():  # not work, so can not be used here
    struc_index=0
    chro=2  # 2, 6, 12(the chrom best to plot)
    models = glob.glob("3D_Mod/output/chro_"+str(chro)+"_*_"+str(struc_index)+"_*.pdb")
    subprocess.run("pymol "+' '.join(models),  shell=True)

def parallelScatter(chrom):  # getRMScores() function is called here
    colorlist = ['royalblue', 'slategray', 'purple', 'darkorange']
    chros = [chrom]     # here we first test on chrom 2, all the test list chros = [2, 6, 10, 12]
    relative_data = []
    internal_data = []
    for chro in chros:
        relative, internal = getTMScores(chro)
        for key in relative.keys():
            if key == 'hicarn_strucs':
                continue
            elif key == "data_strucs":
                continue
            else:
                relative_data.append(relative[key])
        for key in internal.keys():
            if key == 'hicarn_strucs':
                continue
            elif key == "data_strucs":
                continue
            elif key == "target_strucs":
                continue
            else:
                internal_data.append(internal[key])
    # pdb.set_trace()
    temp = relative_data[0]
    relative_data[0] = relative_data[1]
    relative_data[1] = temp

    temp = relative_data[2]
    relative_data[2] = relative_data[3]
    relative_data[3] = temp

    #relative
    fig, ax = plt.subplots()   # plot by box with std
    medianprops = dict(linestyle = ':', color = "black", linewidth = 1.0)
    bp = ax.boxplot(relative_data, 
            positions=[1,2,3,4],    #for all chroms [1,2,4,5, 7,8, 10,11]
            patch_artist=True,
            medianprops=medianprops,
            showfliers=False)

    for b, box in enumerate(bp['boxes']):
        box.set(facecolor = colorlist[b])

    ax.set_xticks([2.5])  #4.5, 7.5, 10.5
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
        chros_all = [2]  # [2, 6, 10, 12] for all test chromsomes
    else:
        chros_all = [1, 2, 3, 4, 5, 6]
    for chro in chros_all: # 2, 6,10,12
        # convertChroToConstraints(chro, plotmap = False)
        # buildParameters(chro)  # build parameters
        # runParams(chro)   # build 3D pdb
        # getTMScores(chro)  # caculate the scores
        parallelScatter(chrom = chro)  # draw the resuls
    #viewModels()


