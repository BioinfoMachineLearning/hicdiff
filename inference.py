import os
import sys
import pyrootutils
import torch


from src.model.hicedrn_Diff import hicedrn_Diff  # baseline models' modules
from src.hicdiff_condition import GaussianDiffusion as Gaussiandiff_cond  # baseline models' modules conditional Diff
from src.hicdiff import GaussianDiffusion as Gaussiandiff # baseline models's modules without conditional

from src.Utils import metrics_diff as vm
from src.Utils import metrics_cond as vm_cond
import argparse

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

def create_parser():
    parser = argparse.ArgumentParser(description = 'HiCDiff works for single-cell HI-C data denoising !!!')
    parser.add_argument('-u', '--unspervised', type = bool, default = True, help = 'True means you will use unsupervsed way to train your model, False indicates you will use supervised way to train your model')
    parser.add_argument('-b', '--batch_size', type = int, default = 64, help = 'Batch size for embeddings generation.')
    parser.add_argument('-e', '--epoch', type = int, default = 400, help = 'Number of epochs used for embeddings generation')
    parser.add_argument('-l', '--celline', type = str, default = "Human",
                        help = "Which cell line you want to choose for your dataset, default is 'Human', you should choose one name in ['Human', 'Dros']")
    parser.add_argument('-n', '--celln', type = int, default = 1,
                        help = "Cell number in the dataset you want to feed in you model")

    parser.add_argument('-s', '--sigma', type = float, default = 1,
                        help = f"The Gaussian noise level for the raw dataset, it should be equal or larger than 0.0 but not larger than 1.0, '1.0' means the largest noise added to datasets.")

    args = parser.parse_args()
    return args


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def Inference(batch_size = 64, cellNo = 1, cell_Line = "Human", sigma_t = 0.1, condition = None):
    # below is the cell information for test
    cell_lin = cell_Line  # 'Human' or 'Dros'
    cell_no = cellNo
    deg = 'deno'
    sigma = sigma_t
    image_channel = 1
    image_size = batch_size
    shedular = 'sigmoid'  # 'linear' or 'sigmoid'

    # below information is used for loading the pretrained diffusion models' paths
    cell_not = 1
    cell_lint = "Human"
    sigma_0 = 0.1
    model_type = condition
    timestep = 1000 if not model_type else 2000

    # below is used to load the diffusion model
    if not model_type:
        model_h = hicedrn_Diff(
            self_condition = True
        )
        diffusion_h = Gaussiandiff_cond(
            model_h,
            image_size = 64,
            timesteps = timestep,  # number of steps
            loss_type = 'l2',  # L1 or L2
            beta_schedule = shedular,
            auto_normalize = False
        ).to(device)
    else:
        model_h = hicedrn_Diff(
            self_condition = False
        )
        diffusion_h = Gaussiandiff(
            model_h,
            image_size = 64,
            timesteps = timestep,  # number of steps
            loss_type = 'l2',  # L1 or L2
            beta_schedule = shedular,
            auto_normalize = False
        ).to(device)

    # this is used for the stage to test.
    chro = 'test'

    # below two is used for pretrained models'
    if not model_type:

        # Load Our diffusion models' weight
        model_hicEdrn = diffusion_h.to(device)
        # for human1_0.1 population train
        file_path1 = str(root) + "/Model_Weights/" + "bestg_40000_c64_s64_" + cell_lint + str(cell_not) + "_HiCedrn_cond_l2_" + shedular[:3] + ".pytorch"
        model_hicEdrn.load_state_dict(torch.load(file_path1))
        model_hicEdrn.eval()

        # pay attention: how to pass the model, pass through sub_module of the conditional diffusion model to inference
        visionMetrics = vm_cond.VisionMetrics(image_channel = image_channel, image_size = image_size, timestep = timestep, type = model_type)
        predict = visionMetrics.getMetrics(model = model_hicEdrn.super_resolution, model_name = 'hicedrn_l2_' + shedular[:3], device = device, chro = chro, deg = deg, sigma = sigma, cellN = cell_no, cell_line = cell_lin)
    else:

        # Load Our diffusion models' weight
        model_hicEdrn = diffusion_h.to(device)
        file_path1 = str(root) + "/Model_Weights/" + "bestg_40000_c64_s64_" + cell_lint + str(cell_not) + "_HiCedrn_l2_" + shedular[:3] + "_trans.pytorch"
        model_hicEdrn.load_state_dict(torch.load(file_path1))
        model_hicEdrn.eval()

        # pay attention: how to pass for the model, pass through sub_module of the unconditional diffusion model to inference
        visionMetrics = vm.VisionMetrics(image_channel = image_channel, image_size = image_size, sehedule = shedular, timestep = timestep)
        predict = visionMetrics.getMetrics(model = model_hicEdrn.model, model_name = 'hicedrn_l2_' + shedular[:3], device = device, chro = chro, deg = deg, sigma = sigma, cellN = cell_no, cell_line = cell_lin)

    return predict

if __name__ == "__main__":
    args = create_parser()
    Out = Inference(batch_size = args.batch_size, cellNo = args.celln, cell_Line = args.celline, sigma_t = args.sigma, condition = args.unspervised)

    print("inference is done, and its result is saved to its corresponding file !!!")


