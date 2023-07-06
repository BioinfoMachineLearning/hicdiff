import os
import sys
import torch
import pyrootutils
import torch
from torch import optim
from tqdm import tqdm

import sys
sys.path.append('../')

from src.model.hicedrn_Diff import hicedrn_Diff  # baseline models' modules
from src.hicdiff_condition import GaussianDiffusion  # baseline models' modules conditional Diff
import wandb  # the logger

from processdata.PrepareData_linear import GSE130711Module  # the datasets
from src.datasets import inverse_data_transform
from src.Utils.loss.SSIM import ssim
from math import log10

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

class HiCedrn_Diff:
    def __init__(self, epoch = 500, timestep = 1000, cell_Line = 'Human',  cellNo = 1, res = 40000, batch_size = 64, piece_s = 64, sigma = 0.1, deg='deno'):
        # initialize the parameters that will be used during fit model
        self.epoch = epoch
        self.cell_Line = cell_Line
        self.cellNo = cellNo
        self.res = res
        self.chunk = piece_s
        self.sigma = sigma
        self.deg = deg

        # whether using GPU for training
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')
        self.device = device

        # experiment tracker
        wandb.init(project='HiCDiff')
        wandb.run.name = f'hicedrn_Diff_conditional_L2_linear cell_{cellNo}'
        wandb.run.save()   # get the random run name in my script by Call wandb.run.save(), then get the name with wandb.run.name .

        # out_dir: directory storing checkpoint files and parameters for saving to the our_dir
        dir_name = 'Model_Weights'
        self.out_dir = os.path.join(root, dir_name)
        #self.out_dirM = os.path.join(root, "Metrics")
        os.makedirs(self.out_dir, exist_ok=True)  # makedirs will make all the directories on the path if not exist.
        #os.makedirs(self.out_dirM, exist_ok=True)

        # prepare training and valid dataset
        DataModule = GSE130711Module(batch_size=batch_size, res=res, cell_line=cell_Line, cell_No=cellNo, sigma_0=self.sigma, deg=self.deg)
        DataModule.prepare_data()
        DataModule.setup(stage='fit')

        self.train_loader = DataModule.train_dataloader()
        self.valid_loader = DataModule.val_dataloader()

        # load the network for different models
        model = hicedrn_Diff(
            self_condition = True
        )
        self.diffusion = GaussianDiffusion(
            model,
            image_size=piece_s,
            timesteps=timestep,  # number of steps
            loss_type='l2',  # L1 or L2
            beta_schedule = 'linear',
            auto_normalize = False
        ).to(device)

    def fit_model(self):
        # optimizer
        optimizer = optim.Adam(self.diffusion.parameters(), lr=2e-5)

        best_ssim = 0
        best_loss = 10000
        for epoch in range(1, self.epoch + 1):
            self.diffusion.train()
            run_result = {'nsamples': 0, 'loss': 0}

            train_bar = tqdm(self.train_loader)
            for data, target, _, info in train_bar:  # data is the pure image without noise
                batch_size = data.shape[0]
                run_result['nsamples'] += batch_size
                data = data.to(self.device)
                target = target.to(self.device)
                x = [data, target]

                loss = self.diffusion(x)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                run_result['loss'] +=loss.item() * batch_size
                train_bar.set_description(desc=f"[{epoch}/{self.epoch}] training Loss: {run_result['loss'] / run_result['nsamples']:.6f}")

            train_loss = run_result['loss'] / run_result['nsamples']


            valid_result = {'nsamples': 0, 'loss': 0}
            self.diffusion.eval()
            valid_bar = tqdm(self.valid_loader)
            batch_id = 0
            with torch.no_grad():
                for data, target, _, info in valid_bar:   # data is the pure image without noise
                    batch_size = data.shape[0]
                    valid_result['nsamples'] += batch_size
                    data = data.to(self.device)
                    target = target.to(self.device)
                    x = [data, target]

                    loss = self.diffusion(x)

                    '''
                    #sample_out = self.diffusion.valuate(x)
                    if batch_id == 0:
                        out = self.diffusion.super_resolution(data)
                        print(f'the data shape is {data.shape} predicted results shape is {out.shape}.')
                        out = inverse_data_transform('rescaled', out)
                        hr = inverse_data_transform('rescaled', target)
                        batch_ssim = ssim(out, hr)
                        batch_mse = ((out - hr) ** 2).mean()
                        batch_psnr = 10 * log10(1 / (batch_mse))
                        print(f'the ssim is {batch_ssim} and the psnr is {batch_psnr}\n')
                    batch_id += 1
                    '''

                    valid_result['loss'] += loss.item() * batch_size
                    valid_bar.set_description(
                        desc=f"[{epoch}/{self.epoch}] Validation Loss: {valid_result['loss'] / valid_result['nsamples']:.6f}")

                valid_loss = valid_result['loss'] / valid_result['nsamples']

                # now_ssim = batch_ssim
                now_loss = valid_loss
                if now_loss < best_loss:
                    best_loss = now_loss
                    print(f'Now, Best ssim is {best_loss:.6f}')
                    best_ckpt_file = f'bestg_{self.res}_c{self.chunk}_s{self.chunk}_{self.cell_Line}{self.cellNo}_HiCedrn_cond_l2_lin.pytorch'
                    torch.save(self.diffusion.state_dict(), os.path.join(self.out_dir, best_ckpt_file))
                wandb.log({"Epoch": epoch, 'train/loss':train_loss,'valid/loss': valid_loss})

        final_ckpt_file = f'finalg_{self.res}_c{self.chunk}_s{self.chunk}_{self.cell_Line}{self.cellNo}_HiCedrn_cond_l2_lin.pytorch'
        torch.save(self.diffusion.state_dict(), os.path.join(self.out_dir, final_ckpt_file))

if __name__ == "__main__":
    train_model = HiCedrn_Diff(epoch = 400, batch_size = 64, cellNo = 1, sigma = 0.1, deg = 'deno')
    train_model.fit_model()
    print("\n\nTraining hiedsr_diff is done!!!\n")
