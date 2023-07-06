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
from src.hicdiff import GaussianDiffusion  # baseline models' modules
import wandb  # the logger

from processdata.PrepareData_pure_noise import GSE130711Module  # the datasets

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

class HiCedrn_Diff:
    def __init__(self, epoch = 1000, cell_Line = 'Human', cellNo = 1, res = 40000, batch_size = 64, piece_s = 64, timestep = 1000):
        # initialize the parameters that will be used during fit model
        self.epoch = epoch
        self.cell_Line = cell_Line
        self.cellNo = cellNo
        self.res = res
        self.chunk = piece_s

        # whether using GPU for training
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')
        self.device = device

        # experiment tracker
        wandb.init(project='HiCDiff')
        wandb.run.name = f'hicedrn_diff'
        wandb.run.save()   # get the random run name in my script by Call wandb.run.save(), then get the name with wandb.run.name .

        # out_dir: directory storing checkpoint files and parameters for saving to the our_dir
        dir_name = 'Model_Weights'
        self.out_dir = os.path.join(root, dir_name)
        #self.out_dirM = os.path.join(root, "Metrics")
        os.makedirs(self.out_dir, exist_ok=True)  # makedirs will make all the directories on the path if not exist.
        #os.makedirs(self.out_dirM, exist_ok=True)

        # prepare training and valid dataset
        DataModule = GSE130711Module(batch_size=batch_size, res=res, piece_size=piece_s, cell_line=cell_Line,
                                     cell_No=cellNo)
        DataModule.prepare_data()
        DataModule.setup(stage='fit')

        self.train_loader = DataModule.train_dataloader()
        self.valid_loader = DataModule.val_dataloader()

        # load the network for different models
        model = hicedrn_Diff()
        self.diffusion = GaussianDiffusion(
            model,
            image_size=piece_s,
            timesteps=timestep,  # number of steps
            loss_type='l1',  # L1 or L2
            auto_normalize = False
        ).to(device)

    def fit_model(self):
        # optimizer
        optimizer = optim.Adam(self.diffusion.parameters(), lr=2e-5)

        best_loss = 10000
        for epoch in range(1, self.epoch + 1):
            self.diffusion.train()
            run_result = {'nsamples': 0, 'loss': 0}

            train_bar = tqdm(self.train_loader)
            for _, data, info in train_bar:  # data is the pure image without noise
                batch_size = data.shape[0]
                run_result['nsamples'] += batch_size
                x = data.to(self.device)
                loss = self.diffusion(x)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                run_result['loss'] +=loss.item() * batch_size
                train_bar.set_description(desc=f"[{epoch}/{self.epoch}] training Loss: {run_result['loss'] / run_result['nsamples']:.6f}")

            train_loss = run_result['loss'] / run_result['nsamples']

            final_ckpt_g = f'finalg_{self.res}_c{self.chunk}_s{self.chunk}_{self.cell_Line}{self.cellNo}_HiCedrn.pytorch'
            torch.save({'epoch': epoch,
                        'model_state_dict': self.diffusion.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, os.path.join(self.out_dir, final_ckpt_g))

            valid_result = {'nsamples': 0, 'loss': 0}
            self.diffusion.eval()
            valid_bar = tqdm(self.valid_loader)
            with torch.no_grad():
                for _, data, info in valid_bar:   # data is the pure image without noise
                    batch_size = data.shape[0]
                    valid_result['nsamples'] += batch_size
                    x = data.to(self.device)
                    loss = self.diffusion(x)
                    #sample_out = self.diffusion.sample(x)

                    valid_result['loss'] += loss.item() * batch_size
                    valid_bar.set_description(
                        desc=f"[{epoch}/{self.epoch}] Validation Loss: {valid_result['loss'] / valid_result['nsamples']:.6f}")

                valid_loss = valid_result['loss'] / valid_result['nsamples']

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    print(f'Now, Best ssim is {best_loss:.6f}')
                    best_ckpt_file = f'bestg_{self.res}_c{self.chunk}_s{self.chunk}_{self.cell_Line}{self.cellNo}_HiCedrn.pytorch'
                    torch.save(self.diffusion.state_dict(), os.path.join(self.out_dir, best_ckpt_file))
                wandb.log({"Epoch": epoch, 'train/loss':train_loss,'valid/loss': valid_loss})


if __name__ == "__main__":
    train_model = HiCedrn_Diff(epoch = 500)
    train_model.fit_model()
    print("\n\nTraining hiedsr_diff is done!!!\n")
