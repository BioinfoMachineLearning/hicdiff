import matplotlib.pyplot as plt
import glob
import torch
import pdb
import yaml
import sys
sys.path.append(".")
from Models.VAE_Module import VAE_Model
from torch.nn import functional as F

class VaeLoss(torch.nn.Module):
    def __init__(self, yaml_path, weight_path):
        super(VaeLoss, self).__init__()
        op      = open(yaml_path)
        hparams = yaml.safe_load(op)
        model   = VAE_Model(
                condensed_latent=hparams['condensed_latent'],
                gamma=hparams['gamma'],
                kld_weight=hparams['kld_weight'],
                latent_dim=hparams['latent_dim'],
                lr=hparams['lr'],
                pre_latent=hparams['pre_latent'])
        self.pretrained_model = model.load_from_checkpoint(weight_path)
        self.hparams = hparams

    def forward(self, output, target):
        latent_output, mu_out, var_out       = self.pretrained_model.get_z(output)
        latent_target, mu_target, var_target = self.pretrained_model.get_z(target)
        loss          = F.mse_loss(mu_target, mu_out)
        return loss

