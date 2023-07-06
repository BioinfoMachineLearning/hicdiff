import torch.nn.functional as F
import matplotlib.pyplot as plt
import pdb
import numpy as np
import torch

class computeInsulation(torch.nn.Module):
    def __init__(self, window_radius=8, deriv_size=8):  # because our resolution 40000, so we make the window_radius smaller, original sizes are 10
        super(computeInsulation, self).__init__()
        self.window_radius = window_radius
        self.deriv_size  = deriv_size
        self.di_pool     = torch.nn.AvgPool2d(kernel_size=(2*window_radius+1), stride=1) # Hout = Hin-2*window_radius
        self.top_pool    = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)
        self.bottom_pool = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)
    
    def forward(self, x):
        iv     = self.di_pool(x)  # 4-dimensional
        iv     = torch.diagonal(iv, dim1=2, dim2=3)       
        iv     = torch.log2(iv/torch.mean(iv))
        top    = self.top_pool(iv[:,:,self.deriv_size:])
        bottom = self.bottom_pool(iv[:,:,:-self.deriv_size])  # 3-dimensional
        dv     = (top-bottom)
        left   = torch.cat([torch.zeros(dv.shape[0], dv.shape[1],2), dv], dim=2)
        right  = torch.cat([dv, torch.zeros(dv.shape[0], dv.shape[1],2)], dim=2)
        band   = ((left<0) == torch.ones_like(left)) * ((right>0) == torch.ones_like(right))
        band   = band[:,:,2:-2]
        boundaries = []
        for i in range(0, band.shape[0]):
            cur_bound = torch.where(band[i,0])[0]+self.window_radius+self.deriv_size
            boundaries.append(cur_bound)
        return iv, dv, boundaries

class InsulationLoss(torch.nn.Module):
    def __init__(self, window_radius=4, deriv_size=4): # here we modified the two parameters, both two origial parmaters are 10
        super(InsulationLoss, self).__init__()
        self.deriv_size     = deriv_size
        self.window_radius  = window_radius
        self.di_pool        = torch.nn.AvgPool2d(kernel_size=window_radius, stride=1) # Hout = (Hin - kerneal_size)/stride + 1 == Hin -3
        self.top_pool       = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1) # Hout = (Hin - kerneal_size)/stride + 1 == Hin -3
        self.bottom_pool    = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1) # Hout = (Hin - kerneal_size)/stride + 1 == Hin -3

    def indivInsulation(self, x):
        iv     = self.di_pool(x)
        iv     = torch.diagonal(iv, dim1=2, dim2=3)
        iv     = torch.log2(iv/torch.mean(iv))
        top    = self.top_pool(iv[:,:,self.deriv_size:])
        bottom = self.bottom_pool(iv[:,:,:-self.deriv_size])
        dv     = (top-bottom)
        return dv

    def forward(self, output, target):
        out_dv = self.indivInsulation(output)
        tar_dv = self.indivInsulation(target)
        loss   = F.mse_loss(tar_dv, out_dv)
        return loss
        



