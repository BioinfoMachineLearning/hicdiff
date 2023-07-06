import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()  # because the whole vgg16 model  has 31 nodes
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()  # nn.MSELoss() is the L2 loss function, nn.L1Loss() is the L1 loss function
        self.tv_loss = TVLoss()

    def forward(self, out_images, target_images):
        # Perception Loss
        out_feat = self.loss_network(out_images.repeat([1, 3, 1, 1]))  # THis is because of the vgg first layer have 3 channels.
        target_feat = self.loss_network(target_images.repeat([1, 3, 1, 1]))
        perception_loss = self.mse_loss(out_feat.reshape(out_feat.size(0), -1),
                                        target_feat.reshape(target_feat.size(0), -1))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        tv_loss = self.tv_loss(out_images)
        total = image_loss + 0.001 * perception_loss + 2e-8 * tv_loss
        return image_loss, perception_loss, tv_loss, total


class TVLoss(nn.Module):  # The loss function is computed by its definition
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        b, c, h, w = x.shape
        # count_h = self.tensor_size(x[:, :, 1:, :])
        # count_w = self.tensor_size(x[:, :, :, 1:])
        count_h = x[:, :, 1:, :].size()[1] * x[:, :, 1:, :].size()[2] * x[:, :, 1:, :].size()[3]
        count_w = x[:, :, :, 1:].size()[1] * x[:, :, :, 1:].size()[2] * x[:, :, :, 1:].size()[3]
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w-1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / b

'''
    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
'''