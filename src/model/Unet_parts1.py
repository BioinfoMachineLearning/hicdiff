#The Unet-parts component
#with adjustments to architecture
#the stencil code comes from

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

class unetConv2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, is_batchnorm, mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if is_batchnorm:
            self.double_conv = nn.Sequential(  # this every per Conv2d has Hout = Hin
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:   # if False, then the double_Conv exludes the BatchNorm2e()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1, bias = False),
                nn.ReLU(inplace = True),
                nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
                nn.ReLU(inplace = True)
            )
        self.init_params()

    def forward(self, x):
        out = self.double_conv(x)
        return out

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.BatchNorm2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.ReLU):
                init_weights(module, init_type='kaiming')

class unetUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, is_deconv):
        super().__init__()
        self.conv = unetConv2(in_channels, out_channels, True)

        if is_deconv:     # every ConvTranspose2d has Hout = 2 * Hin, and every DoubleConv has Hout = Hin
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

        else:   # every Upsample has Hout = 2 * Hin, and every DoubleConv has Hout = Hin
            # self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)  # this equals to nn.upsamplingBilinear2d(scale_factor = 2)
            self.up = nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True),
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
                nn.ReLU(inplace = True)
            )

        self.init_params()

    def forward(self, x1, x2):
        x2 = self.up(x2)
        if x1.size()[2] >= x2.size()[2]:
            # input is BCHW
            diffY = x1.size()[2] - x2.size()[2] # in fact x2.size() == x1.size() so the diffy == 0;
            diffX = x1.size()[3] - x2.size()[3] # in fact x2.size() == x1.size() so the diffx == 0;
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]) # in fact there is no padding
        else:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim = 1)  # Cat here will concate the two arguments by the channnels, so that the 2 * out_channels == in_channels;
        return self.conv(x)

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.BatchNorm2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.ReLU):
                init_weights(module, init_type='kaiming')


class unet_2D(nn.Module):

    def __init__(self,
                 feature_scale = 4,
                 n_classes = 1,
                 is_deconv = True,
                 in_channels = 1,
                 is_batchnorm = True,
                 ):
        super().__init__()

        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]  # have 4-layer downsample and upsample
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size = 2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size = 1)
        self.init_params()

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1) # F.sigmoid(final)
        return F.sigmoid(final) 

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.BatchNorm2d):
                init_weights(module, init_type = 'kaiming')
            elif isinstance(module, nn.ReLU):
                init_weights(module, init_type='kaiming')
