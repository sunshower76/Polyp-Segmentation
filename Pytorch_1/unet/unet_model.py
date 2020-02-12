""" Full assembly of the parts to form the complete network """
import torch.nn.functional as F
from torch import nn
from .unet_parts import *

'''
reference

Paper
U-net : https://arxiv.org/pdf/1505.04597.pdf
CBAM : https://arxiv.org/abs/1807.06521
Attention-Unet : https://www.researchgate.net/publication/328682314_Urban_Land_Use_and_Land_Cover_Classification_Using_Novel_Deep_Learning_Models_Based_on_High_Spatial_Resolution_Satellite_Imagery

Code
U-net: https://github.com/milesial/Pytorch-UNet
ASPP: https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/deeplab.py
CBAM: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
Attention-Unet: https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py

'''
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, extra_parts=[True, True, True], bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        channels = [64, 128, 256, 512, 1024]
        self.is_residual = extra_parts[0]
        self.is_CBAM = extra_parts[1]
        self.is_ASPP = extra_parts[2]

        self.double1 = DoubleConv(n_channels, channels[0])
        self.down1 = Down()
        self.shorcut_1 = ShortcutConv(channels[0], channels[1])
        self.double2 = DoubleConv(channels[0], channels[1])
        self.down2 = Down()
        self.shorcut_2 = ShortcutConv(channels[1], channels[2])
        self.double3 = DoubleConv(channels[1], channels[2])
        self.down3 = Down()
        self.shorcut_3 = ShortcutConv(channels[2], channels[3])
        self.double4 = DoubleConv(channels[2], channels[3])
        self.down4 = Down()
        self.shorcut_4 = ShortcutConv(channels[3], channels[3])
        self.double5 = DoubleConv(channels[3], channels[3])

        self.ASPP = ASPP(channels[3], channels[3]//4)

        self.ca1 = ChannelAttention(channels[1])
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(channels[2])
        self.sa2 = SpatialAttention()
        self.ca3 = ChannelAttention(channels[3])
        self.sa3 = SpatialAttention()

        self.up1 = Up(channels[4], channels[2], bilinear)
        self.up2 = Up(channels[3], channels[1], bilinear)
        self.up3 = Up(channels[2], channels[0], bilinear)
        self.up4 = Up(channels[1], channels[0], bilinear)

        self.outc = OutConv(channels[0], n_classes)

    def forward(self, x):
        d_x1 = self.double1(x)
        x1 = self.down1(d_x1)
        if self.is_residual :
            shortcut1 = self.shorcut_1(x1)
            d_x2 = self.double2(x1)  + shortcut1  # residual
        else:
            d_x2 = self.double2(x1)

        if self.is_CBAM:
            x2 = self.ca1(d_x2) * d_x2
            x2 = self.sa1(x2) * x2
            x2 = self.down2(x2)
        else:
            x2 = self.down2(d_x2)

        if self.is_residual:
            shortcut2 = self.shorcut_2(x2)
            d_x3 = self.double3(x2) + shortcut2  # residual
        else:
            d_x3 = self.double3(x2)

        if self.is_CBAM:
            x3 = self.ca2(d_x3) * d_x3
            x3 = self.sa2(x3) * x3
            x3 = self.down2(x3)
        else:
            x3 = self.down2(d_x3)

        if self.is_residual:
            shortcut3 = self.shorcut_3(x3)
            d_x4 = self.double4(x3)  + shortcut3  # residual
        else:
            d_x4 = self.double4(x3)

        if self.is_CBAM:
            x4 = self.ca3(d_x4) * d_x4
            x4 = self.sa3(x4) * x4
            x4 = self.down2(x4)
        else:
            x4 = self.down2(d_x4)

        if self.is_residual:
            shortcut4 = self.shorcut_4(x4)

        if self.is_ASPP:
            x5 = self.ASPP(x4)
            if self.is_residual:
                x5 = self.double5(x5) + shortcut4
            else:
                x5 = self.double5(x5)
        else:
            if self.is_residual:
                x5 = self.double5(x4) + shortcut4
            else:
                x5 = self.double5(x4)

        x = self.up1(x5, d_x4)
        x = self.up2(x, d_x3)
        x = self.up3(x, d_x2)
        x = self.up4(x, d_x1)
        logits = self.outc(x)

        return logits
