""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        #print("After Double conv shape : {}".format(x.size()))
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        #print("After Dowm shape : {}".format(x.size()))
        return x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.doubleConv = DoubleConv(in_channels, out_channels)  # in the case of no attention

        self.attentionConv = UpAttention(in_channels//2)  # in the case of attention
        self.singleConv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # in the case of attention

    def forward(self, x1, x2):
        '''
        :param x1: bottom
        :param x2: skip connection
        '''
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        '''
        # no attention
        x = torch.cat([x2, x1], dim=1)
        x = self.doubleConv(x)
        '''

        # attention
        x = self.attentionConv(x1, x2)
        x = torch.cat([x1, x], dim=1)
        x = self.singleConv(x)


        #print("After Up shape : {}".format(x.size()))
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        #print("After Out shape : {}".format(x.size()))
        return x

class ShortcutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShortcutConv, self).__init__()
        self.shortConv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                                       nn.BatchNorm2d(out_channels))
    def forward(self, x):
        x = self.shortConv(x)
        #print("After shortcut shape : {}".format(x.size()))
        return x

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        # Cmab attention
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        # Cmab attention
       # print(x.size())
        input = x
        x = self.ca(x) * input
        x = self.sa(x) * x
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        #print("Channel attention shape : {}".format(out.size()))
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        #print("spatial attention shape : {}".format(x.size()))
        return self.sigmoid(x)

class UpAttention(nn.Module):
    """
    Attention Block
    x : bottom
    g : skip connection
    """
    def __init__(self, up_channels):
        super(UpAttention, self).__init__()

        self.W_x = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(up_channels)
        )

        self.W_g = nn.Sequential(
            nn.Conv2d(up_channels, up_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(up_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(up_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        x1 = self.W_x(x)
        g1 = self.W_g(g)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1,  bias=False)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               dilation=int(2*mult), padding=int(2*mult),
                               bias=False)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               dilation=int(3*mult), padding=int(3*mult),
                               bias=False)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               dilation=int(4 * mult), padding=int(4 * mult),
                               bias=False)

        self.aspp1_bn = nn.BatchNorm2d(out_channels, momentum)
        self.aspp2_bn = nn.BatchNorm2d(out_channels, momentum)
        self.aspp3_bn = nn.BatchNorm2d(out_channels, momentum)
        self.aspp4_bn = nn.BatchNorm2d(out_channels, momentum)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)

        x = torch.cat((x1, x2, x3, x4), 1)
        return x