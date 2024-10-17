import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .CBAM import Channel_Attention, Spartial_Attention

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class conv_block_single(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block_single,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out, scale_factor):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=False),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class conv_block_singlenorelu(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block_singlenorelu,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class refUNet_3p_CBAM(nn.Module):
    def __init__(self, img_ch=1, num_classes: int = 2):
        super(refUNet_3p_CBAM, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.Maxpool8 = nn.MaxPool2d(kernel_size=8, stride=8)

        self.En_Conv11 = conv_block(ch_in=img_ch, ch_out=64)
        self.En_Conv22 = conv_block(ch_in=64, ch_out=128)
        self.En_Conv33 = conv_block(ch_in=128, ch_out=256)
        self.En_Conv44 = conv_block(ch_in=256, ch_out=512)
        self.En_Conv55 = conv_block(ch_in=512, ch_out=1024)

        self.Conn_conv24 = conv_block_single(ch_in=128, ch_out=64)
        self.Conn_conv44 = conv_block_single(ch_in=512, ch_out=64)
        self.Up_conv54 = up_conv(ch_in=1024, ch_out=64, scale_factor=2)

        self.Conn_conv13 = conv_block_single(ch_in=64, ch_out=64)
        self.Conn_conv33 = conv_block_single(ch_in=256, ch_out=64)
        self.Up_conv43 = up_conv(ch_in=192, ch_out=64, scale_factor=2)

        self.Conn_conv22 = conv_block_single(ch_in=128, ch_out=64)
        self.Up_conv32 = up_conv(ch_in=192, ch_out=64, scale_factor=2)

        self.Conn_conv11 = conv_block_single(ch_in=64, ch_out=64)
        self.Up_conv21 = up_conv(ch_in=128, ch_out=64, scale_factor=2)
        self.Up_conv31 = up_conv(ch_in=192, ch_out=64, scale_factor=4)
        self.Up_conv41 = up_conv(ch_in=192, ch_out=64, scale_factor=8)

        self.De_conv44 = conv_block_singlenorelu(ch_in=192, ch_out=192)
        self.De_conv33 = conv_block_singlenorelu(ch_in=192, ch_out=192)
        self.De_conv22 = conv_block_singlenorelu(ch_in=128, ch_out=128)
        self.De_conv11 = conv_block_singlenorelu(ch_in=256, ch_out=256)

        self.ca1 = Channel_Attention(channel=256, r=16)
        self.sa1 = Spartial_Attention(kernel_size=7)

        self.ca2 = Channel_Attention(channel=128, r=16)
        self.sa2 = Spartial_Attention(kernel_size=7)

        self.ca3 = Channel_Attention(channel=192, r=16)
        self.sa3 = Spartial_Attention(kernel_size=7)

        self.ca4 = Channel_Attention(channel=192, r=16)
        self.sa4 = Spartial_Attention(kernel_size=7)

        self.out_conv_1x1 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # encoding path
        e1 = self.En_Conv11(x)

        e2 = self.Maxpool2(e1)
        e2 = self.En_Conv22(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.En_Conv33(e3)

        e4 = self.Maxpool2(e3)
        e4 = self.En_Conv44(e4)

        de5 = self.Maxpool2(e4)
        de5 = self.En_Conv55(de5)

        # decoding + concat path
        e2_4 = self.Maxpool4(e2)
        de24 = self.Conn_conv24(e2_4)
        de44 = self.Conn_conv44(e4)
        d54 = self.Up_conv54(de5)
        d_cat4 = torch.cat((de24, de44, d54), dim=1)
        d4 = self.De_conv44(d_cat4)
        d44 = self.ca4(d4)
        d44 = self.sa4(d44)
        d4 = self.relu(d44)

        e1_3 = self.Maxpool4(e1)
        de13 = self.Conn_conv13(e1_3)
        de33 = self.Conn_conv33(e3)
        d43 = self.Up_conv43(d4)
        d_cat3 = torch.cat((de13, de33, d43), dim=1)
        d3 = self.De_conv33(d_cat3)
        d33 = self.ca3(d3)
        d33 = self.sa3(d33)
        d3 = self.relu(d33)

        de22 = self.Conn_conv22(e2)
        d32 = self.Up_conv32(d3)
        d_cat2 = torch.cat((de22, d32), dim=1)
        d2 = self.De_conv22(d_cat2)
        d22 = self.ca2(d2)
        d22 = self.sa2(d22)
        d2 = self.relu(d22)

        d21 = self.Up_conv21(d2)
        d31 = self.Up_conv31(d3)
        d41 = self.Up_conv41(d4)
        de11 = self.Conn_conv11(e1)
        d_cat1 = torch.cat((de11, d21, d31, d41), dim=1)
        d1 = self.De_conv11(d_cat1)
        d11 = self.ca1(d1)
        d11 = self.sa1(d11)
        d1 = self.relu(d11)

        out = self.out_conv_1x1(d1)

        return out
