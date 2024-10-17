import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

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

class UNet_3plus(nn.Module):
    def __init__(self, img_ch=3, num_classes: int = 2):
        super(UNet_3plus, self).__init__()

        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.Maxpool8 = nn.MaxPool2d(kernel_size=8, stride=8)

        self.En_Conv11 = conv_block(ch_in=img_ch, ch_out=64)
        self.En_Conv22 = conv_block(ch_in=64, ch_out=128)
        self.En_Conv33 = conv_block(ch_in=128, ch_out=256)
        self.En_Conv44 = conv_block(ch_in=256, ch_out=512)
        self.En_Conv55 = conv_block(ch_in=512, ch_out=1024)

        self.Conn_conv14 = conv_block_single(ch_in=64, ch_out=64)
        self.Conn_conv24 = conv_block_single(ch_in=128, ch_out=64)
        self.Conn_conv34 = conv_block_single(ch_in=256, ch_out=64)
        self.Conn_conv44 = conv_block_single(ch_in=512, ch_out=64)
        self.Up_conv54 = up_conv(ch_in=1024, ch_out=64, scale_factor=2)

        self.Conn_conv13 = conv_block_single(ch_in=64, ch_out=64)
        self.Conn_conv23 = conv_block_single(ch_in=128, ch_out=64)
        self.Conn_conv33 = conv_block_single(ch_in=256, ch_out=64)
        self.Up_conv43 = up_conv(ch_in=320, ch_out=64, scale_factor=2)
        self.Up_conv53 = up_conv(ch_in=1024, ch_out=64, scale_factor=4)

        self.Conn_conv12 = conv_block_single(ch_in=64, ch_out=64)
        self.Conn_conv22 = conv_block_single(ch_in=128, ch_out=64)
        self.Up_conv32 = up_conv(ch_in=320, ch_out=64, scale_factor=2)
        self.Up_conv42 = up_conv(ch_in=320, ch_out=64, scale_factor=4)
        self.Up_conv52 = up_conv(ch_in=1024, ch_out=64, scale_factor=8)

        self.Conn_conv11 = conv_block_single(ch_in=64, ch_out=64)
        self.Up_conv21 = up_conv(ch_in=320, ch_out=64, scale_factor=2)
        self.Up_conv31 = up_conv(ch_in=320, ch_out=64, scale_factor=4)
        self.Up_conv41 = up_conv(ch_in=320, ch_out=64, scale_factor=8)
        self.Up_conv51 = up_conv(ch_in=1024, ch_out=64, scale_factor=16)

        self.De_conv44 = conv_block_single(ch_in=320, ch_out=320)
        self.De_conv33 = conv_block_single(ch_in=320, ch_out=320)
        self.De_conv22 = conv_block_single(ch_in=320, ch_out=320)
        self.De_conv11 = conv_block_single(ch_in=320, ch_out=320)

        self.out_conv_1x1 = nn.Conv2d(320, num_classes, kernel_size=1, stride=1, padding=0)

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
        e1_4 = self.Maxpool8(e1)
        de14 = self.Conn_conv14(e1_4)
        e2_4 = self.Maxpool4(e2)
        de24 = self.Conn_conv24(e2_4)
        e3_4 = self.Maxpool2(e3)
        de34 = self.Conn_conv34(e3_4)
        de44 = self.Conn_conv44(e4)
        d54 = self.Up_conv54(de5)
        d_cat4 = torch.cat((de14, de24, de34, de44, d54), dim=1)
        d4 = self.De_conv44(d_cat4)

        e1_3 = self.Maxpool4(e1)
        de13 = self.Conn_conv13(e1_3)
        e2_3 = self.Maxpool2(e2)
        de23 = self.Conn_conv23(e2_3)
        de33 = self.Conn_conv33(e3)
        d43 = self.Up_conv43(d4)
        d53 = self.Up_conv53(de5)
        d_cat3 = torch.cat((de13, de23, de33, d43, d53), dim=1)
        d3 = self.De_conv33(d_cat3)

        e1_2 = self.Maxpool2(e1)
        de12 = self.Conn_conv12(e1_2)
        de22 = self.Conn_conv22(e2)
        d32 = self.Up_conv32(d3)
        d42 = self.Up_conv42(d4)
        d52 = self.Up_conv52(de5)
        d_cat2 = torch.cat((de12, de22, d32, d42, d52), dim=1)
        d2 = self.De_conv22(d_cat2)

        d21 = self.Up_conv21(d2)
        d31 = self.Up_conv31(d3)
        d41 = self.Up_conv41(d4)
        d51 = self.Up_conv51(de5)
        de11 = self.Conn_conv11(e1)
        d_cat1 = torch.cat((de11, d21, d31, d41, d51), dim=1)
        d1 = self.De_conv11(d_cat1)

        out = self.out_conv_1x1(d1)

        return {"out": out}