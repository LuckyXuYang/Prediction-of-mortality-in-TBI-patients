import torch
import torch.nn as nn
import torch.nn.functional as F
from .CBAM import Channel_Attention, Spartial_Attention

# class Up(nn.Module):  # 将x1上采样，然后调整为x2的大小
#     """Upscaling then double conv"""
#
#     def __init__(self):
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)  # 将传入数据上采样，
#
#         diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
#         diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
#
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])  # 填充为x2相同的大小
#         return x1


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


class conv_block_single(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(conv_block_single,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    expansion = 4

    '''
    espansion是通道扩充的比例
    注意实际输出channel = middle_channels * BottleNeck.expansion
    '''

    def __init__(self, in_channels, middle_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != middle_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels * BottleNeck.expansion, stride=stride, kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(middle_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU()
        )
        self.second = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.first(x)
        out = self.second(out)

        return out


# unet+resnet 可以更换 layers
class UResnet3p_cbam(nn.Module):
    def __init__(self, num_classes, input_channels=1, block=BasicBlock, layers=[3, 4, 6, 3]):  #50层以上选BottleNeck 一下选BasicBlock
        super().__init__()
        nb_filter = [64, 128, 256, 512, 1024]
        # self.Up = Up()

        self.in_channel = nb_filter[0]
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool8 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block(input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[1], layers[0], 1)
        self.conv2_0 = self._make_layer(block, nb_filter[2], layers[1], 1)
        self.conv3_0 = self._make_layer(block, nb_filter[3], layers[2], 1)
        self.conv4_0 = self._make_layer(block, nb_filter[4], layers[3], 1)

        #跳跃链接
        self.Conn_conv11 = conv_block_single(in_channels=64, out_channels=64)
        self.Conn_conv22 = conv_block_single(in_channels=256, out_channels=64)
        self.Conn_conv33 = conv_block_single(in_channels=512, out_channels=64)
        self.Conn_conv44 = conv_block_single(in_channels=1024, out_channels=64)
        self.Conn_conv13 = conv_block_single(in_channels=64, out_channels=64)
        self.Conn_conv24 = conv_block_single(in_channels=512, out_channels=64)

        self.Up_conv54 = up_conv(ch_in=2048, ch_out=64, scale_factor=2)
        self.Up_conv43 = up_conv(ch_in=192, ch_out=64, scale_factor=2)
        self.Up_conv32 = up_conv(ch_in=192, ch_out=64, scale_factor=2)
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

    def _make_layer(self, block, middle_channel, num_blocks, stride):
        '''
        middle_channels中间维度，实际输出channels = middle_channels * block.expansion
        num_blocks，一个Layer包含block的个数
        '''

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, middle_channel, stride))
            self.in_channel = middle_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        e1 = self.conv0_0(input) #256
        e2 = self.conv1_0(self.pool(e1)) #256
        e3 = self.conv2_0(self.pool(e2)) #128
        e4 = self.conv3_0(self.pool(e3)) #64
        de5 = self.conv4_0(self.pool(e4)) #32

        # decoding + concat path
        e2_4 = self.pool4(e2)
        de24 = self.Conn_conv24(e2_4)
        de44 = self.Conn_conv44(e4)
        d54 = self.Up_conv54(de5)
        d_cat4 = torch.cat((de24, de44, d54), dim=1)
        d4 = self.De_conv44(d_cat4)
        d44 = self.ca4(d4)
        d44 = self.sa4(d44)
        d4 = self.relu(d44)

        e1_3 = self.pool4(e1)
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

        return out    #UResnet50 = UResnet(block=BottleNeck,layers=[3,4,6,3],num_classes=2)

