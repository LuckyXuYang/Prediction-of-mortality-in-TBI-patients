import torch
import torch.nn as nn
import torchvision.models as models
from .CBAM import Channel_Attention, Spartial_Attention
# 6.9定稿版本
# 参考：
# arxiv 1505.04597
# arxiv 1801.05746，官方实现：https://github.com/ternaus/TernausNet
# https://blog.csdn.net/github_36923418/article/details/83273107
# pixelshuffle参考: arxiv 1609.05158

backbone = 'resnet50'

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

class DecoderBlock(nn.Module):
    """
    U-Net中的解码模块
    采用每个模块一个stride为1的3*3卷积加一个上采样层的形式
    上采样层可使用'deconv'、'pixelshuffle', 其中pixelshuffle必须要mid_channels=4*out_channles
    定稿采用pixelshuffle
    BN_enable控制是否存在BN，定稿设置为True
    """

    def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='pixelshuffle', BN_enable=True):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.BN_enable = BN_enable

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1,
                              bias=False)

        if self.BN_enable:
            self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

        if self.upsample_mode == 'deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels,

                                               kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        elif self.upsample_mode == 'pixelshuffle':
            self.upsample = nn.PixelShuffle(upscale_factor=2)
        if self.BN_enable:
            self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.BN_enable:
            x = self.norm1(x)
        x = self.relu1(x)
        x = self.upsample(x)
        if self.BN_enable:
            x = self.norm2(x)
        x = self.relu2(x)
        return x


class Resnet_Unet(nn.Module):
    """
    定稿使用resnet50作为backbone

    BN_enable控制是否存在BN，定稿设置为True
    """

    def __init__(self, num_classes, BN_enable=True, resnet_pretrain=False):
        super().__init__()
        self.BN_enable = BN_enable
        # encoder部分
        # 使用resnet34或50预定义模型，由于单通道入，因此自定义第一个conv层，同时去掉原fc层
        # 剩余网络各部分依次继承
        # 经过测试encoder取三层效果比四层更佳，因此降采样、升采样各取4次
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=resnet_pretrain)
            filters = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=resnet_pretrain)
            filters = [64, 256, 512, 1024, 2048]
        self.firstconv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1()
        self.encoder2 = resnet.layer2()
        self.encoder3 = resnet.layer3()
        self.encoder4 = resnet.layer4()

        # 跳跃链接
        self.Conn_conv11 = conv_block_single(in_channels=64, out_channels=64)
        self.Conn_conv22 = conv_block_single(in_channels=256, out_channels=64)
        self.Conn_conv33 = conv_block_single(in_channels=512, out_channels=64)
        self.Conn_conv44 = conv_block_single(in_channels=1024, out_channels=64)
        self.Conn_conv13 = conv_block_single(in_channels=64, out_channels=64)
        self.Conn_conv24 = conv_block_single(in_channels=256, out_channels=64)

        self.Up_conv54 = up_conv(ch_in=2048, ch_out=64, scale_factor=2)
        self.Up_conv43 = up_conv(ch_in=192, ch_out=64, scale_factor=2)
        self.Up_conv32 = up_conv(ch_in=192, ch_out=64, scale_factor=2)
        self.Up_conv21 = up_conv(ch_in=128, ch_out=64, scale_factor=2)
        self.Up_conv31 = up_conv(ch_in=192, ch_out=64, scale_factor=4)
        self.Up_conv41 = up_conv(ch_in=256, ch_out=64, scale_factor=8)

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

        # decoder部分
        # self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3] * 4, out_channels=filters[3],
        #                            BN_enable=self.BN_enable)
        # self.decoder1 = DecoderBlock(in_channels=filters[3] + filters[2], mid_channels=filters[2] * 4,
        #                              out_channels=filters[2], BN_enable=self.BN_enable)
        # self.decoder2 = DecoderBlock(in_channels=filters[2] + filters[1], mid_channels=filters[1] * 4,
        #                              out_channels=filters[1], BN_enable=self.BN_enable)
        # self.decoder3 = DecoderBlock(in_channels=filters[1] + filters[0], mid_channels=filters[0] * 4,
        #                              out_channels=filters[0], BN_enable=self.BN_enable)
        # if self.BN_enable:
        #     self.final = nn.Sequential(
        #         nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(32),
        #         nn.ReLU(inplace=False),
        #         nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
        #         nn.Sigmoid()
        #     )
        # else:
        #     self.final = nn.Sequential(
        #         nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
        #         nn.ReLU(inplace=False),
        #         nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
        #         nn.Sigmoid()
        #     )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)

        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e3)

        d2 = self.decoder1(torch.cat([center, e2], dim=1))
        d3 = self.decoder2(torch.cat([d2, e1], dim=1))
        d4 = self.decoder3(torch.cat([d3, x], dim=1))

        return self.final(d4)