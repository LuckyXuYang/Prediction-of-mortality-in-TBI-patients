import torch.nn as nn
import torch

class Channel_Attention(nn.Module):

    def __init__(self, channel, r):
        super(Channel_Attention, self).__init__()

        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.__fc = nn.Sequential(
            nn.Conv2d(channel, channel//r, 1, bias=True),
            # DeformConv2d(channel, channel//r, kernel_size=1, stride=1, padding=0, bias=True, modulation=True),
            nn.ReLU(True),
            nn.Conv2d(channel//r, channel, 1, bias=True),
            # DeformConv2d(channel//r, channel, kernel_size=1, stride=1, padding=0, bias=True, modulation=True),
        )
        self.__sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1)

        y2 = self.__max_pool(x)
        y2 = self.__fc(y2)

        y = self.__sigmoid(y1+y2)
        return x * y

# if __name__ == '__main__':
#     x = torch.rand(1,64,240,240)
#     y = Channel_Attention(64,8)
#     out = y(x)*x
#     print(out.shape)

class Spartial_Attention(nn.Module):

    def __init__(self, kernel_size):
        super(Spartial_Attention, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2

        self.__layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding), #ch_in, ch_out, kernel_size,stride,padding,bias
            # DeformConv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=True, modulation=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.__layer(mask)
        return x * mask

# if __name__ == '__main__':
#     x = torch.rand(1,64,240,240)
#     y = Spartial_Attention(3)
#     out = y(x)
#     print(out.shape)