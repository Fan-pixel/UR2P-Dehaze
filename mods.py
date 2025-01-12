import torch
from torch import nn
def default_conv(in_channels, out_channels, kernel_size, bias=True,dtype=torch.float32):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias,dtype=dtype)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size//2), bias=True,dtype=torch.float32)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, kernel_size=5):
        super(RDB, self).__init__()
        self.conv1 = default_conv(3, 64, kernel_size)
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate, kernel_size=kernel_size) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)  #k=1

    def forward(self, x, lrl=True):
        if lrl:
            return x + self.lff(self.layers(x))  # local residual learning
        else:
            return self.layers(x)
        
class HFRM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HFRM, self).__init__()
        # 更新default_conv的dtype为torch.float16
        conv = lambda in_channels, out_channels, kernel_size: default_conv(in_channels, out_channels, kernel_size, dtype=torch.float32)

        self.conv1 = conv(32, 64, kernel_size=5)  # 确保这里使用float16
        self.RDB_1 = nn.Sequential(
            RDB(64, 64, 3),
            nn.ReLU(True)
        )

        self.RDB_2 = nn.Sequential(
            RDB(64, 64, 3),
            nn.ReLU(True)
        )
        self.conv_block2 = nn.Sequential(
            conv(64, 64, 5),  # 确保这里使用float16
            nn.ReLU(True)
        )

        self.conv2 = conv(64, 32, 5)  # 确保这里使用float16

        # self.L_HERM_clip= clip_score.L_HFRM_clip()

    def forward(self, x):
        #x = x.float()
        x = x.to(next(self.parameters()).device)
        y=x
        out1 =self.conv1(x)
        out2 = self.RDB_1(out1)
        out3 = self.RDB_2(out2)
        out4 = out1 + out2 + out3
        out5 = self.conv_block2(out4)
        out = self.conv2(out5)

        return y - out