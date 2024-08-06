import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from VAN import van_b2


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.van = van_b2(pretrained=True, num_classes=1)
        self.AngBranch = AngBranch()
        self.SFE = nn.Conv2d(1, 3, kernel_size=3, stride=1, dilation=7, padding=7, bias=False)
        self.AFE = nn.Conv2d(1, 32, kernel_size=7, stride=7, padding=0, bias=False)

        self.rerange_layer = Rearrange('b c h w -> b (h w) c')
        self.avg_pool = nn.AdaptiveAvgPool2d(224 // 32)

        # Adaptive head
        embed_dim = 1216
        self.head_score = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.ReLU()
        )
        self.head_weight = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_ang = self.AFE(x)
        a1, a2 = self.AngBranch(x_ang)
        a1 = self.avg_pool(a1)
        a2 = self.avg_pool(a2)

        x_spa = self.SFE(x)
        layer1_s, layer2_s, layer3_s, layer4_s = self.van(x_spa)    # (b,64,56,56); (b,128,28,28); (b,320,14,14); (b,512,7,7)
        s1 = self.avg_pool(layer1_s)
        s2 = self.avg_pool(layer2_s)
        s3 = self.avg_pool(layer3_s)
        s4 = self.avg_pool(layer4_s)

        feats = torch.cat((s1, s2, s3, s4, a1, a2), dim=1)
        feats = self.rerange_layer(feats)  # (b, c, h, w) -> (b, h*w, c)
        assert feats.shape[-1] == 1216 and len(feats.shape) == 3, 'Unexpected stacked features: {}'.format(feats.shape)

        scores = self.head_score(feats)
        weights = self.head_weight(feats)
        q = torch.sum(scores * weights, dim=1) / torch.sum(weights, dim=1)

        return q


class BasicBlockSem(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(BasicBlockSem, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_planes)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        # Channel Attention Module
        out = self.ca(out) * out
        out = self.relu(out)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.relu1 = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class AngBranch(nn.Module):

    def __init__(self):
        super(AngBranch, self).__init__()

        self.in_block_sem_1 = BasicBlockSem(32, 64, kernel_size=3, stride=2, padding=1)
        self.in_block_sem_2 = BasicBlockSem(64, 128, kernel_size=3, stride=2, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        y1 = self.in_block_sem_1(x)
        y2 = self.in_block_sem_2(y1)
        return y1, y2


if __name__ == "__main__":
    net = Network().cuda()
    from thop import profile

    input1 = torch.randn(1, 1, 224, 224).cuda()
    flops, params = profile(net, inputs=(input1,))
    print('   Number of parameters: %.5fM' % (params / 1e6))
    print('   Number of FLOPs: %.5fG' % (flops / 1e9))
