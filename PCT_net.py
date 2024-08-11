import einops
import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as func


class Integrate(nn.Module):
    def __init__(self):
        super(Integrate, self).__init__()
        device = torch.device('cuda:0')

        self.alpha = nn.Parameter(torch.rand((1,), device=device), requires_grad=True)
        self.beta = nn.Parameter(torch.rand((1,), device=device), requires_grad=True)

        self.conv = nn.Conv2d(64, 16, kernel_size=(6, 2), stride=(6, 4))
        self.de_conv = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(768, 256)),
            ('norm_1', nn.LayerNorm(256)),
            ('acti_1', nn.ReLU()),
            ('linear_2', nn.Linear(256, 128)),
            ('norm_2', nn.LayerNorm(128)),
            ('acti_2', nn.ReLU()),
            ('linear_3', nn.Linear(128, 64)),
            ('norm_3', nn.LayerNorm(64)),
            ('acti_3', nn.ReLU()),
        ]))
        self.linear = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(64, 32)),
            ('norm_1', nn.LayerNorm(32)),
            ('acti_1', nn.ReLU()),
            ('linear_2', nn.Linear(32, 16)),
            ('norm_2', nn.LayerNorm(16)),
        ]))

    def forward(self, x, t):
        c = self.conv(x)
        c = c.flatten(1)
        c = self.de_conv(c)
        t = t[:, 0, :]

        o = c + t
        o = self.linear(o)

        return o


class Featurer(nn.Module):
    def __init__(self, add_axis=1):
        super(Featurer, self).__init__()
        self.stem = Stem(add_axis=add_axis, stem_out=16)
        self.project = Project()

        self.conv_block = nn.Sequential(OrderedDict([
            ('block_1', ConvBlock(channels_in=16, channels_out=32)),
            ('block_2', ConvBlock(channels_in=32, channels_out=64)),
        ]))
        self.trans_block = nn.Sequential(OrderedDict([
            ('block_1', TransBlock()),
            ('block_2', TransBlock())
        ]))

        self.integrate = Integrate()

    def forward(self, x):
        x = self.stem(x)

        c = self.conv_block(x)
        t = self.trans_block(self.project(x))

        o = self.integrate(c, t)

        return o


class TransBlock(nn.Module):
    def __init__(self, length=64, embed_dim=3*64, attention_head_num=4, trans_drop_rate=0.3):
        super(TransBlock, self).__init__()
        self.scale = 8
        self.embed_dim = embed_dim
        self.head_num = attention_head_num

        self.drop = nn.Dropout(p=trans_drop_rate)
        self.layers_1 = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(length, embed_dim)),
            ('norm_1', nn.LayerNorm(embed_dim)),
            ('drop_1', nn.Dropout(p=trans_drop_rate))
        ]))
        self.layers_2 = nn.Sequential(OrderedDict([
            ('linear_2', nn.Linear(length, length)),
            ('norm_2', nn.LayerNorm(length)),
            ('drop_2', nn.Dropout(p=trans_drop_rate))
        ]))

    def forward(self, x):
        b, c, le = x.shape
        y = self.layers_1(x)

        qkv = y.reshape(b, c, 3, self.head_num, self.embed_dim//(3*self.head_num)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        atte = func.softmax(((q @ k.transpose(3, 2)) / self.scale), dim=-1)
        atte = self.drop(self.drop(atte) @ v)
        atte = atte.transpose(1, 2).flatten(2)

        z = self.layers_2(atte) + x

        return z


class ConvBlock(nn.Module):
    def __init__(self, channels_in=16, channels_out=32):
        super(ConvBlock, self).__init__()

        self.layers = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(channels_in, channels_out, kernel_size=(3, 3), padding=(1, 1))),
            ('norm_1', nn.BatchNorm2d(channels_out)),
            ('acti_1', nn.ReLU()),
            ('conv_2', nn.Conv2d(channels_out, channels_out, kernel_size=(1, 1))),
            ('norm_2', nn.BatchNorm2d(channels_out)),
            ('acti_2', nn.ReLU()),
            ('conv_3', nn.Conv2d(channels_out, channels_out, kernel_size=(3, 3), padding=(1, 1))),
            ('norm_3', nn.BatchNorm2d(channels_out)),
            ('acti_3', nn.ReLU()),
        ]))

    def forward(self, x):

        x = self.layers(x)

        return x


class Project(nn.Module):
    def __init__(self, dim_in=16*52, dim_out=64):
        super(Project, self).__init__()
        self.channels_token = nn.Parameter(torch.zeros((1, 1, 1)))

        self.layers = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(dim_in, dim_out)),
            ('norm', nn.LayerNorm(dim_out)),
            ('acti', nn.GELU())
        ]))

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b w (c h)')
        b, c, w = x.shape
        channels_token = self.channels_token.expand(b, -1, w)
        x = self.layers(torch.cat([channels_token, x], dim=1))

        return x


class Stem(nn.Module):
    def __init__(self, add_axis=1, stem_out=16):
        super(Stem, self).__init__()

        self.layers = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(add_axis, 4, kernel_size=(32, 1), stride=(10, 1), padding=(16, 1))),
            ('norm_1', nn.BatchNorm2d(4)),
            ('acti_1', nn.ReLU()),
            ('conv_2', nn.Conv2d(4, stem_out, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))),
            ('norm_2', nn.BatchNorm2d(stem_out)),
            ('acti_2', nn.ReLU())
        ]))

    def forward(self, x):

        x = self.layers(x)

        return x


class PCT(nn.Module):
    def __init__(self, add_axis=1):
        super(PCT, self).__init__()

        self.featurer = Featurer(add_axis)

        # self.classifier = nn.Linear(2, 2)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(16, 4)),
            ('linear_2', nn.Linear(4, 2)),
        ]))

    def forward(self, x):

        x = self.featurer(x)

        o = self.classifier(x)

        return o


if __name__ == '__main__':

    inputs = torch.ones((128, 1, 1024, 22)).cuda()
    model = PCT().cuda()
    outputs = model.featurer(inputs)

    # print number of parameters
    total_parameter = sum(param.numel() for param in model.parameters())
    print('total parameters in the model is {}'.format(total_parameter))
    total_training_parameter = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print('total training parameters in the model is {}'.format(total_parameter))

    print('CCT')
