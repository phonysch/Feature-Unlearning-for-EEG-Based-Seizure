import einops
import torch
from collections import OrderedDict
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(64, 16)),
            ('linear_2', nn.Linear(16, 2)),
        ]))

    def forward(self, x):

        y = self.classifier(x)

        return y


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


class Featurer(nn.Module):
    def __init__(self, add_axis=1):
        super(Featurer, self).__init__()
        self.stem = Stem(add_axis=add_axis, stem_out=16)

        self.conv_block_1 = ConvBlock(channels_in=16, channels_out=16)
        self.conv_block_2 = ConvBlock(channels_in=16, channels_out=16)

        self.conv = nn.Conv2d(16, 16, kernel_size=(6, 2), stride=(6, 4))

        self.linear = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(768, 256)),
            ('linear_2', nn.Linear(256, 128)),
            ('linear_3', nn.Linear(128, 64)),
        ]))

    def forward(self, x):
        x = self.stem(x)

        y = self.conv_block_1(x)
        c = self.conv_block_2(y+x)
        c = self.conv(c+y)

        c = c.flatten(1)
        c = self.linear(c)

        return c


class Featurer_kaggle(nn.Module):
    def __init__(self, add_axis=1):
        super(Featurer_kaggle, self).__init__()
        self.extend = Extend()

        self.stem = Stem(add_axis=add_axis, stem_out=16)

        self.conv_block = nn.Sequential(OrderedDict([
            ('block_1', ConvBlock(channels_in=16, channels_out=32)),
            ('block_2', ConvBlock(channels_in=32, channels_out=64)),
        ]))

        self.conv = nn.Conv2d(64, 16, kernel_size=(6, 2), stride=(6, 4))

        self.linear = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(768, 256)),
            ('linear_2', nn.Linear(256, 128)),
            ('linear_3', nn.Linear(128, 64)),
        ]))

    def forward(self, x):
        x = self.extend(x)

        x = self.stem(x)

        c = self.conv_block(x)

        c = self.conv(c)
        c = c.flatten(1)
        c = self.linear(c)

        return c


class Extend(nn.Module):
    def __init__(self, length=800, width=16):
        super(Extend, self).__init__()

        self.extend_l = nn.Linear(length, 1024)

        self.extend_w = nn.Linear(width, 22)

    def forward(self, x):
        x = self.extend_w(x)

        x = einops.rearrange(x, 'b c h w -> b c w h')

        x = self.extend_l(x)

        x = einops.rearrange(x, 'b c w h -> b c h w')

        return x


class CNN(nn.Module):
    def __init__(self, add_axis=1):
        super(CNN, self).__init__()

        self.featurer = Featurer(add_axis)

        self.classifier = Classifier()

    def forward(self, x):

        x = self.featurer(x)

        o = self.classifier(x)

        return o


class CNN_kaggle(nn.Module):
    def __init__(self, add_axis=1):
        super(CNN_kaggle, self).__init__()

        self.featurer = Featurer_kaggle(add_axis)

        self.classifier = Classifier()

    def forward(self, x):

        x = self.featurer(x)

        o = self.classifier(x)

        return o


if __name__ == '__main__':

    inputs1 = torch.ones((128, 1, 1024, 22)).cuda()
    model1 = CNN().cuda()
    outputs1 = model1(inputs1)

    inputs2 = torch.ones((128, 1, 800, 16)).cuda()
    model2 = CNN_kaggle().cuda()
    outputs2 = model2(inputs2)

    # print number of parameters
    total_parameter = sum(param.numel() for param in model1.parameters())
    print('total parameters in the model is {}'.format(total_parameter))
    total_training_parameter = sum(param.numel() for param in model1.parameters() if param.requires_grad)
    print('total training parameters in the model is {}'.format(total_parameter))

    print('CNN')
