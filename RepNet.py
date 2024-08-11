import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as func


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return func.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(x)


def fuse_bn(conv, bn):
    weight = conv.weight    # (16, 1, 5, 5)
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return weight * t, beta - running_mean * gamma / std


def conv_bn(in_channels, out_channels, kernel_size, padding, groups):
    results = nn.Sequential()
    results.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                         padding=padding, groups=groups, bias=False))
    results.add_module('bn', nn.BatchNorm2d(out_channels))
    return results


class ReparamBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, small_kernel=3):  # exp,
        super(ReparamBlock, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2

        self.lk_origin = conv_bn(dim, dim, kernel_size, padding, dim)

        if small_kernel is not None:
            assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be large than the large kernel'
            self.small_conv = conv_bn(dim, dim, small_kernel, (1, 1), dim)

    def forward(self, x):  # (64,32,128,10) 7
        add = x
        if hasattr(self, 'lk_reparam'):  # testing
            out = self.lk_reparam(x)
        else:   # training
            out = self.lk_origin(x)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(x)
        return out + add

    def get_equivalent_weight_bias(self):
        eq_w, eq_b = fuse_bn(self.lk_origin.conv, self.lk_origin.bn)
        if hasattr(self, 'small_conv'):
            small_w, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #  add to the central part
            eq_w += nn.functional.pad(small_w, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_w, eq_b

    def merge_kernel(self):
        eq_w, eq_b = self.get_equivalent_weight_bias()
        self.lk_reparam = nn.Conv2d(in_channels=self.lk_origin.conv.in_channels,
                                    out_channels=self.lk_origin.conv.out_channels,
                                    kernel_size=self.lk_origin.conv.kernel_size, stride=self.lk_origin.conv.stride,
                                    padding=self.lk_origin.conv.padding, dilation=self.lk_origin.conv.dilation,
                                    groups=self.lk_origin.conv.groups, bias=True)
        self.lk_reparam.weight.data = eq_w
        self.lk_reparam.bias.data = eq_b
        self.__delattr__('lk_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Featurer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        dims = [16, 32, 64]
        stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=(21, 1), stride=(10, 1)),  # , padding=(10, 0)
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(stem)
        for i in range(2):
            downsample_layer = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(dims[i], dims[i], 1, 1, 0, bias=False),  # False
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=(2, 2), stride=(2, 2), bias=False),
                nn.BatchNorm2d(dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(3):
            stage = nn.Sequential(
                *[ReparamBlock(dim=dims[i])]
            )
            self.stages.append(stage)

        ks_dim = (self.channels // 2) // 2  # ((self.input[3] - 2) // 2) // 2  # -2
        self.downsample = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[-1], dims[-1], 1, 1, 0, bias=False),  # False
            nn.Conv2d(dims[-1], dims[-1], kernel_size=(7, ks_dim), stride=(2, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(dims[-1]),
        )
        self.transition = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
        )

        self.feature = None
        self.logit_x = None
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

    def forward(self, x):
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = self.downsample(x)
        x = self.transition(x)

        return x


class RepNet(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.featurer = Featurer(channels)
        self.classifier = Classifier()

    def forward(self, x):

        x = self.featurer(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":

    inputs1_ = torch.randn((128, 1, 4 * 256, 22)).cuda()  # N C H W
    model1 = RepNet(22).cuda()
    outputs1 = model1(inputs1_)

    inputs2_ = torch.randn((128, 1, 4 * 200, 22)).cuda()  # N C H W
    model2 = RepNet(22).cuda()
    outputs2 = model2(inputs2_)

    # print number of parameters
    total_parameter = sum(param.numel() for param in model1.parameters())
    print('total parameters in the model is {}'.format(total_parameter))
    total_training_parameter = sum(param.numel() for param in model1.parameters() if param.requires_grad)
    print('total training parameters in the model is {}'.format(total_parameter))

    a = 1



