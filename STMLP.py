import einops
import torch
import torch.nn as nn
from STMLP_configs import configs


class ChannelsuppressionBlock(nn.Module):
    def __init__(
            self,
            device,
            channels,
    ):
        super().__init__()
        self.channel_weight = nn.Parameter(torch.rand((channels, 1, 1), device=device), requires_grad=True)

    def forward(self, x):
        y = torch.matmul(self.channel_weight, x)

        return y


class FilterLayer(nn.Module):
    def __init__(
            self,
            filter_matrix,
    ):
        super().__init__()
        self.fft = torch.fft.fft
        self.filter = filter_matrix
        self.ifft = torch.fft.ifft

    def forward(self, x):
        y = self.fft(x)
        y = self.filter * y
        y = self.ifft(y).real

        return y


class TemporalMlpBlock(nn.Module):  # equivalent to 1*1 convolution
    def __init__(
            self,
            hidden_dim,
            channels_mlp_dim,
            drop_rate
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear_1 = nn.Linear(hidden_dim, channels_mlp_dim)
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(channels_mlp_dim, hidden_dim)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        y = self.norm(x)
        y = self.linear_1(y)
        y = self.dropout(y)
        y = self.activation(y)
        y = self.linear_2(y)

        return y


class SpatialMlpBlock(nn.Module):
    def __init__(
            self,
            hidden_dim,
            reduced_dim,
            token_mlp_dim,
            drop_rate
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear_1 = nn.Linear(reduced_dim, token_mlp_dim)
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(token_mlp_dim, reduced_dim)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        y = self.norm(x)
        y = y.permute(0, 2, 1)
        y = self.linear_1(y)
        y = self.dropout(y)
        y = self.activation(y)
        y = self.linear_2(y)
        y = y.permute(0, 2, 1)

        return y


class MLPBlock(nn.Module):
    def __init__(
            self,
            hidden_dim,
            reduced_dim,
            token_mlp_dim,
            channels_mlp_dim,
            drop_rate
    ):
        super().__init__()
        self.mlpblock_spatial = SpatialMlpBlock(hidden_dim, reduced_dim, token_mlp_dim, drop_rate)
        self.mlpblock_temporal = TemporalMlpBlock(hidden_dim, channels_mlp_dim, drop_rate)

    def forward(self, x):
        y = self.mlpblock_spatial(x)
        x = x + y
        y = self.mlpblock_temporal(x)
        y = x + y

        return y


class Featurer(nn.Module):
    def __init__(self,
                 device,
                 channels,
                 length,
                 num_classes,
                 num_layers,
                 patch_size,
                 hidden_dim,
                 tokens_hidden_dim,
                 channels_hidden_dim,
                 drop_rate):
        super().__init__()

        num_patch = (length // patch_size)
        filter_ = torch.nn.Parameter(torch.rand((channels, 1, length), device=device, requires_grad=True))

        self.layers = num_layers
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=drop_rate)
        self.filter_layer = FilterLayer(filter_)
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.channelsuppresion = ChannelsuppressionBlock(device, channels)
        self.conv2d = nn.Conv2d(channels, hidden_dim, kernel_size=(1, patch_size), stride=(1, patch_size))
        self.mixerblock = MLPBlock(hidden_dim, num_patch, tokens_hidden_dim, channels_hidden_dim, drop_rate)

    def forward(self, x):
        x = einops.rearrange(x, 'b a l c -> b c a l ')

        y = self.filter_layer(x)
        y = self.channelsuppresion(y)
        y = self.conv2d(y)
        y = einops.rearrange(y, 'n c h w -> n (h w) c')

        for _ in range(self.layers):
            y = self.mixerblock(y)

        y = self.dropout(y)
        y = self.norm(y)
        y = torch.mean(y, dim=1)

        return y


class STMLP(nn.Module):
    def __init__(
            self,
            device,
            channels,
            length,
            num_classes,
            num_layers,
            patch_size,
            hidden_dim,
            tokens_hidden_dim,
            channels_hidden_dim,
            drop_rate
    ):
        super().__init__()

        self.featurer = Featurer(device, channels, length, num_classes, num_layers, patch_size, hidden_dim,
                                 tokens_hidden_dim, channels_hidden_dim, drop_rate)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        y = self.featurer(x)

        y = self.classifier(y)

        return y


if __name__ == "__main__":

    # print the total number of network parameters
    test_data = torch.rand((16, 1, 4*256, 22)).cuda()
    devices = 'cuda'
    model = STMLP(**configs["ST-MLP"]).to(devices)  # parameters: 695256
    # model = MlpMixerGating(**configs["mixer+gate_source_K"]).to(devices)  # parameters: 1200496
    # preds = model(test_data)

    test_out = model(test_data)

    # print number of parameters
    total_parameter = sum(param.numel() for param in model.parameters())
    print('total parameters in the model is {}'.format(total_parameter))
    total_training_parameter = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print('total training parameters in the model is {}'.format(total_parameter))

    print('STMLP')
