import einops
import torch
import torch.nn.functional as func
from torch import nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(1, 2, (1, 51), (1, 1)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, emb_size, (16, 5), stride=(1, 5)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = func.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=5, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):

        outs = self.clshead(x)

        return outs


class ChannelAttention(nn.Module):
    def __init__(self, sequence_num=1000, inter=30):
        super(ChannelAttention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(16, 16),
            nn.LayerNorm(16),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(16, 16),
            nn.LayerNorm(16),
            nn.Dropout(0.3)
        )

        self.projection = nn.Sequential(
            nn.Linear(16, 16),
            nn.LayerNorm(16),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, 'b o c s->b o s c')
        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling

        channel_atten_score = func.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)
        '''
        projections after or before multiplying with attention score are almost the same.
        '''
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')
        return out


class Featurer(nn.Module):
    def __init__(self, embedding_dim=10, depth=3):
        super(Featurer, self).__init__()

        self.change = nn.Linear(22, 16)
        self.attention = ChannelAttention()
        self.embedding = PatchEmbedding(embedding_dim)
        self.encoder = TransformerEncoder(depth, embedding_dim)

    def forward(self, x):

        x = einops.rearrange(self.change(x), 'b a l c -> b a c l')
        for i in range(4):
            x = self.attention(x) + x
        x = self.embedding(x)
        x = self.encoder(x)

        return x


class ViT(nn.Sequential):
    def __init__(self, embedding_dim=10, depth=3, n_classes=2):
        super(ViT, self).__init__()

        self.featurer = Featurer(embedding_dim, depth)
        self.classifier = ClassificationHead(embedding_dim, n_classes)

    def forward(self, x):

        x = self.featurer(x)
        x = self.classifier(x)

        return x


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


class Featurer_kaggle(nn.Module):
    def __init__(self, embedding_dim=10, depth=3):
        super(Featurer_kaggle, self).__init__()
        self.extend = Extend()

        self.change = nn.Linear(22, 16)
        self.attention = ChannelAttention()
        self.embedding = PatchEmbedding(embedding_dim)
        self.encoder = TransformerEncoder(depth, embedding_dim)

    def forward(self, x):

        x = self.extend(x)
        x = einops.rearrange(self.change(x), 'b a l c -> b a c l')
        x = self.attention(x)
        x = self.embedding(x)
        x = self.encoder(x)

        return x


class ViT_kaggle(nn.Sequential):
    def __init__(self, embedding_dim=10, depth=3, n_classes=2):
        super(ViT_kaggle, self).__init__()

        self.featurer = Featurer_kaggle(embedding_dim, depth)
        self.classifier = ClassificationHead(embedding_dim, n_classes)

    def forward(self, x):

        x = self.featurer(x)
        x = self.classifier(x)

        return x


if __name__ == '__main__':

    # # test ChannelAttention()
    # inputs1 = torch.ones((128, 1, 16, 1024))  # B 1 C W
    # model1 = ChannelAttention()
    # outputs1 = model1(inputs1)
    #
    # # test PatchEmbedding()
    # inputs2 = outputs1  # B 1 C W
    # model2 = PatchEmbedding(10)
    # outputs2 = model2(inputs2)
    #
    # # test TransformerEncoder()
    # inputs3 = outputs2
    # model3 = TransformerEncoder(3, 10)
    # outputs3 = model3(inputs3)
    #
    # # test ClassificationHead()
    # inputs4 = outputs3
    # model4 = ClassificationHead(10, 2)
    # outputs4 = model4(inputs3)

    # test ViT()
    inputs1 = torch.ones((128, 1, 1024, 22))  # B 1 W C
    model1 = ViT()
    outputs1 = model1(inputs1)

    inputs2 = torch.ones((128, 1, 800, 16))  # B 1 W C
    model2 = ViT_kaggle()
    outputs2 = model2(inputs2)

    # print number of parameters
    total_parameter = sum(param.numel() for param in model1.parameters())
    print('total parameters in the model is {}'.format(total_parameter))
    total_training_parameter = sum(param.numel() for param in model1.parameters() if param.requires_grad)
    print('total training parameters in the model is {}'.format(total_parameter))

    print('VIT')
