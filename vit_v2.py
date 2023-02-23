"""
Transformer for EEG classification
"""


import argparse
import os
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from torchsummary import summary

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('/home/syh/Documents/MI/code/Trans/TensorBoardX/')

# torch.cuda.set_device(6)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 10, emb_size: int = 10, img_size: int = 1000):
        self.patch_size = patch_size
        super().__init__()
        self.reduce_channel = nn.Sequential(
            nn.Conv2d(3, 3, (4, 1), (4, 1)),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2),

            # nn.Conv2d(2, 2, (22, 1), (1, 1)),
            # nn.BatchNorm2d(2),
            # nn.LeakyReLU(0.2),
            # nn.ELU(),
        )
        self.projection = nn.Sequential(
            # nn.Conv2d(1, emb_size, (1, 5), stride=(1, 5)),
            nn.Conv2d(in_channels, emb_size, (16, 5), stride=(1, 5)),# 5 is better than 10, (16,5),(1,5)
            # nn.MaxPool2d( kernel_size=(1,5), stride=(1,5)),
            Rearrange('b e (h) (w) -> b (h w) e'),

        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # self.positions = nn.Parameter(torch.randn((100 + 1, emb_size)))
        # self.positions = nn.Parameter(torch.randn((2200 + 1, emb_size)))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        # x = self.spatial(x)
        # x=rearrange(x,'b o c s -> b o (s c)')
        # x = rearrange(x, 'b o (h s) -> b o h s',h=1)
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # cls
        x = torch.cat([cls_tokens, x], dim=1)
        # position
        # x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 10, num_heads: int = 5, dropout: float = 0):
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
        att = F.softmax(energy, dim=-1) / scaling
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
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 10,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.3,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
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
    def __init__(self, depth: int = 6, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 10, n_classes: int = 4):
        super().__init__(
            # Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, 64),
            nn.LeakyReLU(),
            nn.LayerNorm(64),

            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.LayerNorm(32),

            nn.Linear(32, n_classes),

        )

# class ClassificationHead(nn.Sequential):
#     def __init__(self, emb_size: int = 10, n_classes: int = 4):
#         super().__init__()
#         self.layer1=nn.Sequential(
#             # Reduce('b n e -> b e', reduction='mean'),
#             nn.LayerNorm(emb_size),
#             nn.Linear(emb_size, 64),
#             nn.LeakyReLU(),
#             nn.LayerNorm(64),
#
#             nn.Linear(64, 32),
#             nn.LeakyReLU(),
#             nn.LayerNorm(32),
#
#         )
#         self.layer2=nn.Sequential(nn.Linear(32, n_classes),)
#
#     def forward(self,x):
#         f=self.layer1(x)
#         x=self.layer2(f)
#         return x,f



class Reduce_channel(nn.Sequential):
    def __init__(self,h,stride):
        super().__init__(
            nn.Conv2d(3, 3, (h, 1), (stride, 1)),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2),

        )

class ViT(nn.Sequential):
    def __init__(self,
                in_channels: int = 3,
                patch_size: int = 16,#16
                emb_size: int = 15,#15
                img_size: int = 1000,
                depth: int = 3,#3
                n_classes: int = 4,
                **kwargs):
        super().__init__(
            # channel_attention(),
            # ResidualAdd(
            #     nn.Sequential(
            #         nn.LayerNorm(1000),
            #         channel_attention(),
            #         nn.Dropout(0.4),
            #     )
            # ),

            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            Reduce('b n e -> b e', reduction='mean'),
            # Rearrange('b n e -> b (n e)'),
            # ClassificationHead(emb_size, n_classes)
        )


class temporalConv(nn.Module):
    def __init__(self):
        super(temporalConv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1, 23), stride=(1, 3)),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1, 17), stride=(1, 1)),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1, 7), stride=(1, 1)),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2)
        )


    def forward(self, x):
        out = self.layer1(x)
        out=self.layer2(out)
        # out=self.layer3(out)
        return out

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.time=temporalConv()
        self.transformer=ViT(in_channels=3)
    def forward(self,x):

        out = self.time(x)
        out=self.transformer(out)
        return out

'''
avoid many parameters during calculating attention score and outputting the value with attention
otherwise
overfitting is obvious

some tricks which avoid overfitting can reduce test loss
BEST performance currently: test loss~0.53, test AVE acc~0.7, test BEST acc~0.833. ~ means about
'''
class channel_attention(nn.Module):
    def __init__(self,sequence_num=1000,inter=10):#(batch,1,channel,sequence)
        super(channel_attention,self).__init__()
        self.sequence_num=sequence_num
        self.inter=inter
        self.extract_sequence=int(self.sequence_num/self.inter)

        self.query=nn.Sequential(
            nn.Linear(16,16),
            # nn.LeakyReLU(),
            nn.LayerNorm(16),
            nn.Dropout(0.5)
        )
        self.key=nn.Sequential(
            nn.Linear(16,16),
            # nn.LeakyReLU(),
            nn.LayerNorm(16),
            nn.Dropout(0.5)
        )
        self.conv=nn.Sequential(
            nn.Conv2d(3,1,kernel_size=(1,1),stride=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1),
            nn.Dropout(0.2),
        )
        # self.value=nn.Sequential(
        #     nn.Linear(self.sequence_num,self.sequence_num),
        #     nn.LayerNorm(self.sequence_num)
        #     nn.ReLU(),
        #     nn.Dropout(0.2)
        # )
        '''
        self.value has too many parameters. 
        so transpose the input on channel and sequence dimension. the transposed input multiply with weight [22,22] which is much smaller than [1000,1000]. transpose back finally.
        self.projection works like that.
        '''
        self.value=self.key
        self.projection=nn.Sequential(
            nn.Linear(16,16),
            # nn.LeakyReLU(),
            nn.LayerNorm(16),
            nn.Dropout(0),# no leakyrelu,yes layernorm
        )

        self.drop_out=nn.Dropout(0.4)
        self.pooling=nn.AvgPool2d(kernel_size=(1,self.inter),stride=(1,self.inter))
        # self.pooling = nn.MaxPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))# maxpooling is better than averagepooling


        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0.0)

    def forward(self,x):

        temp=rearrange(x,'b o c s->b o s c')
        temp_query=rearrange(self.query(temp),'b o s c -> b o c s')
        temp_key=rearrange(self.key(temp),'b o s c -> b o c s')

        # channel_query=self.pooling(self.query(x))
        # channel_key=self.pooling(self.key(x))
        channel_query=self.conv(temp_query)
        channel_query=self.pooling(channel_query)
        channel_key=self.conv(temp_key)
        channel_key=self.pooling(channel_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten=torch.einsum('b o c s, b o m s -> b o c m', channel_query,channel_key)/scaling
        channel_atten=self.drop_out(channel_atten)
        channel_atten_score=F.softmax(channel_atten,dim=-1)

        # x=rearrange(x,'b o c s -> b o s c')
        # x=self.projection(x)
        # x=rearrange(x,'b o s c -> b o c s')

        out=torch.einsum('b t c s, b o c m -> b t c s', x,channel_atten_score)

        '''
        projections after or before multiplying with attention score are almost the same.
        '''

        out=rearrange(out,'b o c s -> b o s c')
        out=self.projection(out)
        out=rearrange(out,'b o s c -> b o c s')
        return out

