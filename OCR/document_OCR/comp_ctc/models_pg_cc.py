import sys
import random
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tanh, log_softmax, softmax, relu, sigmoid
from torch.nn import Conv1d, Conv2d, Dropout, Linear, AdaptiveMaxPool2d, InstanceNorm1d, AdaptiveMaxPool1d, AdaptiveAvgPool2d
from torch.nn import ReLU, GELU
from torch.nn import InstanceNorm1d, InstanceNorm2d, LayerNorm
from torch.nn import Flatten, LSTM, Embedding, MultiheadAttention
from torch.nn import Dropout, Dropout1d
from basic.models import get_activation, get_norm, MixDropout, MixDropout1d, DepthSepConv2D



class ToyDecoder(nn.Module):
    def __init__(self, params, param):
        super(ToyDecoder, self).__init__()
        self.input_size = params["features_size"]
        self.vocab_size = params["vocab_size"]
        num_class = self.vocab_size + 1
        self.end_conv = Conv2d(in_channels=self.input_size, out_channels=num_class, kernel_size=1)

    def forward(self, x, *args):
        """
        x (B, C, H, W)
        """
        out = self.end_conv(x)
        out = log_softmax(out, dim=1)
        return out


class DecoderH(nn.Module):
    def __init__(self, params, param):
        super(DecoderH, self).__init__()
        self.features_size = params["features_size"]
        self.vocab_size = params["vocab_size"]
        self.ada_pool_width = param.get('ada_pool_width', 100)
        self.ks_1d = param.get('kernel_size_1d', 3)
        pool_method = param.get('pool_method', 'avg')

        num_class = self.vocab_size + 1
        self.end_conv = Conv2d(in_channels=self.features_size, out_channels=num_class, kernel_size=1)
        """horizontal features for transformer"""
        if pool_method == 'avg':
            self.ada_pool = AdaptiveAvgPool2d((None, self.ada_pool_width))
        elif pool_method == 'max':
            self.ada_pool = AdaptiveMaxPool2d((None, self.ada_pool_width))
        self.dense_width = Linear(self.ada_pool_width, 1)

        self.h_pos = nn.Sequential(
            nn.Conv1d(self.features_size, 128, kernel_size=self.ks_1d, padding=self.ks_1d//2),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=self.ks_1d, padding=self.ks_1d//2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=self.ks_1d, padding=self.ks_1d//2)
        )

        self.end_conv_pos = Conv1d(64, out_channels=2, kernel_size=1)
        self.dropout = param.get('dropout', 0.)
        self.dropout = MixDropout1d(self.dropout)

    def forward(self, x, *args):
        """
        x (B, C, H, W)
        """
        out = self.end_conv(x)
        out = log_softmax(out, dim=1)
        h_features = self.ada_pool(x) # (B, C, H, 100)
        h_features = self.dense_width(h_features) # (B, C, H, 1)
        h_features = self.h_pos(h_features.squeeze(dim=-1))
        h_features = self.dropout(h_features)
        out_h = self.end_conv_pos(h_features) # (B, 2, H, 1)
        out_h = log_softmax(out_h, dim=1)
        return out, out_h


# class DecoderH_v2(nn.Module):
#     def __init__(self, params, param):
#         super(DecoderH_v2, self).__init__()
#         self.features_size = params["features_size"]
#         self.vocab_size = params["vocab_size"]
#         self.ada_pool_width = param.get('ada_pool_width', 100)
#         self.ks_1d = param.get('kernel_size_1d', 3)
#         pool_method = param.get('pool_method', 'avg')
#
#         num_class = self.vocab_size + 1
#         self.end_conv = Conv2d(in_channels=self.features_size, out_channels=num_class, kernel_size=1)
#         """horizontal features for transformer"""
#         if pool_method == 'avg':
#             self.ada_pool = AdaptiveAvgPool2d((None, self.ada_pool_width))
#         elif pool_method == 'max':
#             self.ada_pool = AdaptiveMaxPool2d((None, self.ada_pool_width))
#         self.dense_width = Linear(self.ada_pool_width, 1)
#
#         self.h_conv1 = nn.Conv1d(self.features_size, 128, kernel_size=self.ks_1d, padding=self.ks_1d//2)
#         self.h_conv2 = nn.Conv1d(128, 128, kernel_size=self.ks_1d, padding=self.ks_1d//2)
#         self.h_conv3 = nn.Conv1d(128, 128, kernel_size=self.ks_1d, padding=self.ks_1d//2)
#         self.activation = nn.ReLU()
#
#
#         self.end_conv_pos = Conv1d(128, out_channels=2, kernel_size=1)
#         self.dropout = param.get('dropout', 0.)
#         self.dropout = MixDropout1d(self.dropout)
#
#     def forward(self, x, *args):
#         """
#         x (B, C, H, W)
#         """
#         out = self.end_conv(x)
#         out = log_softmax(out, dim=1)
#         h_features = self.ada_pool(x) # (B, C, H, 100)
#         h_features = self.dense_width(h_features) # (B, C, H, 1)
#
#         # h_features = self.h_pos(h_features.squeeze(dim=-1))
#         # h_features = self.dropout(h_features)
#
#         pos = random.randint(1, 2)
#
#         h_features = self.h_conv1(h_features.squeeze(dim=-1))
#         h_features = self.activation(h_features)
#         if pos == 1:
#             h_features = self.dropout(h_features)
#
#         h_features = self.h_conv2(h_features)
#         h_features = self.activation(h_features)
#         if pos == 2:
#             h_features = self.dropout(h_features)
#
#         h_features = self.h_conv3(h_features)
#
#         out_h = self.end_conv_pos(h_features) # (B, 2, H, 1)
#         out_h = log_softmax(out_h, dim=1)
#         return out, out_h


class LocalDWConvAttention(nn.Module):
    def __init__(self, params, param):
        super(LocalDWConvAttention, self).__init__()
        self.input_size = params["features_size"]
        # self.vocab_size = params["vocab_size"]
        self.attn_activation = params.get('attn_activation', 'softmax')
        self.la_conv = nn.Sequential(
            DepthSepConv2D(self.input_size, 128, ks=3, stride=(1, 1)),
            nn.ReLU(),
            DepthSepConv2D(128, 64, ks=3, stride=(1, 1)),
            nn.ReLU(),
            DepthSepConv2D(64, 32, ks=3, stride=(1, 1)),
            nn.ReLU(),
            DepthSepConv2D(32, 9, ks=3, stride=(1, 1))
        )

    def forward(self, x):
        """
        x (B, C, H, W)
        """
        char_attn = self.la_conv(x) # (B, 9, H, W)
        if self.attn_activation == 'softmax':
            char_attn = softmax(char_attn, dim=1).permute(0, 2, 3, 1).unflatten(dim=-1, sizes=(3, 3)) # (B, H, W, 3, 3)
        elif self.attn_activation == 'sigmoid':
            char_attn = sigmoid(char_attn).permute(0, 2, 3, 1).unflatten(dim=-1, sizes=(3, 3)) # (B, H, W, 3, 3)
        else:
            raise NotImplementedError
        # out1 = DynamicConv3x3(x, char_attn)
        out = dynamic_conv(x, char_attn)
        # torch.allclose(out, out1, atol=1e-3)
        return out, char_attn



def dynamic_conv(x, filter):
    # x: (B, C, H, W)
    # filter: (B, H, W, 3, 3)
    B, C, H, W = x.shape
    x_ = F.unfold(x, kernel_size=(3, 3), padding=(1, 1)) # (B, C*3*3, H*W)
    # i = 0
    # j = 0
    # c = 0
    # torch.allclose(x_[0, c*9: (c+1)*9, W*i+j].unflatten(dim=0, sizes=(3, 3)), F.pad(x, (1, 1, 1, 1))[0, c, i:(i+3), j:(j+3)])
    filter_ = filter.flatten(start_dim=3, end_dim=4).flatten(start_dim=1, end_dim=2).permute(0, 2, 1) # (B, 3*3, H*W)
    filter_ = torch.cat([filter_]*C, dim=1) # (B, C*3*3, H, W)
    res_ = x_ * filter_
    res_ = res_.unflatten(dim=1, sizes=(C, 9)).unflatten(dim=-1, sizes=x.shape[-2:])
    res = torch.sum(res_, dim=2)
    # x__ = F.fold(x_, x.shape[-2:], kernel_size=(3, 3))
    return res

# def iterate_regions(image):
#     '''
#     Generates all possible 3x3 image regions using valid padding.
#     - image is a 2d numpy array
#     '''
#     b, c, h, w = image.shape
#     image = F.pad(image, (1, 1, 1, 1))
#     for i in range(h):
#         for j in range(w):
#             im_region = image[:, :, i:(i + 3), j:(j + 3)]
#             yield im_region, i, j
#     # 将 im_region, i, j 以 tuple 形式存储到迭代器中
#     # 以便后面遍历使用
#
#
# def DynamicConv3x3(input, filters):
#     '''
#     Performs a forward pass of the conv layer using the given input.
#     Returns a 3d numpy array with dimensions (h, w, num_filters).
#     - input is a 2d numpy array
#     '''
#     # input 为 image，即输入数据
#     # output 为输出框架，默认都为 0，都为 1 也可以，反正后面会覆盖
#     # input: (B, C, H, W)
#     # filters: (B, H, W, 3, 3)
#     # output: 26x26x8
#     b, c, h, w = input.shape
#     output = torch.zeros((b, c, h, w), dtype=input.dtype, device=input.device)
#
#     for im_region, i, j in iterate_regions(input):
#         # 卷积运算，点乘再相加，ouput[i, j] 为向量，8 层
#         filter = filters[:, i, j, :, :].unsqueeze(dim=1) # (B, 1, 3, 3)
#         output[:, :, i, j] = torch.sum(im_region * torch.cat([filter]*c, dim=1), dim=(2, 3))
#     # 最后将输出数据返回，便于下一层的输入使用
#     return output
#
#
# import numpy as np
#
# def overlap_ratio(dynk1, dynk2):
#     """
#     dynk1, dynk2: (W, 3, 3)
#     """
#     assert dynk1.shape == dynk2.shape
#     overlap_sum = np.sum(dynk1[:, :2, :]) + np.sum(dynk2[:, 1:, :])
#     return overlap_sum / (dynk1.shape[0] + dynk2.shape[0])
