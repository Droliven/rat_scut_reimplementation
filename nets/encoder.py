#!/usr/bin/env python
# encoding: utf-8
'''
@project : rat_scut_reimplementation
@file    : encoder.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-05-03 10:44
'''
import torch
import torch.nn as nn
import copy

from .operations import LayerNorm, SublayerConnection

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            #            print("Encoder:",x)
            x = layer(x, mask)
        #            print("Encoder:",x.size())
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([copy.deepcopy(SublayerConnection(size, dropout)) for _ in range(2)])
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, None, None))
        return self.sublayer[1](x, self.feed_forward)
