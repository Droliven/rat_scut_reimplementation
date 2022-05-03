#!/usr/bin/env python
# encoding: utf-8
'''
@project : rat_scut_reimplementation
@file    : decoder.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-04-29 17:01
'''
import torch
import torch.nn as nn
import copy

from .operations import LayerNorm, SublayerConnection

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, price_series_mask, local_price_mask, padding_price):
        for layer in self.layers:
            x = layer(x, memory, price_series_mask, local_price_mask, padding_price)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([copy.deepcopy(SublayerConnection(size, dropout)) for _ in range(3)])


    def forward(self, x, memory, price_series_mask, local_price_mask, padding_price):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, local_price_mask, padding_price, padding_price))
        x = x[:, :, -1:, :]
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, price_series_mask, None, None))
        return self.sublayer[2](x, self.feed_forward)
