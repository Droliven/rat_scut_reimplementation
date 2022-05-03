#!/usr/bin/env python
# encoding: utf-8
'''
@project : rat_scut_reimplementation
@file    : model.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-05-03 10:52
'''
import torch
import torch.nn as nn
import copy

from .multi_head_attention import MultiHeadedAttention
from .operations import PositionalEncoding
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .feedforward import PositionwiseFeedForward


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, batch_size, coin_num, window_size, feature_number, d_model_Encoder, d_model_Decoder, encoder,
                 decoder, price_series_pe, local_price_pe, local_context_length):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size
        self.coin_num = coin_num
        self.window_size = window_size
        self.feature_number = feature_number
        self.d_model_Encoder = d_model_Encoder
        self.d_model_Decoder = d_model_Decoder
        self.linear_price_series = nn.Linear(in_features=feature_number, out_features=d_model_Encoder)
        self.linear_local_price = nn.Linear(in_features=feature_number, out_features=d_model_Decoder)
        self.price_series_pe = price_series_pe
        self.local_price_pe = local_price_pe
        self.local_context_length = local_context_length
        self.linear_out = nn.Linear(in_features=1 + d_model_Encoder, out_features=1)
        self.linear_out2 = nn.Linear(in_features=1 + d_model_Encoder, out_features=1)
        self.bias = torch.nn.Parameter(torch.zeros([1, 1, 1]))
        self.bias2 = torch.nn.Parameter(torch.zeros([1, 1, 1]))

    def forward(self, price_series, local_price_context, previous_w, price_series_mask, local_price_mask,
                padding_price):  ##[4, 128, 31, 11]
        # price_series:[4,128,31,11]
        price_series = price_series / price_series[0:1, :, -1:, :]
        price_series = price_series.permute(3, 1, 2, 0)  # [4,128,31,11]->[11,128,31,4]
        price_series = price_series.contiguous().view(price_series.size()[0] * price_series.size()[1], self.window_size,
                                                      self.feature_number)  # [11,128,31,4]->[11*128,31,4]
        price_series = self.linear_price_series(price_series)  # [11*128,31,3]->[11*128,31,2*12]
        price_series = self.price_series_pe(price_series)  # [11*128,31,2*12]
        price_series = price_series.view(self.coin_num, -1, self.window_size,
                                         self.d_model_Encoder)  # [11*128,31,2*12]->[11,128,31,2*12]
        encode_out = self.encoder(price_series, price_series_mask)
        #        encode_out=self.linear_src_2_embedding(encode_out)
        ###########################padding price#######################################################################################
        if (padding_price is not None):
            local_price_context = torch.cat([padding_price, local_price_context],
                                            2)  # [11,128,5-1,4] cat [11,128,1,4] -> [11,128,5,4]
            local_price_context = local_price_context.contiguous().view(
                local_price_context.size()[0] * price_series.size()[1], self.local_context_length * 2 - 1,
                self.feature_number)  # [11,128,5,4]->[11*128,5,4]
        else:
            local_price_context = local_price_context.contiguous().view(
                local_price_context.size()[0] * price_series.size()[1], 1, self.feature_number)
        ##############Divide by close price################################
        local_price_context = local_price_context / local_price_context[:, -1:, 0:1]
        local_price_context = self.linear_local_price(local_price_context)  # [11*128,5,4]->[11*128,5,2*12]
        local_price_context = self.local_price_pe(local_price_context)  # [11*128,5,2*12]
        if (padding_price is not None):
            padding_price = local_price_context[:, :-self.local_context_length, :]  # [11*128,5-1,2*12]
            padding_price = padding_price.view(self.coin_num, -1, self.local_context_length - 1,
                                               self.d_model_Decoder)  # [11,128,5-1,2*12]
        local_price_context = local_price_context[:, -self.local_context_length:, :]  # [11*128,5,2*12]
        local_price_context = local_price_context.view(self.coin_num, -1, self.local_context_length,
                                                       self.d_model_Decoder)  # [11,128,5,2*12]
        #################################padding_price=None###########################################################################
        decode_out = self.decoder(local_price_context, encode_out, price_series_mask, local_price_mask, padding_price)
        decode_out = decode_out.transpose(1, 0)  # [11,128,1,2*12]->#[128,11,1,2*12]
        decode_out = torch.squeeze(decode_out, 2)  # [128,11,1,2*12]->[128,11,2*12]
        previous_w = previous_w.permute(0, 2, 1)  # [128,1,11]->[128,11,1]
        out = torch.cat([decode_out, previous_w], 2)  # [128,11,2*12]  cat [128,11,1] -> [128,11,2*12+1]
        ###################################  Decision making ##################################################
        out2 = self.linear_out2(out)  # [128,11,2*12+1]->[128,11,1]
        out = self.linear_out(out)  # [128,11,2*12+1]->[128,11,1]

        bias = self.bias.repeat(out.size()[0], 1, 1)  # [128,1,1]
        bias2 = self.bias2.repeat(out2.size()[0], 1, 1)  # [128,1,1]

        out = torch.cat([bias, out], 1)  # [128,11,1] cat [128,1,1] -> [128,12,1]
        out2 = torch.cat([bias2, out2], 1)  # [128,11,1] cat [128,1,1] -> [128,12,1]

        out = out.permute(0, 2, 1)  # [128,1,12]
        out2 = out2.permute(0, 2, 1)  # [128,1,12]

        out = torch.softmax(out, dim=-1)
        out2 = torch.softmax(out2, dim=-1)

        out = out * 2
        out2 = -out2
        return out + out2  # [128,1,12]


def make_model(batch_size=128, coin_num=11, window_size=31, feature_number=4, N=1, d_model_Encoder=24, d_model_Decoder=24,
               d_ff_Encoder=24, d_ff_Decoder=24, h=2, dropout=0.01, local_context_length=5):
    "Helper: Construct a model from hyperparameters."
    attn_Encoder = MultiHeadedAttention(True, h, d_model_Encoder, 0.1, local_context_length)
    attn_Decoder = MultiHeadedAttention(True, h, d_model_Decoder, 0.1, local_context_length)
    attn_En_Decoder = MultiHeadedAttention(False, h, d_model_Decoder, 0.1, 1)
    ff_Encoder = PositionwiseFeedForward(d_model_Encoder, d_ff_Encoder, dropout)
    ff_Decoder = PositionwiseFeedForward(d_model_Decoder, d_ff_Decoder, dropout)
    position_Encoder = PositionalEncoding(d_model_Encoder, 0, dropout)
    position_Decoder = PositionalEncoding(d_model_Decoder, window_size - local_context_length * 2 + 1, dropout)

    model = EncoderDecoder(batch_size, coin_num, window_size, feature_number, d_model_Encoder, d_model_Decoder, Encoder(
        EncoderLayer(d_model_Encoder, copy.deepcopy(attn_Encoder), copy.deepcopy(ff_Encoder), dropout), N), Decoder(
        DecoderLayer(d_model_Decoder, copy.deepcopy(attn_Decoder), copy.deepcopy(attn_En_Decoder),
                     copy.deepcopy(ff_Decoder), dropout), N), copy.deepcopy(position_Encoder),
                           # price series position ecoding
                           copy.deepcopy(position_Decoder),  # local_price_context position ecoding
                           local_context_length)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
