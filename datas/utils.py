#!/usr/bin/env python
# encoding: utf-8
'''
@project : rat_scut_reimplementation
@file    : utils.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-05-03 11:38
'''
import pandas as pd
import numpy as np
import time
from datetime import datetime
import torch

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(local_price_context,batch_size):
    "Create a mask to hide padding and future words."
    local_price_mask = (torch.ones(batch_size,1,1)==1)
    local_price_mask = local_price_mask & (subsequent_mask(local_price_context.size(-2)).type_as(local_price_mask.data))
    return local_price_mask


def panel_fillna(panel, type="bfill"):
    """
    fill nan along the 3rd axis
    :param panel: the panel to be filled
    :param type: bfill or ffill
    """
    # frames = {}
    # for item in panel.items: # ['close', 'high', 'low', 'open']
    #     if type == "both":
    #         frames[item] = panel.loc[item].fillna(axis=1, method="bfill").fillna(axis=1, method="ffill")
    #     else:
    #         frames[item] = panel.loc[item].fillna(axis=1, method=type)
    # return pd.Panel(frames)

    for item in panel.columns.levels[0]:  # ['close', 'high', 'low', 'open']
        if type == "both":
            panel[item] = panel[item].fillna(axis=0, method="bfill").fillna(axis=0, method="ffill")
        else:
            panel[item] = panel[item].fillna(axis=0, method=type)
    return panel


def get_type_list(feature_number):
    """
    :param feature_number: an int indicates the number of features
    :return: a list of features n
    """
    if feature_number == 1:
        type_list = ["close"]
    elif feature_number == 2:
        type_list = ["close", "volume"]
        raise NotImplementedError("the feature volume is not supported currently")
    elif feature_number == 3:
        type_list = ["close", "high", "low"]
    elif feature_number == 4:
        type_list = ["close", "high", "low", "open"]
    else:
        raise ValueError("feature number could not be %s" % feature_number)
    return type_list


def get_volume_forward(time_span, portion, portion_reversed):
    volume_forward = 0
    if not portion_reversed:
        volume_forward = time_span*portion
    return volume_forward

class Experience:
    def __init__(self, state_index):
        self.state_index = int(state_index)


class ReplayBuffer:
    def __init__(self, start_index, end_index, batch_size, is_permed, coin_number, sample_bias=1.0):
        """
        :param start_index: start index of the training set on the global data matrices
        :param end_index: end index of the training set on the global data matrices
        """
        self.__coin_number = coin_number
        self.__experiences = [Experience(i) for i in range(start_index, end_index)]
        self.__is_permed = is_permed
        # NOTE: in order to achieve the previous w feature
        self.__batch_size = batch_size
        self.__sample_bias = sample_bias
        print("buffer_bias is %f" % sample_bias)

    def append_experience(self, state_index):
        self.__experiences.append(Experience(state_index))
        print("a new experience, indexed by %d, was appended" % state_index)

    def __sample(self, start, end, bias):
        """
        @:param end: is excluded
        @:param bias: value in (0, 1)
        """
        # TODO: deal with the case when bias is 0
        ran = np.random.geometric(bias)
        while ran > end - start:
            ran = np.random.geometric(bias)
        result = end - ran
        return result

    def next_experience_batch(self):
        # First get a start point randomly
        batch = []
        if self.__is_permed:
            for i in range(self.__batch_size):
                batch.append(self.__experiences[self.__sample(self.__experiences[0].state_index,
                                                              self.__experiences[-1].state_index,
                                                              self.__sample_bias)])
        else:
            batch_start = self.__sample(0, len(self.__experiences) - self.__batch_size,
                                        self.__sample_bias)
            batch = self.__experiences[batch_start:batch_start+self.__batch_size]
        return batch

def parse_time(time_string):
    return time.mktime(datetime.strptime(time_string, "%Y/%m/%d").timetuple())

def max_drawdown(pc_array):
    """calculate the max drawdown with the portfolio changes
    @:param pc_array: all the portfolio changes during a trading process
    @:return: max drawdown
    """
    portfolio_values = []
    drawdown_list = []
    max_benefit = 0
    for i in range(pc_array.shape[0]):
        if i > 0:
            portfolio_values.append(portfolio_values[i - 1] * pc_array[i])
        else:
            portfolio_values.append(pc_array[i])
        if portfolio_values[i] > max_benefit:
            max_benefit = portfolio_values[i]
            drawdown_list.append(0.0)
        else:
            drawdown_list.append(1.0 - portfolio_values[i] / max_benefit)
    return max(drawdown_list)
