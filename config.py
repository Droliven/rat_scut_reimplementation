#!/usr/bin/env python
# encoding: utf-8
'''
@project : rat_scut_reimplementation
@file    : config.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-04-29 15:32
'''

import os
import getpass

class Config:
    def __init__(self):

        self.x_window_size = 31
        self.batch_size = 128
        self.multihead_num = 2
        self.local_context_length = 5
        self.model_dim = 12
        self.weight_decay = 5e-8
        self.daily_interest_rate = 0.001

        self.total_step = 50000

        self.coin_num = 11
        self.feature_number = 4
        self.output_step = 1000
        self.test_portion = 0.08
        self.trading_consumption = 0.0025
        self.variance_penalty = 0.0
        self.cost_penalty = 0.0
        self.learning_rate = 1e-4

        self.start = "2016/01/01"
        self.end = "2018/01/01"

        self.model_index = [1, 2, 3, 4, 5]
        self.devive = "cuda:0"

        user = getpass.getuser()
        if user == "Drolab":
            self.base_dir = os.path.join(r"G:\second_model_report_data\portfolio_management\datas\rat_scut")
        elif user == "songbo":
            if os.path.exists(os.path.join(r"/home/ml_group/songbo/danglingwei204")):
                self.base_dir = os.path.join(r"/home/ml_group/songbo/danglingwei204/datasets/financial_songbo/rat")

        self.pretrained_model_path = os.path.join(self.base_dir, "rat_scut.pkl")

        self.data_path = os.path.join(self.base_dir, "data.db")
        assert os.path.exists(self.data_path), f"Database_path {self.data_path} Not Exists!"

        self.ckpt_dir = os.path.join("./ckpt")
        os.makedirs(os.path.join(self.ckpt_dir, "models"), exist_ok=True)

        pass

if __name__ == '__main__':
    c = Config()
    pass