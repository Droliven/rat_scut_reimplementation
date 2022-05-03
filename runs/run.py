#!/usr/bin/env python
# encoding: utf-8
'''
@project : rat_scut_reimplementation
@file    : run.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-05-03 17:21
'''
from config import Config
from nets import make_model
from datas import DataMatrices
from datas.utils import parse_time, make_std_mask
from .losses import Batch_Loss, SimpleLossCompute, SimpleLossCompute_tst, Test_Loss
from .optimizer import Optimizer

import os
import numpy as np
import torch
import time
import pandas as pd
from tensorboardX import SummaryWriter

class Runner():
    def __init__(self, ):

        self.cfg = Config()

        self.model = make_model(self.cfg.batch_size, self.cfg.coin_num, self.cfg.x_window_size, self.cfg.feature_number,
                                N=1, d_model_Encoder=self.cfg.multihead_num * self.cfg.model_dim,
                                d_model_Decoder=self.cfg.multihead_num * self.cfg.model_dim,
                                d_ff_Encoder=self.cfg.multihead_num * self.cfg.model_dim,
                                d_ff_Decoder=self.cfg.multihead_num * self.cfg.model_dim, h=self.cfg.multihead_num,
                                dropout=0.01, local_context_length=self.cfg.local_context_length)

        if self.cfg.devive != "cpu":
            self.model.to(self.cfg.devive)

        print(f"Total params: {sum([p.numel() for p in self.model.parameters()]) / 1e6} M")

        self.optimizer = Optimizer(model_size=5120, factor=self.cfg.learning_rate, warmup=0,
                                   optimizer=torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98),
                                                              eps=1e-9, weight_decay=self.cfg.weight_decay))

        self.train_loss_compute = SimpleLossCompute(Batch_Loss(self.cfg.trading_consumption, self.cfg.daily_interest_rate / 24 / 2, self.cfg.variance_penalty, self.cfg.cost_penalty, True), self.optimizer)
        self.evaluate_loss_compute = SimpleLossCompute(Batch_Loss(self.cfg.trading_consumption, self.cfg.daily_interest_rate / 24 / 2, self.cfg.variance_penalty, self.cfg.cost_penalty, False), None)
        self.test_loss_compute = SimpleLossCompute_tst(Test_Loss(self.cfg.trading_consumption, self.cfg.daily_interest_rate / 24 / 2, self.cfg.variance_penalty,self.cfg.cost_penalty, False), None)

        self.data_matrices = DataMatrices(start=parse_time(self.cfg.start), end=parse_time(self.cfg.end),
                                          market="poloniex",
                                          feature_number=self.cfg.feature_number,
                                          window_size=self.cfg.x_window_size,
                                          online=False,
                                          period=1800,
                                          coin_filter=11,
                                          is_permed=False,
                                          buffer_bias_ratio=5e-5,
                                          batch_size=self.cfg.batch_size,  # 128,
                                          volume_average_days=30,
                                          test_portion=self.cfg.test_portion,  # 0.08,
                                          portion_reversed=False, database_path=self.cfg.data_path)
        self.summary = SummaryWriter(self.cfg.ckpt_dir)


    def save(self, checkpoint_path, best_err=-1, curr_err=-1):
        state = {
            # "lr": self.lr,
            # "best_err": best_err,
            # "curr_err": curr_err,
            "model": self.model.state_dict(),
            # "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        # self.optimizer.load_state_dict(state["optimizer"])
        # self.lr = state["lr"]
        # best_err = state['best_err']
        # curr_err = state["curr_err"]
        print("loaded from {}.".format(checkpoint_path))


    def train_one_step(self):
        batch = self.data_matrices.next_batch()
        batch_input = batch["X"]  # (128, 4, 11, 31)
        batch_y = batch["y"]  # (128, 4, 11)
        batch_last_w = batch["last_w"]  # (128, 11)
        batch_w = batch["setw"]
        #############################################################################
        previous_w = torch.tensor(batch_last_w, dtype=torch.float).cuda()
        previous_w = torch.unsqueeze(previous_w, 1)  # [128, 11] -> [128,1,11]
        batch_input = batch_input.transpose((1, 0, 3, 2))
        src = torch.tensor(batch_input, dtype=torch.float).cuda()
        price_series_mask = (torch.ones(src.size()[1], 1, self.cfg.x_window_size) == 1)  # [128, 1, 31]
        currt_price = src.permute((3, 1, 2, 0))  # [4,128,31,11]->[11,128,31,4]
        if (self.cfg.local_context_length > 1):
            padding_price = currt_price[:, :, -(self.cfg.local_context_length) * 2 + 1:-1, :]
        else:
            padding_price = None

        currt_price = currt_price[:, :, -1:, :]  # [11,128,31,4]->[11,128,1,4]
        trg_mask = make_std_mask(currt_price, src.size()[1])
        batch_y = batch_y.transpose((0, 2, 1))  # [128, 4, 11] ->#[128,11,4]
        trg_y = torch.tensor(batch_y, dtype=torch.float).cuda()
        out = self.model(src, currt_price, previous_w, price_series_mask, trg_mask, padding_price)
        new_w = out[:, :, 1:]  # 去掉cash
        new_w = new_w[:, 0, :]  # #[109,1,11]->#[109,11]
        new_w = new_w.detach().cpu().numpy()
        batch_w(new_w)

        loss, portfolio_value = self.train_loss_compute(out, trg_y)
        return loss, portfolio_value

    def eval_batch(self):
        tst_batch = self.data_matrices.get_test_set()
        tst_batch_input = tst_batch["X"]  # (128, 4, 11, 31)
        tst_batch_y = tst_batch["y"]
        tst_batch_last_w = tst_batch["last_w"]
        tst_batch_w = tst_batch["setw"]

        tst_previous_w = torch.tensor(tst_batch_last_w, dtype=torch.float).cuda()
        tst_previous_w = torch.unsqueeze(tst_previous_w, 1)  # [2426, 1, 11]
        tst_batch_input = tst_batch_input.transpose((1, 0, 2, 3))
        tst_batch_input = tst_batch_input.transpose((0, 1, 3, 2))
        tst_src = torch.tensor(tst_batch_input, dtype=torch.float).cuda()
        tst_src_mask = (torch.ones(tst_src.size()[1], 1, self.cfg.x_window_size) == 1)  # [128, 1, 31]
        tst_currt_price = tst_src.permute((3, 1, 2, 0))  # (4,128,31,11)->(11,128,31,3)
        #############################################################################
        if (self.cfg.local_context_length > 1):
            padding_price = tst_currt_price[:, :, -(self.cfg.local_context_length) * 2 + 1:-1, :]  # (11,128,8,4)
        else:
            padding_price = None
        #########################################################################

        tst_currt_price = tst_currt_price[:, :, -1:, :]  # (11,128,31,4)->(11,128,1,4)
        tst_trg_mask = make_std_mask(tst_currt_price, tst_src.size()[1])
        tst_batch_y = tst_batch_y.transpose((0, 2, 1))  # (128, 4, 11) ->(128,11,4)
        tst_trg_y = torch.tensor(tst_batch_y, dtype=torch.float).cuda()
        ###########################################################################################################
        tst_out = self.model(tst_src, tst_currt_price, tst_previous_w, tst_src_mask, tst_trg_mask, padding_price) # [128,1,11]   [128, 11, 31, 4])

        tst_loss, tst_portfolio_value = self.evaluate_loss_compute(tst_out, tst_trg_y)
        return tst_loss, tst_portfolio_value

    def test_online(self):
        tst_batch = self.data_matrices.get_test_set_online(self.data_matrices._test_ind[0], self.data_matrices._test_ind[-1], self.cfg.x_window_size)
        tst_batch_input = tst_batch["X"] # [1, 4, 11, 2806]
        tst_batch_y = tst_batch["y"] # [1, 4, 11, 2776]
        tst_batch_last_w = tst_batch["last_w"]  # [1, 11]
        tst_batch_w = tst_batch["setw"]

        tst_previous_w = torch.tensor(tst_batch_last_w, dtype=torch.float).cuda()
        tst_previous_w = torch.unsqueeze(tst_previous_w, 1) # [1, 1, 11]

        tst_batch_input = tst_batch_input.transpose((1, 0, 3, 2)) # [4, 1, 2806, 11]
        long_term_tst_src = torch.tensor(tst_batch_input, dtype=torch.float).cuda()
        #########################################################################################
        tst_src_mask = (torch.ones(long_term_tst_src.size()[1], 1, self.cfg.x_window_size) == 1)

        long_term_tst_currt_price = long_term_tst_src.permute((3, 1, 2, 0))
        long_term_tst_currt_price = long_term_tst_currt_price[:, :, self.cfg.x_window_size - 1:, :]
        ###############################################################################################
        tst_trg_mask = make_std_mask(long_term_tst_currt_price[:, :, 0:1, :], long_term_tst_src.size()[1])

        tst_batch_y = tst_batch_y.transpose((0, 3, 2, 1)) # [1, 2776, 11, 4]
        tst_trg_y = torch.tensor(tst_batch_y, dtype=torch.float).cuda()
        tst_long_term_w = []
        tst_y_window_size = len(self.data_matrices._test_ind) - self.cfg.x_window_size - 1 - 1
        for j in range(tst_y_window_size + 1):  # 0-9
            tst_src = long_term_tst_src[:, :, j:j + self.cfg.x_window_size, :]
            tst_currt_price = long_term_tst_currt_price[:, :, j:j + 1, :]
            if (self.cfg.local_context_length > 1):
                padding_price = long_term_tst_src[:, :,
                                j + self.cfg.x_window_size - 1 - self.cfg.local_context_length * 2 + 2:j + self.cfg.x_window_size - 1,
                                :]
                padding_price = padding_price.permute((3, 1, 2, 0))  # [4, 1, 2, 11] ->[11,1,2,4]
            else:
                padding_price = None
            out = self.model.forward(tst_src, tst_currt_price, tst_previous_w,
                                     # [109,1,11]   [109, 11, 31, 3]) torch.Size([109, 11, 3]
                                     tst_src_mask, tst_trg_mask, padding_price)
            if (j == 0):
                tst_long_term_w = out.unsqueeze(0)  # [1,109,1,12]
            else:
                tst_long_term_w = torch.cat([tst_long_term_w, out.unsqueeze(0)], 0)
            out = out[:, :, 1:]  # 去掉cash #[109,1,11]
            tst_previous_w = out
        tst_long_term_w = tst_long_term_w.permute(1, 0, 2, 3)  ##[2776,1,12, 1]->#[1,2776,1,12]
        tst_loss, tst_portfolio_value, SR, CR, St_v, tst_pc_array, TO = self.test_loss_compute(tst_long_term_w, tst_trg_y)
        return tst_loss, tst_portfolio_value, SR, CR, St_v, tst_pc_array, TO

    def run(self):
        "Standard Training and Logging Function"
        #### train
        for i in range(self.cfg.total_step):
            self.model.train()
            loss, portfolio_value = self.train_one_step()
            self.summary.add_scalar(f"Train/loss", loss, i)
            self.summary.add_scalar(f"Train/portfolio_value", portfolio_value, i)

            if (i % self.cfg.output_step == 0):
                print(f"Train Step: {i}: Loss per batch: {loss.item()} | Portfolio_Value: {portfolio_value.item()}")
            #### Eval 
            if (i % self.cfg.output_step == 0):
                self.model.eval()
                with torch.no_grad():
                    eval_loss, eval_portfolio_value = self.eval_batch()

                self.summary.add_scalar(f"Eval/loss", eval_loss, i)
                self.summary.add_scalar(f"Eval/portfolio_value", eval_portfolio_value, i)

                print(f"Test Step: {i}: Loss per batch: {eval_loss.item()} | Portfolio_Value: {eval_portfolio_value.item()}")
                
                #### Test
                self.model.eval()
                with torch.no_grad():
                    tst_loss, tst_portfolio_value, SR, CR, St_v_list, tst_pc_array, TO = self.test_online()

                csv_dir = os.path.join(self.cfg.ckpt_dir, "test_summary.csv")
                d = {"step": [i],
                     "fAPV": [tst_portfolio_value.item()],
                     "SR": [SR.item()],
                     "CR": [CR.item()],
                     "TO": [TO.item()],
                     "St_v": [''.join(str(e) + ', ' for e in St_v_list)],
                     "backtest_test_history": [''.join(str(e) + ', ' for e in tst_pc_array.cpu().numpy())],
                     }
                new_data_frame = pd.DataFrame(data=d).set_index("step")
                if os.path.isfile(csv_dir):
                    dataframe = pd.read_csv(csv_dir).set_index("step")
                    dataframe = dataframe.append(new_data_frame)
                else:
                    dataframe = new_data_frame
                dataframe.to_csv(csv_dir)

                print(f"Eval online Step: {i}: Loss: {tst_loss.item()} | Portfolio_Value: {tst_portfolio_value.item()} | SR: {SR.item()} | CR: {CR.item()} | TO: {TO.item()}")
                self.summary.add_scalar(f"Test/loss", tst_loss, i)
                self.summary.add_scalar(f"Test/portfolio_value", tst_portfolio_value, i)
                self.summary.add_scalar(f"Test/SR", SR, i)
                self.summary.add_scalar(f"Test/CR", CR, i)
                self.summary.add_scalar(f"Test/TO", TO, i)

                # 保存模型
                self.save(os.path.join(self.cfg.ckpt_dir, "models", f"step{i}.pth"))
                