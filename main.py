#!/usr/bin/env python
# encoding: utf-8
'''
@project : rat_scut_reimplementation
@file    : main.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-04-29 15:24
'''
# ****************************************************************************************************************
# *********************************************** 环境部分 ********************************************************
# ****************************************************************************************************************

import numpy as np
import random
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def seed_torch(seed=3450):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

seed_torch()

# ****************************************************************************************************************
# *********************************************** 主体部分 ********************************************************
# ****************************************************************************************************************

import argparse
import pandas as pd
from pprint import pprint

from runs import Runner

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--exp_name', type=str, default="rat", help="")
parser.add_argument('--is_train', type=bool, default='', help="")
parser.add_argument('--is_load', type=bool, default='1', help="")
parser.add_argument('--model_path', type=str, default=os.path.join(r"./ckpt/models", f"step{2000}.pth"), help="")

args = parser.parse_args()

print("\n================== Arguments =================")
pprint(vars(args), indent=4)
print("==========================================\n")
csv_dir = os.path.join(r.cfg.ckpt_dir + "test_summary.csv")
r = Runner()

if args.is_load:
    r.restore(args.model_path)

if args.is_train:
    r.run()
else:
    r.model.eval()
    with torch.no_grad():
        tst_loss, tst_portfolio_value, SR, CR, St_v_list, tst_pc_array, TO = r.test_online()

    csv_dir = os.path.join(r.cfg.ckpt_dir,  "test_summary.csv")
    d = {"step": [-1],
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

    print(
        f"Eval online Step: {-1}: Loss: {tst_loss.item()} | Portfolio_Value: {tst_portfolio_value.item()} | SR: {SR.item()} | CR: {CR.item()} | TO: {TO.item()}")
