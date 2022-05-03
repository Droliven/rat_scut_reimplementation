#!/usr/bin/env python
# encoding: utf-8
'''
@project : rat_scut_reimplementation
@file    : optimizer.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-05-03 16:11
'''


class Optimizer:
    "Optim wrapper that implements rate."

    # 512, 1, 400
    def __init__(self, model_size=5120, factor=1e-4, warmup=0, optimizer=None):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if self.warmup == 0:
            return self.factor
        else:
            return self.factor * \
                   (self.model_size ** (-0.5) *
                    min(step ** (-0.5), step * self.warmup ** (-1.5)))
