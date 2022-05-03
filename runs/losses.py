#!/usr/bin/env python
# encoding: utf-8
'''
@project : rat_scut_reimplementation
@file    : losses.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-05-03 16:07
'''
import torch
import torch.nn as nn
from datas.utils import max_drawdown

class Batch_Loss(nn.Module):
    def __init__(self, commission_ratio, interest_rate, gamma=0.1, beta=0.1, size_average=True):
        super(Batch_Loss, self).__init__()
        self.gamma = gamma  # variance penalty
        self.beta = beta
        self.size_average = size_average
        self.commission_ratio = commission_ratio
        self.interest_rate = interest_rate

    def forward(self, w, y):  # w:[128,1,12]   y:[128,11,4]
        close_price = y[:, :, 0:1].cuda()  # [128,11,1]
        # future close prise (including cash)
        close_price = torch.cat([torch.ones(close_price.size()[0], 1, 1).cuda(), close_price], 1).cuda()  # [128,11,1]cat[128,1,1]->[128,12,1]
        reward = torch.matmul(w, close_price)  # [128,1,1]
        close_price = close_price.view(close_price.size()[0], close_price.size()[2],
                                       close_price.size()[1])  # [128,1,12]
        ###############################################################################################################
        element_reward = w * close_price
        interest = torch.zeros(element_reward.size(), dtype=torch.float).cuda()
        interest[element_reward < 0] = element_reward[element_reward < 0]
        interest = torch.sum(interest, 2).unsqueeze(2) * self.interest_rate  # [128,1,1]
        ###############################################################################################################
        future_omega = w * close_price / reward  # [128,1,12]
        wt = future_omega[:-1]  # [128,1,12]
        wt1 = w[1:]  # [128,1,12]
        pure_pc = 1 - torch.sum(torch.abs(wt - wt1), -1) * self.commission_ratio  # [128,1]
        pure_pc = pure_pc.cuda()
        pure_pc = torch.cat([torch.ones([1, 1]).cuda(), pure_pc], 0)
        pure_pc = pure_pc.view(pure_pc.size()[0], 1, pure_pc.size()[1])  # [128,1,1]

        cost_penalty = torch.sum(torch.abs(wt - wt1), -1)
        ################## Deduct transaction fee ##################
        reward = reward * pure_pc  # reward=pv_vector
        ################## Deduct loan interest ####################
        reward = reward + interest
        portfolio_value = torch.prod(reward, 0)
        batch_loss = -torch.log(reward)
        #####################variance_penalty##############################
        #        variance_penalty = ((torch.log(reward)-torch.log(reward).mean())**2).mean()
        if self.size_average:
            loss = batch_loss.mean()  # + self.gamma*variance_penalty + self.beta*cost_penalty.mean()
            return loss, portfolio_value[0][0]
        else:
            loss = batch_loss.mean()  # +self.gamma*variance_penalty + self.beta*cost_penalty.mean() #(dim=0)
            return loss, portfolio_value[0][0]


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y): # [1, 2776, 1, 12], [1, 2776, 11, 4]
        loss, portfolio_value = self.criterion(x, y)
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss, portfolio_value


class SimpleLossCompute_tst:
    "A simple loss compute and train function."

    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y):
        if self.opt is not None:
            loss, portfolio_value = self.criterion(x, y)
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
            return loss, portfolio_value
        else:
            loss, portfolio_value, SR, CR, St_v, tst_pc_array, TO = self.criterion(x, y)
            return loss, portfolio_value, SR, CR, St_v, tst_pc_array, TO

class Test_Loss(nn.Module):
    def __init__(self, commission_ratio,interest_rate,gamma=0.1,beta=0.1, size_average=True):
        super(Test_Loss, self).__init__()
        self.gamma = gamma  #variance penalty
        self.beta = beta
        self.size_average = size_average
        self.commission_ratio=commission_ratio
        self.interest_rate=interest_rate

    def forward(self, w, y):               # w:[1,2776,1,12] y(1,2776,11,4)
        close_price = y[:,:,:,0:1].cuda()    #   [128,10,11,1]
        close_price = torch.cat([torch.ones(close_price.size()[0],close_price.size()[1],1,1).cuda(),close_price],2).cuda()       #[128,10,11,1]cat[128,10,1,1]->[128,10,12,1]
        reward = torch.matmul(w,close_price)   #  [128,10,1,12] * [128,10,12,1] ->[128,10,1,1]
        close_price = close_price.view(close_price.size()[0],close_price.size()[1],close_price.size()[3],close_price.size()[2])  #[128,10,12,1] -> [128,10,1,12]
##############################################################################
        element_reward = w*close_price
        interest = torch.zeros(element_reward.size(),dtype = torch.float).cuda()
        interest[element_reward<0] = element_reward[element_reward<0]
#        print("interest:",interest.size(),interest,'\r\n')
        interest = torch.sum(interest,3).unsqueeze(3)*self.interest_rate  #[128,10,1,1]
##############################################################################
        future_omega = w*close_price/reward    #[128,10,1,12]*[128,10,1,12]/[128,10,1,1]
        wt=future_omega[:,:-1]                 #[128, 9,1,12]
        wt1=w[:,1:]                            #[128, 9,1,12]
        pure_pc=1-torch.sum(torch.abs(wt-wt1),-1)*self.commission_ratio     #[128,9,1]
        pure_pc=pure_pc.cuda()
        pure_pc=torch.cat([torch.ones([pure_pc.size()[0],1,1]).cuda(),pure_pc],1)      #[128,1,1] cat  [128,9,1] ->[128,10,1]
        pure_pc=pure_pc.view(pure_pc.size()[0],pure_pc.size()[1],1,pure_pc.size()[2])  #[128,10,1] ->[128,10,1,1]
        cost_penalty = torch.sum(torch.abs(wt-wt1),-1)                                 #[128, 9, 1]
################## Deduct transaction fee ##################
        reward = reward*pure_pc                                                        #[128,10,1,1]*[128,10,1,1]  test: [1,2808-31,1,1]
################## Deduct loan interest ####################
        reward= reward+interest
        if not self.size_average:
            tst_pc_array=reward.squeeze()
            sr_reward=tst_pc_array-1
            SR=sr_reward.mean()/sr_reward.std()
#            print("SR:",SR.size(),"reward.mean():",reward.mean(),"reward.std():",reward.std())
            SN=torch.prod(reward,1) #[1,1,1,1]
            SN=SN.squeeze() #
#            print("SN:",SN.size())
            St_v=[]
            St=1.
            MDD=max_drawdown(tst_pc_array)
            for k in range(reward.size()[1]):  #2808-31
                St*=reward[0,k,0,0]
                St_v.append(St.item())
            CR=SN/MDD
            TO=cost_penalty.mean()
##############################################
        portfolio_value=torch.prod(reward,1)     #[128,1,1]
        batch_loss=-torch.log(portfolio_value)   #[128,1,1]

        if self.size_average:
            loss = batch_loss.mean()
            return loss, portfolio_value.mean()
        else:
            loss = batch_loss.mean()
            return loss, portfolio_value[0][0][0],SR,CR,St_v,tst_pc_array,TO


