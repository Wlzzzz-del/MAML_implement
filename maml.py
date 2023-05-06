import click
import os, sys
import random
import inspect

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim import SGD, Adam
from torch.nn.modules.loss import CrossEntropyLoss

from task import OmniglotTask,MNISTTask
from inner_loop import InnerLoop
from omniglot_net import OmniglotNet
from score import *
from data_loading import *

class MetaLearner(object):
    def __init__(self,
                 dataset,
                 num_classes,
                 num_inst,
                 meta_batch_size,
                 meta_step_size,
                 inner_batch_size,
                 inner_step_size,
                 num_updates,
                 num_inner_updates,
                 loss_fn):
        super(self.__class__, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_inst = num_inst,
        self.meta_batch_size = meta_batch_size
        self.meta_step_size = meta_step_size
        self.inner_batch_size = inner_batch_size
        self.inner_step_size = inner_step_size
        self.num_updates = num_updates
        self.num_inner_updates = num_inner_updates
        self.loss_fn = loss_fn

        #Make the nets
        #TODO: don't actually need two nets
        num_input_channels = 1 if self.dataset == "mnist" else 3
        self.net = OmniglotNet(num_classes, self.loss_fn, num_input_channels)

        # 记得启动CUDA
        # self.net.cuda()

        # fast_net也许是Innerloop的模型
        self.fast_net = InnerLoop(num_classes, self.loss_fn, self.num_inner_updates, self.inner_step_size, self.inner_batch_size, self.meta_batch_size, num_input_channels)

        # 记得启动CUDA
        # self.fast_net.cuda()

        self.opt = Adam(self.net.parameters(), lr=meta_step_size)

    def get_task(self, root, n_cl, n_inst, split="train"):
        if 'mnist' in root:
            return MNISTTask(root, n_cl, n_inst, split)
        elif 'omniglot' in root:
            return OmniglotTask(root, n_cl, n_inst, split)
        else:
            print("unknown dataset.")
            raise(Exception)

    def meta_update(self, task, ls):
        # 这一步是OuterLoop，取InnerLoop任务梯度的平均
        # 参数ls是记录了每个任务计算得到的"梯度字典"

        print("---------META UPDATE NOW------------")
        # 先创建任务集，再将任务集传入get_data_loader中获得data_loader
        loader = get_data_loader(task, self.inner_batch_size, split='val')
        in_, target = loader.__iter__().next()
        # 使用一个虚拟的前向/后向传递来让正确的梯度进入self.net
        loss, out = forward_pass(self.net, in_, target)
        # 解压 "装载有梯度字典" 的列表, 模型每个模块的参数对应的梯度求和
        gradients = {k: sum(d[k] for d in ls) for k in ls[0].keys()}
        # 在当前虚拟网络模型的梯度中的每个参数记录一个hook
        # 在meta-batch中累计梯度
        # ------前向传播产生梯度，用hook方法把梯度替换成聚合后的梯度！-----
        hooks = []
        for (k,v) in self.net.named_parameters():
            def get_closure():
                key = k
                def replace_grad(grad):
                    return gradients[key]
                return replace_grad
            """
            torch在反向传播时默认不保留梯度，如果需要对某一层梯度进行操作，
            可以用register_hook传入某个函数指针对其操作.
            如：
            v = torch.tensor([0,0,0], requires_grad = True)
            h = v.register_hook(lambda grad : grad * 2)# double the grad
            v.back_ward(torch.tensor([1,2,3]))
            """
            hooks.append(v.register_hook(get_closure()))
        # 计算最近一次迭代的梯度，
        self.opt.zero_grad()
        loss.backward()
        # 用聚合的梯度更新网络模型
        self.opt.step()
        # 在下一个训练阶段以前删除所有的hook
        for h in hooks:
            h.remove()

    def test(self):
        num_in_channels = 1 if self.dataset == 'mnist' else 3
        test_net = OmniglotNet(self.num_classes, self.loss_fn, num_in_channels)
        mtr_loss, mtr_acc, mval_loss, mval_acc = 0.0, 0.0, 0.0, 0.0
        # 从测试集中随机挑选10个任务进行测试
        for _ in range(10):
            # 使得受测试的网络有相同的参数
            test_net.copy_weight(self.net)

            # 记得启动CUDA
            # test_net.cuda()

            # 测试所用的优化器
            test_opt = SGD(test_net.parameters(), lr=self.inner_step_size)
            task = self.get_task('../data/{}'.format(self.dataset), self.num_classes, self.num_inst, split="test")
            # 在训练样本上进行训练，在训练中使用相同数量的更新轮数？
            train_loader = get_data_loader(task, self.inner_batch_size, split='train')
            for i in range(self.num_inner_updates):
                in_, target = train_loader.__iter__().next()
                loss, _ = forward_pass(test_net, in_, target)
                test_opt.zero_grad()
                loss.backward()
                test_opt.step()
            # 在训练集样本和测试集样本上测试训练模型
            tloss, tacc = evaluate(test_net, train_loader)
            val_loader = get_data_loader(task, self.inner_batch_size, split='val')
            vloss, vacc = evaluate(test_net, val_loader)
            mtr_loss += tloss
            mtr_acc += tacc
            mval_loss += vloss
            mval_acc += vacc

        # 取10个样本的loss和ACC的平均值
        mtr_loss = mtr_loss/10
        mtr_acc = mtr_acc/10
        mval_loss = mval_loss/10
        mval_acc = mval_acc/10

        print("------------------------------")
        print("Meta train:", mtr_loss, mtr_acc)
        print("Meta val:", mval_loss, mval_acc)
        print("------------------------------")

        return mtr_loss, mtr_acc, mval_loss, mval_acc
"""
-----------MAML TEST HERE------------
"""
dataset='mnist'
num_classes = 3
num_inst = 4
meta_batch_size = 10
meta_step_size = 1
inner_batch_size = 10
inner_step_size = 10
num_updates = 10
num_inner_updates = 10
loss_fn = CrossEntropyLoss()
m = MetaLearner("mnist",
                 num_classes,
                 num_inst,
                 meta_batch_size,
                 meta_step_size,
                 inner_batch_size,
                 inner_step_size,
                 num_updates,
                 num_inner_updates,
                 loss_fn)