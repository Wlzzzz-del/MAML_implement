import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from data_loading import *

from omniglot_net import OmniglotNet
from layers import *
from score import *

class InnerLoop(OmniglotNet):
    """
        这个类展现了MAML的inner loop
        前向方法在数据的梯度步骤更新权重，然后返回元梯度。
    """
    def __init__(self, num_classes, loss_fn, num_updates, step_size, batch_size, meta_batch_size, num_in_channels=3):
        super(InnerLoop, self).__init__(num_classes, loss_fn, num_in_channels)
        # 将要更新的轮数
        self.num_updates = num_updates
        # 更新的步长
        self.step_size = step_size
        # 每一类批次的样本数量
        self.batch_size = batch_size

        # loss normalization
        self.meta_batch_size = meta_batch_size

    def net_forward(self, x, weights=None):
        # 继承父类OmniglotNet的前向传播方法
        return super(InnerLoop, self).forward(x, weights)

    def forward_pass(self, in_, target, weights=None):
        input_var = torch.autograd.Variable(in_).cuda()
        target_var = torch.autograd.Variable(target).cuda()
        # 在网络上计算batch，计算损失
        out = self.net_forward(input_var, weights)
        loss = self.loss_fn(out, target_var)
        return loss, out

    def forward(self, task):
        train_loader = get_data_loader(task, self.batch_size)
        val_loader = get_data_loader(task, self.batch_size, split="val")
        ### 在训练前测试网络，应该是随机准确率 ###
        tr_pre_loss, tr_pre_acc = evaluate(self, train_loader)
        val_pre_loss, val_pre_acc = evaluate(self, val_loader)
        fast_weights = OrderedDict((name, param) for (name, param) in self.named_parameters())
        # 这边取出一个batch的任务做inner loop
        for i in range(self.num_updates):
            print("the "+str(i)+" Inner step")
            in_, target = train_loader.__iter__().next()
            if i==0:
                loss, _ = self.forward_pass(in_, target)
                # create_graph创建导数的计算图，否则无法求导
                grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
            else:
                loss, _ = self.forward_pass(in_, target, fast_weights)
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            # 梯度下降更新weight
            fast_weights = OrderedDict((name, param - self.step_size*grad) for ((name, param), grad) in zip(fast_weights.items(), grads))
        # 在训练后测试模型，应该比之前的随机准确率表现更好

        tr_post_loss, tr_post_acc = evaluate(self, train_loader)
        val_post_loss, val_post_acc = evaluate(self, val_loader)
        print("\n Train Inner step loss:", tr_pre_loss, tr_post_loss)
        print("Train Inner step accuracy", tr_pre_acc, tr_post_acc)
        print("Val Inner step loss", val_pre_loss, val_pre_acc)
        print("val Inner step accuracy", val_post_loss, val_post_acc)

        # 计算元梯度并返回
        # ------------------这边不知道为什么还要做一次梯度？-----------------------
        in_, target = val_loader.__iter__().next()
        loss,_ = self.forward_pass(in_, target, fast_weights)
        loss = loss / self.meta_batch_size # normalize loss
        grads = torch.autograd.grad(loss, self.parameters())
        meta_grads = {name:g for ((name, _), g) in zip(self.named_parameters(), grads)}
        metrics = (tr_post_loss, tr_post_acc, val_post_loss, val_post_acc)
        return metrics, meta_grads
