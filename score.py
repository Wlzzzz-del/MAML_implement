import numpy as np
import torch
from torch.autograd import Variable

"""
    测试网络模型的分类性能
"""

def count_correct(pred, target):
    """
        计算正确分类的样本数
    """
    pairs = [int(x==y) for x,y in zip(pred, target)]
    return sum(pairs)

def forward_pass(net, in_, target, weights=None):
    """
        把样本送入模型中，返回loss和输出
    """
    # 获得data和label
    """
        NOTE:Variable是autograd的核心类，浅封装了Tensor，用于整合实现反向传播。
    """
    input_var = Variable(in_).cuda()
    target_var = Variable(target).cuda()
    out = net.net_forward(input_var, weights)
    loss = net.loss_fn(out, target_var)

    return loss,out

def evaluate(net, loader, weights=None):
    num_correct = 0
    loss = 0
    for i, (in_, target) in enumerate(loader):
        batch_size = in_.numpy().shape[0]
        l, out = forward_pass(net, in_, target, weights)
        loss += l.data.cpu().numpy()
        num_correct += count_correct(np.argmax(out.data.cpu().numpy(), axis=1), target.numpy())
    return float(loss)/ len(loader), float(num_correct)/(len(loader)*batch_size)
    pass