from typing import Optional, Sized
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from task import OmniglotTask,MNISTTask
from dataset import Omniglot, MNIST

"""
从数据集中加载样本平衡的少样任务集
"""

class ClassBalancedSampler(Sampler):
    """
    所有Sampler的基类，
    每个Sampler的基类都必须提供__iter__方法返回迭代器
    提供__len__方法返回长度
    """
    def __init__(self, num_cl, num_inst, batch_cutoff = None):
        self.num_cl = num_cl  # 类别数
        self.num_inst = num_inst # 每个类别包含的样本数量
        self.batch_cutoff = batch_cutoff # 只要某个指定的batch的数量

    def __iter__(self):
        """
        返回一个索引的列表，其中每一项都将被类分组
        """
        # 首先构造batches,每类有一个实例
        # torch.randperm()返回一个随机列表序列由0,n-1组成
        # 实现每个batche中每个类别的数量一致
        batches = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cl)]
        batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]
        # 打乱顺序
        for sublist in batches:
            random.shuffle(sublist)
        # batchcutoff不理解
        if self.batch_cutoff is not None:
            random.shuffle(batches)
            batches = batches[:self.batch_cutoff]
        batches = [item for sublist in batches for item in sublist]

        return iter(batches)

    def __len__(self):
        return 1

def get_data_loader(task, batch_size=1, split = 'train'):
    # NOTE: 这边的batchsize是每个类别的例子数
    if task.dataset == "mnist":
        normalize = transforms.Normalize(mean=[0.13066,0.13066,0.13066], std=[0.301031, 0.301031, 0.301031])
        dset = MNIST(task, transform = transforms.Compose([transforms.ToTensor(), normalize]), split = split)
        pass
    else:
        normalize = transforms.Normalize(mean=[0.92206,0.92206,0.92206], std=[0.08426,0.08426,0.08426])
        dset = Omniglot(task, transform = transforms.Compose([transforms.ToTensor(), normalize]), split = split)
        pass
    Sampler = ClassBalancedSampler(task.num_cls, task.num_inst, batch_cutoff=(None if split != 'train' else batch_size))
    loader = DataLoader(dset, batch_size=batch_size*task.num_cls, sampler=Sampler, num_workers=1, pin_memory=True)
    return loader

mni = MNISTTask("mnist",3,4)
Loader = get_data_loader(mni)