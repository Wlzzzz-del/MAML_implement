import numpy as np
import random
import math
from collections import OrderedDict

import torch
import torch.nn as nn

from layers import *

class OmniglotNet(nn.Module):
    """
        Omniglot少样本学习上的基础模型
    """
    def __init__(self, num_classes, loss_fn, num_in_channels=3):
        super(OmniglotNet, self).__init__()
        # 定义网络结构
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(num_in_channels, 64, 3)),
                ('bn1', nn.BatchNorm2d(64, momentum=1, affine=True)),
                ('relu1',nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(2,2)),
                ('conv2', nn.Conv2d(64,64,3)),
                ('bn2', nn.BatchNorm2d(64, momentum=1, affine=True)),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(2,2)),
                ('conv3', nn.Conv2d(64,64,3)),
                ('bn3', nn.BatchNorm2d(64, momentum=1, affine=True)),
                ('relu3', nn.ReLU(inplace=True)),
                ('pool3', nn.MaxPool2d(2,2))
            ]))
        self.add_module('fc', nn.Linear(64, num_classes))

        # 定义损失函数
        self.loss_fn = loss_fn
        # 初始化权重
        self._init_weights()

    def forward(self, x, weights=None):
        """
        定义了数据进入网络模型时各层的计算顺序
        """
        # 如果没有权重
        if weights == None:
            x = self.features(x)
            x = x.view(x.size(0), 64)
            x = self.fc(x)
        else:
            x = conv2d(x, weights['features.conv1.weight'], weights['features.conv1.bias'])
            x = batchnorm(x, weight = weights['features.bn1.weight'], bias= weights['features.bn1.bias'],momentum=1)
            x = relu(x)
            x = maxpool(x, kernel_size=2, stride=2)
            x = conv2d(x, weights['features.conv2.weight'],weights['features.conv2.bias'])
            x = batchnorm(x, weight = weights['features.bn2.weight'], bias = weights['features.bn2.bias'],momentum=1)
            x = relu(x)
            x = maxpool(x, kernel_size=2, stride=2)
            x = conv2d(x, weights['features.conv3.weight'],weights['features.conv3.bias'])
            x = batchnorm(x, weight = weights['features.bn3.weight'], bias = weights['features.bn3.bias'],momentum=1)
            x = relu(x)
            x = maxpool(x, kernel_size=2, stride=2)
            x = x.view(x.size(0),64)
            x = linear(x, weights['fc.weight'], weights['fc.bias'])
        return x

    def net_forward(self, x, weights=None):
        # 转调用前向传播函数
        return self.forward(x, weights)

    def _init_weights(self):
        """
            初始化网络结构权重和bias，
            权重用服从高斯分布的随机数，
            偏置用0
        """
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                # 权重置为服从正态分布的随机数
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    # 偏置置为0
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data = torch.ones(m.bias.data.size())

    def copy_weight(self, net):
        """
            将模型的权重设为和参数net一样
            # TODO: 如果net和model的结构不一样则break
        """
        for m_from, m_to in zip(net.modules() ,self.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.BatchNorm2d) or isinstance(m_to, nn.Conv2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()