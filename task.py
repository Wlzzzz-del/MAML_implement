import os
import random
import numpy as np
import torch

class OmniglotTask(object):
    """
    从Omniglot数据集中采用少样学习任务
    采样N-way k-shot的训练集和测试集，根据以下规则：
    - split()
    - N-way 小样本训练中的类别数
    - k-shot 每个类有几个样本
    假定训练集和测试集有相同的size
    """
    def __init__(self, root, num_cls, num_inst, split="train"):
        self.dataset = "omniglot"
        # 训练集测试集路径
        self.root = "{}/images_background".format(root) if split=="train" else "{}/images_evaluation".format(root)
        self.num_cls = num_cls
        self.num_inst = num_inst
        languages = os.listdir(self.root)
        chars = []
        for l in languages:
            # 获得每个类别文件夹内的文件路径
            chars += [os.path.join(l, x) for x in os.listdir(os.path.join(self.root, l))]
        random.shuffle(chars)
        classes = chars[:num_cls]# 打乱后取前num_cls类
        labels = np.array(range(len(classes)))# 给前num_cls类按顺序打上label
        labels = dict(zip(classes, labels))# 此处已经构建好了一个classes-labels的字典
        instances = dict()
        # 现在从已选择的类中创建类别均衡的训练集和测试集
        self.train_ids = []
        self.val_ids = []
        for c in classes:
            # 首先获取类的所有样本
            temp = [os.path.join(c, x)for x in os.listdir(os.path.join(self.root, c))]
            instances[c] = random.sample(temp, len(temp))
            # 随机采样num_inst个实例到训练集和测试集中
            self.train_ids += instances[c][:num_inst]
            self.val_ids += instances[c][num_inst:2*num_inst]
        # 保持label和id一致
        self.train_labels = [labels[self.get_class(x)] for x in self.train_ids]
        self.val_labels = [labels[self.get_class(x)] for x in self.val_ids]
    def get_class(self, instance):
        return os.path.join(*instance.split('/')[:-1])

    def view_label(self):
        print("训练集:",self.train_ids)
        print("训练集的label：",self.train_labels)
        print("测试集:",self.val_ids)
        print("测试集的label:",self.val_labels)

class MNISTTask(object):
    """
    从Mnist数据集中采用少样学习任务
    采样N-way k-shot的训练集和测试集，根据以下规则：
    - split()
    - N-way 小样本训练中的类别数
    - k-shot 每个类有几个样本
    假定训练集和测试集有相同的size
    """
    def __init__(self, root, num_cls, num_inst, split='train'):
        self.num_cls = num_cls
        self.num_inst = num_inst
        self.dataset = 'mnist'
        self.split = split
        self.root = root + '/' + split
        all_ids = []
        for i in range(10):
            d = os.path.join(root, self.split, str(i))
            files = os.listdir(d)
            all_ids.append([str(i) + '/' + f[:-4] for f in files])
        # 为了创建一个任务，需要先随机打乱label
        # permutation打乱序列
        self.label_map = dict(zip(range(10), np.random.permutation(np.array(range(10)))))

        # 选择num_inst个ids 从每10类中
        self.train_ids = []
        self.val_ids = []
        for i in range(10):
            permutation = list(np.random.permutation(np.array(range(len(all_ids)))))[:num_inst*2]
            self.train_ids += [all_ids[i][j] for j in permutation[:num_inst]]
            self.train_labels = self.relabel(self.train_ids)
            self.val_ids += [all_ids[i][j] for j in permutation[num_inst:]]
            self.val_labels = self.relabel(self.val_ids)

    def relabel(self, img_ids):
        """
        重新分配新的label
        """
        print(img_ids)
        orig_labels = [int(x[0]) for x in img_ids]
        return [self.label_map[x] for x in orig_labels]

    def view_label(self):
        print("训练集:",self.train_ids)
        print("训练集的label：",self.train_labels)
        print("测试集:",self.val_ids)
        print("测试集的label:",self.val_labels)







# testhere
# print(os.getcwd())
# OMN = OmniglotTask(root='./omniglot',num_cls=3, num_inst=5)
# OMN.view_label()
# MNI = MNISTTask(root = './mnist',num_cls = 3 ,num_inst=4)
# MNI.view_label()
