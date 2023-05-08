import sys
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

class FewShotDataset(data.Dataset):
    """
    加载图像-标签数据集，将task传到Torch DataLoader中
    task由data和label组成
    """
    def __init__(self, task, split="train", transform=None, target_transform=None,) -> None:
        self.transform = transform
        self.target_transform = target_transform
        self.task = task
        self.root = self.task.root# 这个root是什么意思,路径吗？
        self.split = split

        # 根据在什么集上切分来读取训练和测试集
        self.img_ids = self.task.train_ids if self.split == 'train' else self.task.val_ids
        self.labels = self.task.train_labels if self.split == 'train' else self.task.val_labels
    def __len__(self):
        return len(self.img_ids)
    def __getitem__(self, index):
        raise NotImplementedError("This is a abstract class.")
    pass

class Omniglot(FewShotDataset):
    """
    继承自FewShotDataset，
    将Omniglot数据集改造成task
    """
    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def load_image(self, idx):
        """加载图像"""
        im = Image.open('{}/{}'.format(self.root, idx)).convert("RGB")
        # lanczos重采样
        im = im.resize((28,28),resample=Image.LANCZOS)
        im = np.array(im, dtype = np.float32)
        return im

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        im = self.load_image(img_id)
        # 如果对数据做变换的参数不为空，则做变换
        if self.transform is not None:
            im = self.transform(im)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return im,label

class MNIST(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(MNIST, self).__init__(*args, **kwargs)
        pass
    def load_image(self, idx):
        # NOTE:在黑白图数据上使用RGB是因为Torch使用黑白图会导致错误
        im = Image.open("{}/{}.jpg".format(self.root, idx)).convert("RGB")
        im = np.array(im)
        return im
    def __getitem__(self, idx):
        img_idx = self.img_ids[idx]
        im = self.load_image(img_idx)
        if self.transform is not None:
            self.transform(im)
        target = self.labels[idx]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return im, target