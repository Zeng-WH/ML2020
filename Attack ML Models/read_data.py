import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

def read_file(path):
    '''读取照片'''
    image_dir = os.listdir(path)
    len_image_dir = len(image_dir)
    x = np.zeros((len_image_dir, 224, 224, 3), dtype=np.uint8)
    y = np.zeros((len_image_dir), dtype=np.uint8)
    for index, image_path in enumerate(image_dir):
        test = '/'.join([path, image_path])
        temp = cv2.imread(test)
        x[index, :, :, :] = cv2.resize(temp, (224, 224))
    return x


class ImgDataset(Dataset):
    def __init__(self, x, transforms=None):
        self.x = x
        self.transform = transforms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        X = self.x[item]
        if self.transform is not None:
            X = self.transform(X)
        return X


class Adverdataset(Dataset):
    def __init__(self, path, label, transforms):
        # 图片的文件夹
        self.path = path
        self.transformers = transforms
        self.label = torch.from_numpy(label).long()
        self.fnames = []
        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, item):
        # 利用路径读取图片
        img = Image.open(os.path.join(self.path, self.fnames[item] + '.png'))
        # 将输入的图片转换成符合预训练模型的形式
        img = self.transformers(img)
        label = self.label[item]
        return img, label

    def __len__(self):
        return 200

