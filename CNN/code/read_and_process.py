# Import需要的套件

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
def read_file(path, label):
    '''label是一个bool label用来判断是否需要返回y'''
    image_dir = os.listdir(path)
    len_image_dir = len(image_dir)
    x = np.zeros((len_image_dir, 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len_image_dir), dtype=np.uint8)
    for index, image_path in enumerate(image_dir):
        test = '/'.join([path, image_path])
        temp = cv2.imread(test)
        x[index, :, :, :] = cv2.resize(temp, (128, 128))
    if label == True:
        for index, name in enumerate(image_dir):
            name_temp = name.split('_')
            y[index] = int(name_temp[0])
    if label == True:
        return x,y
    else:
        return x



class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X,Y
        else:
            return X


