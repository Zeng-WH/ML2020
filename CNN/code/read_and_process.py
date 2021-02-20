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
def normalize(image):
    '''不同图片之间的gradient scale可能存在巨大落差，需要对每一张saliency各自做normalize'''
    return (image - image.min()) / (image.max()-image.min())

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
    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
            image, label = self.__getitem__(index)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)


class Explain_ImgDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        # label is required to be a LongTensor
        self.y =y
        if y is not None:
            self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        X = self.x[item]
        if self.y is not None:
            Y = self.y[item]
            return X, Y
        else:
            return X


