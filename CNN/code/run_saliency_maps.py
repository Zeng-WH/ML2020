import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle
import read_and_process
import saliency_maps
from torch.utils.data import DataLoader, Dataset

# 读取文件
print('Reading Data')
print('Reading Training Data')
train_x, train_y = read_and_process.read_file('./food-11/training', True)

print("Size of training data = {}".format(len(train_x)))

print('Read Validation Data')

val_x, val_y = read_and_process.read_file('./food-11/validation', True)

print("Size of validation data = {}".format(len(val_x)))
saliency_transformers = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]
)

batch_size = 32
train_set = read_and_process.ImgDataset(train_x, train_y, saliency_transformers)
val_set = read_and_process.ImgDataset(val_x, val_y, saliency_transformers)




with open('./cnn_model.pickle', 'rb') as f1:
    model = pickle.load(f1)
    model = model.cuda()


print('----------------------saliency maps--------------------')

img_indeices = [18, 72, 146, 173]
images, labels = train_set.getbatch(img_indeices)
maps = saliency_maps.compute_saliency_map(images, labels, model)
print('bupt')

fig, axs = plt.subplots(2, len(images), figsize=(15, 8))

for row, target in enumerate([images, maps]):
    for column, img in enumerate(target):
        axs[row][column].imshow(img.permute(1, 2, 0).numpy())

plt.show()
plt.close()







