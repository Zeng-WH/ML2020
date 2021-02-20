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
import filter_visualiton


# 读取文件
print('Reading Data')
print('Reading Training Data')
train_x, train_y = read_and_process.read_file('./food-11/training', True)

print("Size of training data = {}".format(len(train_x)))

saliency_transformers = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]
)

batch_size = 32
train_set = read_and_process.ImgDataset(train_x, train_y, saliency_transformers)

with open('./cnn_model.pickle', 'rb') as f1:
    model = pickle.load(f1)
    model = model.cuda()

print('-------------------filter visualization--------------------')
img_indices = [83, 4218, 4707, 8598]
images, labels = train_set.getbatch(img_indices)
activations, visualization = filter_visualiton.filter_explanation(images, model, cnnid=15, filterid=0, iteration=100, lr=0.1)
# 画出filter visualization
plt.imshow(filter_visualiton.normalize(visualization.permute(1, 2, 0)))
plt.show()
plt.close()

fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for i, img in enumerate(images):
  axs[0][i].imshow(img.permute(1, 2, 0))
for i, img in enumerate(activations):
  axs[1][i].imshow(filter_visualiton.normalize(img))
plt.show()
plt.close()
