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
def normalize(image):
    '''不同图片之间的gradient scale可能存在巨大落差，需要对每一张saliency各自做normalize'''
    return (image - image.min()) / (image.max()-image.min())

def compute_saliency_map(x, y, model):
    '''要计算loss对input image的微分， 原本input x
    只是一个tensor, 不需要gradient, 需要使用requires_grad'''
    model.eval()
    #x = torch.tensor(x, dtype=torch.float)
    x = x.cuda()
    x.requires_grad_()
    y_pred = model(x)
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(y_pred, y.cuda())
    loss.backward()

    saliencies = x.grad.abs().detach().cpu()
    saliencies = torch.stack([normalize(item) for item in saliencies])
    # （batches, channel, height, weight）
    return saliencies




