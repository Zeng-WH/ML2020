import os
import read_data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import numpy as np
import time
import pandas as pd
import torchvision.models as models
import torch.nn.functional as F
import FGSM


def L_Infinity_Loss(pred, actual):
    abs_value = torch.abs(pred-actual)
    max_value = torch.max(abs_value, dim=1)
    return max_value[0]


# 读取data
print('-----------------read data-------------------')


transform = transforms.Compose(
    [
        #transforms.ToPILImage(),
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),  # 将图片转成Tensor, 并把数值normalize到[0, 1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            inplace=False),

    ]

)

df = pd.read_csv('./data/labels.csv')
df = df.loc[:, 'TrueLabel']
label = []
for i in df:
    label.append(i)
label = np.array(label)
ad_set = read_data.Adverdataset('./data/images', label, transform)

batch_size = 1


ad_loader = DataLoader(ad_set, batch_size=batch_size, shuffle=False)
vgg16_model = models.vgg16(pretrained=True).cuda()
#num_epoch = 25
print('------------------test model-------------------')

'''
for epoch in range(num_epoch):
    vgg16_model.eval()
    with torch.no_grad():
        for i, data in enumerate(ad_loader):
            test_pred = vgg16_model(data[0].cuda())
            final_pred = test_pred.max(1, keepdim=True)[1]
            batch_loss = F.nll_loss(test_pred, data[1].cuda())
            print('test')
'''
wrong, fail, success = 0, 0, 0
vgg16_model.eval()
adv_examples = []
for i, data in enumerate(ad_loader):
    final_pred = vgg16_model(data[0].cuda()).max(1, keepdim=True)[1]
    if final_pred.item() != data[1].item():
        wrong = wrong + 1
    else:
        FGSM_out = FGSM.FGSM_model(data[0].cuda(), data[1].cuda(), vgg16_model, 0.1)
        attack_pred = vgg16_model(FGSM_out).max(1, keepdim=True)[1]
        if final_pred.item() != attack_pred.item():
            success = success+1
            adv_ex = FGSM_out * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda() + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
            adv_ex = adv_ex.squeeze().detach().cpu().numpy()
            data_raw = data[0].cuda() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda() + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
            data_raw = data_raw.squeeze().detach().cpu().numpy()
            adv_examples.append([final_pred.item(), attack_pred.item(), data_raw, adv_ex])
        else:
            fail = fail + 1
print('success rate:')
print(success/(success+wrong+fail))
np.save('adv_examples.npy', adv_examples)










