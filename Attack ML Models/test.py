import numpy as np
import torch

def L_Infinity_Loss(pred, actual):
    abs_value = torch.abs(pred-actual)
    max_value = torch.max(abs_value, dim=1)
    return max_value[0]

a = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
b = torch.tensor([[6.,6.,6.],[7.,10.,9.]])
test = L_Infinity_Loss(a,b)

print('a')


