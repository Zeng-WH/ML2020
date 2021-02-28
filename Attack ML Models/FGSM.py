import torch
import torch.nn.functional as F

def L_Infinity_Loss(pred, actual):
    abs_value = torch.abs(pred-actual)
    max_value = torch.max(abs_value, dim=1)
    return max_value[0]

def FGSM_model(x, label, model, eps=0.1):
    # x: 需要训练的图片
    model.eval()
    x.requires_grad = True
    # 求偏微分
    batch_loss = F.nll_loss(model(x), label)
    batch_loss.backward()
    data_grad = x.grad.data
    out = x + eps * torch.sign(data_grad)
    return out



