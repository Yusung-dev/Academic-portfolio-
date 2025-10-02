import torch

def l2n(x, eps=1e-6):
    return x / (x.norm(dim=1, keepdim=True) + eps)
