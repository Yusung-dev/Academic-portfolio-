import torch
import torch.nn.functional as F

def l2_normalize(feat, eps=1e-6):
    return feat / (feat.norm(dim=1, keepdim=True) + eps)

def imd(Fs, Ft, Ms):
    B, C, H, W = Fs.shape
    Ms_flat = Ms.view(B, -1)
    Fs_flat = (Fs * Ms).view(B, C, -1).transpose(1, 2)
    Ft_flat = Ft.view(B, C, -1)

    sim = torch.bmm(Fs_flat, Ft_flat)
    nn_idx = sim.argmax(dim=-1)
    # gather NN feature
    Ft_nn = torch.gather(Ft_flat, 2, nn_idx.unsqueeze(1).expand(-1, C, -1))
    Ft_nn = Ft_nn.transpose(1, 2)

    dist = ((Fs_flat - Ft_nn)**2).sum(dim=-1)
    eps = 1e-6
    return (dist * Ms_flat).sum(dim=1) / (Ms_flat.sum(dim=1) + eps)
