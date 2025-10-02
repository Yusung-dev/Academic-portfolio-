import torch
import torch.nn.functional as F
from .imd import imd, l2_normalize

def simple_augments(imgs, modes=("none", "hflip", "rot90", "rot270")):
    outs = []
    for x in imgs:
        cand = []
        for m in modes:
            if m == "none":
                cand.append(x)
            elif m == "hflip":
                cand.append(torch.flip(x, dims=[-1]))
            elif m == "rot90":
                cand.append(torch.rot90(x, k=1, dims=(-2, -1)))
            elif m == "rot270":
                cand.append(torch.rot90(x, k=3, dims=(-2, -1)))
        outs.append(cand)
    return outs

def pick_best_pose(Fs, Ft, Ms, pose_feats):
    scores = []
    for Fa in pose_feats:
        scores.append(imd(Fa, Ft, Ms))
    scores = torch.stack(scores, dim=1)
    idx = scores.argmin(dim=1)
    return idx
