import torch
import torch.nn as nn
from models.tnet import TNet

class PointNetCls(nn.Module):
    def __init__(self, t = 40):
        super().__init__()
        self.tnet3 = TNet(k=3)
        self.tnet64 = TNet(k=64)

        self.net1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()    
        )

        self.net2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.net3 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, t)
        )


    def forward(self, x:torch.Tensor):
        x, T3 = self.tnet3(x)
        x = self.net1(x)
        x, T64 = self.tnet64(x)
        x = self.net2(x)
        x = torch.max(x, dim=2, keepdim = False)[0]
        x = self.net3(x)
        return x
