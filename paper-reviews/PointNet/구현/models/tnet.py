import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn2   = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x: torch.Tensor):

        B, k, N = x.size()
        assert k == self.k

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        out = torch.max(out, dim=2, keepdim=False)[0]

        out = F.relu(self.bn4(self.fc1(out)))
        out = F.relu(self.bn5(self.fc2(out)))
        T   = self.fc3(out).view(B, self.k, self.k)

        x_t = torch.bmm(x.transpose(1, 2), T).transpose(1, 2)
        return x_t, T

def tnet_orthogonal_regularizer(T: torch.Tensor) -> torch.Tensor:
    B, k, _ = T.size()
    I = torch.eye(k, device=T.device).unsqueeze(0).expand(B, -1, -1)
    TT_t = torch.bmm(T, T.transpose(1, 2))
    diff = I - TT_t
    return torch.mean(torch.norm(diff, dim=(1, 2))**2)
