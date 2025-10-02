import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    def __init__(self, c, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=False),
            nn.BatchNorm2d(c)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))

class PostProcessor(nn.Module):
    def __init__(self, in_channels=1536, hidden=768, num_blocks=4):
        super().__init__()
        self.blocks = nn.Sequential(*[BottleneckBlock(in_channels, hidden) for _ in range(num_blocks)])

    def forward(self, F):
        return self.blocks(F)
