## 📄 논문 구현: Telling Left from Right: Identifying Geometry-Aware Semantic Correspondence

> 논문 링크: https://arxiv.org/abs/2311.17034 
> 
> 발표 학회/연도: CVPR 2024
> 
> 논문 저자: Junyi Zhang, Charles Herrmann, Junhwa Hur, Eric Chen, Varun Jampani, Deqing Sun, Ming-Hsuan Yang
<br>

---

### Overview

이 리포는 GeoAware-SC 아이디어(geometry-aware semantic correspondence)를 최소 코드로 재구성한 구조체입니다.
실제 데이터셋 없이 torch.randn()으로 생성한 가짜 피처를 사용해 파이프라인의 흐름(포즈 정렬 → 윈도우 soft-argmax)을 확인하도록 설계했습니다

처음 구현 관점에서, 구조를 다음처럼 나눴습니다:
- `models/` : 논문에서 제안한 경량 후처리 모듈(4-layer bottleneck) 구현
- `inference/` : IMD, Adaptive Pose Alignment(테스트 시 포즈 정렬), Window Soft-Argmax
- `utils/` : 공용 유틸 (정규화 등)
- `demo.py` : 위 블록들을 엮어 훈련 없이 동작 검증

Dataset과 config는 포함하지 않았습니다. 본 스켈레톤은 실제 입력 대신 랜덤 텐서를 사용해 구조만 빠르게 검증하기 위함입니다

---
### File Tree

```pthon
구현/
├─ models/
│  └─ postprocessor.py          # 4-layer bottleneck (≈5M params 느낌의 경량 구조)
├─ inference/
│  ├─ window_softargmax.py      # argmax → local window에서 soft-argmax
│  ├─ imd.py                    # Instance Matching Distance
│  └─ pose_align.py             # Test-time adaptive pose alignment (개념 구현)
├─ utils/
│  └─ norm.py                   # L2 normalize 등
├─ demo.py                      # 전체 파이프라인 데모(학습 없음, 랜덤 텐서)
├─ requirements.txt             # torch만 있으면 됨
└─ README.md
```

---
### Components

1. `models/postprocessor.py`
   - **목적**: SD+DINO에서 온 융합 피처를 geometry-aware하게 정제
   - **구조**: 채널/해상도 유지형 4-layer bottleneck block (skip connection 포함)
   - **학습**: 이 스켈레톤에서는 학습하지 않고 구조만 시연
     
```python
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
```

<br><br>

2. `inference/imd.py`
   - IMD(Instance Matching Distance): 소스 피처 vs 타깃 피처 간 최근접 특성의 평균 L2 거리
   - 포즈 정렬 단계에서 최적 pose 후보 선택에 사용
  
```python
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

```

<br><br>

3. `inference/pose_align.py`
   - Test-time Adaptive Pose Alignment
   - 소스 이미지 여러 pose로 변환했다고 가정
   - 각 후보와 타겟사이의 IMD를 계산 --> 최소 IMD 후보를 선택

```python
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

```

<br><br>

4. `inference/window_softargmax.py`
   - 기본 argmax로 coarse 위치를 잡은 뒤, local window 범위에서 soft-argmax로 서브픽셀 정밀도 확보
   - 잡음 민감도를 낮추면서 정밀한 좌표 예측
  
```python
import torch
import torch.nn.functional as F

def softargmax2d(sim):
    B, H, W = sim.shape
    sim = sim.view(B, -1)
    sim = sim - sim.max(dim=1, keepdim=True).values
    prob = sim.softmax(dim=1)

    xs = torch.linspace(0, W-1, W, device=sim.device)
    ys = torch.linspace(0, H-1, H, device=sim.device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0)

    coords = prob @ grid.T
    return coords

def window_softargmax(sim, k=15):
    B, H, W = sim.shape
    idx = sim.view(B, -1).argmax(dim=1)
    cy, cx = (idx // W), (idx % W)
    r = k // 2

    outs = []
    for b in range(B):
        y0, y1 = int(max(0, cy[b]-r)), int(min(H, cy[b]+r+1))
        x0, x1 = int(max(0, cx[b]-r)), int(min(W, cx[b]+r+1))
        patch = sim[b, y0:y1, x0:x1].unsqueeze(0)
        off = softargmax2d(patch).squeeze(0)
        outs.append(torch.stack([x0 + off[0], y0 + off[1]]))
    return torch.stack(outs, dim=0)

```

<br><br>

5. `utils/norm.py`
   -피처 L2 정규화

```python
import torch

def l2n(x, eps=1e-6):
    return x / (x.norm(dim=1, keepdim=True) + eps)
```

---
### How to Run

```python
pip install -r requirements.txt
python demo.py
```
- `demo.py`는 다음 순서로 동작합니다
1. (가짜) DINO + SD 피처 텐서 생성 (torch.randn)
2. Post-processor 통과 --> 정제 피처
3. 여러 pose 후보 생성 --> IMD로 최적 pose 선택
4. 선택된 소스 피처 기준으로 타깃과 유사도 맵 계싼
5. Window Soft-Argmax로 최종 대응 좌표 추정

---

