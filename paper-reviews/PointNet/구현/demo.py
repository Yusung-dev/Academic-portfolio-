import torch
from models.pointnet_cls import PointNetCls

torch.manual_seed(0)

B, N, K = 2, 1024, 40
x = torch.randn(B, 2, N)

model = PointNetCls(t=K).eval()

with torch.no_grad():
    logits = model(x)
    pred = logits.argmax(dim=1)

print("[PointNetCls] logits shape:", logits.shape)
print("[PointNetCls] pred:", pred.tolist())
