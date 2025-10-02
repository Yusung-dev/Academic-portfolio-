import torch
from models.postprocessor import PostProcessor
from inference.window_softargmax import window_softargmax
from inference.imd import imd, l2_normalize
from utils.norm import l2n

torch.manual_seed(0)

B, C, H, W = 2, 1536, 60, 60
Fs_raw = torch.randn(B, C, H, W)
Ft_raw = torch.randn(B, C, H, W)

Fs = l2n(Fs_raw)
Ft = l2n(Ft_raw)

Ms = torch.ones(B, 1, H, W)

post = PostProcessor(in_channels=C, hidden=768, num_blocks=4).eval()
with torch.no_grad():
    Fs_t = post(Fs)
    Ft_t = post(Ft)

pose_variants = [Fs_t, torch.flip(Fs_t, dims=[-1]), torch.rot90(Fs_t, 1, dims=(-2, -1))]
scores = [imd(p, Ft_t, Ms) for p in pose_variants]  # list of (B,)
scores = torch.stack(scores, dim=1)
best = scores.argmin(dim=1)
print("[Pose Align] best indices:", best.tolist())

Fs_sel = torch.stack([pose_variants[best[b]][b] for b in range(B)], dim=0)

Q = 3
ys = torch.randint(0, H, (B, Q))
xs = torch.randint(0, W, (B, Q))
coords_pred = []
for b in range(B):
    for q in range(Q):
        fv = Fs_sel[b, :, ys[b, q], xs[b, q]]
        sim = torch.einsum('c,chw->hw', fv, Ft_t[b])
        sim = sim.unsqueeze(0)
        xy = window_softargmax(sim, k=15)[0]
        coords_pred.append(xy)
coords_pred = torch.stack(coords_pred, 0)
print("[Window Soft-Argmax] sample coords:", coords_pred[:5])
