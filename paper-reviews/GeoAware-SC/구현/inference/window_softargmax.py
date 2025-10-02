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
