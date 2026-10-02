"""Point + four distances: a simplified anchor-free regression experiment."""
import torch
from torch import nn
from torch.nn import functional as F


def decode(points, distances, stride):
    left_top, right_bottom = distances[..., :2], distances[..., 2:]
    return torch.cat((points - left_top * stride, points + right_bottom * stride), -1)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    point = torch.tensor([[24., 24.]])  # pixel x,y; stride is pixels per feature cell
    gt = torch.tensor([[12., 16., 40., 36.]])
    distance = torch.cat((point - gt[:, :2], gt[:, 2:] - point), -1) / 8
    assert torch.allclose(distance, torch.tensor([[1.5, 1., 2., 1.5]]))
    assert torch.allclose(decode(point, distance, 8), gt)
    # This head predicts positive continuous distances, not anchor width/height offsets.
    raw = nn.Parameter(torch.zeros(1, 4))
    optimizer = torch.optim.SGD([raw], lr=.5)
    before = F.smooth_l1_loss(F.softplus(raw), distance).item()
    for _ in range(80):
        optimizer.zero_grad()
        loss = F.smooth_l1_loss(F.softplus(raw), distance)
        loss.backward()
        assert torch.isfinite(raw.grad).all()
        optimizer.step()
    after = F.smooth_l1_loss(F.softplus(raw), distance).item()
    assert after < before / 100
    print('target ltrb in feature cells:', distance.tolist())
    print('decoded target pixels:', decode(point, distance, 8).tolist())
    print(f'distance loss {before:.6f} -> {after:.6f}')
    print('learned box pixels:', decode(point, F.softplus(raw).detach(), 8).round(decimals=2).tolist())
    # A point outside the target cannot represent that box with all-positive distances.
    outside = torch.tensor([[48., 24.]])
    assert (torch.cat((outside - gt[:, :2], gt[:, 2:] - outside), -1) < 0).any()
    print('outside point requires a negative distance: assignment is still necessary')


if __name__ == '__main__':
    main()
