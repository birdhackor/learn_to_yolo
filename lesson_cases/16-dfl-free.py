"""Continuous distance regression removes bin softmax and its finite-bin range."""
import torch
from torch import nn
from torch.nn import functional as F


def decode(points, distances, stride):
    return torch.cat((points - distances[..., :2] * stride,
                      points + distances[..., 2:] * stride), -1)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    target = torch.tensor([[1.25, 2.5, 18., 3.]])  # feature-cell units
    distance = nn.Parameter(torch.zeros(1, 4))
    optimizer = torch.optim.SGD([distance], lr=.5)
    initial_terms = F.smooth_l1_loss(distance, target, reduction='none')
    assert torch.allclose(initial_terms, torch.tensor([[.75, 2., 17.5, 2.5]]))
    before = initial_terms.mean().item()
    for step in range(300):
        optimizer.zero_grad()
        loss = F.smooth_l1_loss(distance, target)
        loss.backward()
        if step == 0:
            assert torch.allclose(distance.grad, torch.full_like(distance, -.25))
            print('first Smooth L1 gradient:', distance.grad.tolist())
        optimizer.step()
        if step == 0:
            assert torch.allclose(distance, torch.full_like(distance, .125))
            print('first SGD distance:', distance.detach().tolist())
    after = F.smooth_l1_loss(distance, target).item()
    assert after < 1e-6
    k = 16
    logits = torch.zeros(1, 4, k)
    probabilities = logits.softmax(-1)
    expected = (probabilities * torch.arange(k)).sum(-1)
    assert torch.allclose(probabilities, torch.full_like(probabilities, 1 / 16))
    assert torch.allclose(expected, torch.full_like(expected, 7.5))
    assert expected.max() <= k - 1 and target.max() > k - 1
    dfl_target = torch.zeros(k)
    dfl_target[1], dfl_target[2] = .75, .25
    assert torch.allclose((dfl_target * torch.arange(k)).sum(), target[0, 0])
    # Candidate point: center of feature cell column 10, row 10, i.e. ((10+.5)*8, (10+.5)*8).
    points = torch.tensor([[84., 84.]])  # pixel x,y
    decoded = decode(points, distance.detach(), stride=8)
    assert torch.allclose(decoded, torch.tensor([[74., 64., 228., 108.]]), atol=1e-3)
    signed_box = decode(points, torch.tensor([[-1., 2., 3., 4.]]), stride=8)
    assert torch.allclose(signed_box, torch.tensor([[92., 68., 108., 116.]]))
    assert (signed_box[..., 2:] > signed_box[..., :2]).all()
    b, p, classes = 1, 100, 2
    raw_dfl = torch.zeros(b, p, 4 * k + classes)
    raw_direct = torch.zeros(b, p, 4 + classes)
    print('direct learned distances:', distance.detach().round(decimals=3).tolist())
    print(f'Smooth L1 {before:.6f} -> {after:.6f}')
    print('decoded direct box pixels:', decoded.round(decimals=3).tolist())
    print('signed-distance example box pixels:', signed_box.tolist())
    print('untrained DFL probability per bin:', probabilities[0, 0].tolist())
    print('untrained DFL expected distances:', expected.tolist())
    print('DFL target for 1.25: bins 1/2 weights .75/.25; direct target is the value 1.25')
    print('finite-bin expectation range:', [0, k - 1], '; direct target reaches', target.max().item())
    print('raw head values DFL / direct:', raw_dfl.numel(), raw_direct.numel())
    print('float32 raw head bytes:', raw_dfl.numel() * 4, raw_direct.numel() * 4)
    print('DFL-free still requires box supervision; this experiment uses Smooth L1')


if __name__ == '__main__':
    main()
