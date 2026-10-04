"""DFL computes two-bin cross entropy and a differentiable expected distance."""
import torch
from torch import nn
from torch.nn import functional as F


def dfl(logits, target):
    left = target.floor().long()
    right = left + 1
    weight_right = target - left
    # The final bin cannot have a right neighbor; callers must validate the range.
    assert ((target >= 0) & (target < logits.shape[-1] - 1)).all()
    return ((1 - weight_right) * F.cross_entropy(logits, left, reduction='none')
            + weight_right * F.cross_entropy(logits, right, reduction='none')).mean()


def expected_distance(logits):
    return (logits.softmax(-1) * torch.arange(logits.shape[-1], dtype=logits.dtype)).sum(-1)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    target = torch.tensor([1.25])  # feature-cell units; ideal bin weights .75,.25
    logits = nn.Parameter(torch.zeros(1, 4))
    optimizer = torch.optim.SGD([logits], lr=.5)
    loss = dfl(logits, target)
    loss.backward()
    wanted_grad = torch.tensor([[.25, -.50, .0, .25]])
    assert torch.allclose(logits.grad, wanted_grad)
    print(f'uniform expectation={expected_distance(logits).item():.2f}; DFL={loss.item():.6f}')
    print('initial logit gradient:', logits.grad.tolist())
    optimizer.zero_grad()
    for _ in range(400):
        optimizer.zero_grad()
        dfl(logits, target).backward()
        optimizer.step()
    probs = logits.softmax(-1).detach()
    distance = expected_distance(logits).detach()
    assert abs(distance.item() - 1.25) < .02
    assert probs[0, 1] > .73 and probs[0, 2] > .24
    # round in float64 so 0.0025 prints as 0.0025, not as float32's nearest value 0.0024999999...
    print('learned bin probabilities:', probs.double().round(decimals=4).tolist())
    print(f'learned expectation={distance.item():.4f} cells = {8 * distance.item():.4f} pixels at stride 8')
    # An expectation alone does not fix the distribution.
    a, b = torch.tensor([0., .75, .25, 0.]), torch.tensor([.375, 0., .625, 0.])
    assert torch.allclose((a * torch.arange(4)).sum(), (b * torch.arange(4)).sum())
    print('different distributions can share expectation=1.25')


if __name__ == '__main__':
    main()
