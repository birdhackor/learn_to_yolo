"""Exact tiny global matching plus dual branches; not YOLOv10's actual TAL assigner."""
from itertools import permutations
import torch
from torch import nn
from torch.nn import functional as F


def exact_one_to_one(quality):
    # Exhaustive optimum for G<=P, factorial complexity: for teaching only.
    g, p = quality.shape
    best = max(permutations(range(p), g), key=lambda cols: sum(float(quality[i, c]) for i, c in enumerate(cols)))
    owner = torch.full((p,), -1, dtype=torch.long)
    for i, c in enumerate(best):
        owner[c] = i
    return owner, sum(float(quality[i, c]) for i, c in enumerate(best))


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    quality = torch.tensor([[.90, .85, .10], [.88, .10, .20]])
    one_owner, optimum = exact_one_to_one(quality)
    # Independent top-2 then quality-based conflict resolution: simplified one-to-many.
    selected = torch.zeros_like(quality, dtype=torch.bool)
    selected.scatter_(1, quality.topk(2, dim=1).indices, True)
    many_owner = quality.masked_fill(~selected, -1).argmax(0)
    assert one_owner.tolist() == [1, 0, -1]
    assert many_owner.tolist() == [0, 0, 1]
    greedy_value = float(quality[0, 0]) + float(quality[1, 2])
    assert optimum > greedy_value + 1e-7
    inputs = torch.randn(3, 4)
    backbone = nn.Linear(4, 4)
    many_head, one_head = nn.Linear(4, 2), nn.Linear(4, 2)
    features = backbone(inputs)
    many_logits, one_logits = many_head(features), one_head(features.detach())
    def targets(owner):
        t = torch.zeros(3, 2)
        positives = torch.where(owner >= 0)[0]
        t[positives, owner[positives]] = 1
        return t
    many_loss = F.binary_cross_entropy_with_logits(many_logits, targets(many_owner))
    one_loss = F.binary_cross_entropy_with_logits(one_logits, targets(one_owner))
    one_loss.backward(retain_graph=True)
    assert backbone.weight.grad is None
    assert one_head.weight.grad.abs().sum() > 0
    optimizer = torch.optim.SGD(list(backbone.parameters()) + list(many_head.parameters()) + list(one_head.parameters()), lr=.1)
    optimizer.zero_grad()
    (many_loss + one_loss).backward()
    assert backbone.weight.grad.abs().sum() > 0
    old = one_head.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(old, one_head.weight)
    print('one-to-many owner:', many_owner.tolist())
    print('global one-to-one owner:', one_owner.tolist())
    print(f'global quality={optimum:.2f}; greedy quality={greedy_value:.2f}')
    print('detached one-to-one branch trains its head; only many branch trains backbone')
    print(f'loss many={many_loss.item():.4f}, one={one_loss.item():.4f}')


if __name__ == '__main__':
    main()
