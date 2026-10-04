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


def one_to_many_owner(quality, k=2):
    # Each GT keeps its top-k candidates; a candidate chosen by several GTs goes to the higher quality,
    # and a candidate chosen by no GT stays background (-1). Simplified one-to-many, not YOLOv10's TAL.
    selected = torch.zeros_like(quality, dtype=torch.bool)
    selected.scatter_(1, quality.topk(k, dim=1).indices, True)
    owner = quality.masked_fill(~selected, -1).argmax(0)
    owner[~selected.any(0)] = -1
    return owner


def greedy_one_to_one(quality):
    # Greedy: each step takes the largest number left (unmatched GT, unused candidate) and never revises it.
    # cols[i] is GT i's candidate, the same layout as exact_one_to_one's best.
    g, p = quality.shape
    cols = [-1] * g
    for _ in range(g):
        left = [(i, c) for i in range(g) for c in range(p) if cols[i] == -1 and c not in cols]
        i, c = max(left, key=lambda pair: float(quality[pair]))
        cols[i] = c
    return tuple(cols), sum(float(quality[i, c]) for i, c in enumerate(cols))


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    quality = torch.tensor([[.90, .85, .10], [.88, .10, .20]])
    one_owner, optimum = exact_one_to_one(quality)
    many_owner = one_to_many_owner(quality)
    assert one_owner.tolist() == [1, 0, -1]
    assert many_owner.tolist() == [0, 0, 1]
    assert one_to_many_owner(torch.tensor([[.9, .8, .1], [.7, .6, .2]])).tolist() == [0, 0, -1]
    greedy_cols, greedy_value = greedy_one_to_one(quality)
    assert greedy_cols == (0, 2) and abs(greedy_value - 1.10) < 1e-6
    # The largest number goes first whichever GT it belongs to: with B's p0 at .95, B takes p0 and A gets p1.
    assert greedy_one_to_one(torch.tensor([[.90, .85, .10], [.95, .10, .20]]))[0] == (1, 0)
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
    assert many_head.weight.grad.abs().sum() > 0 and one_head.weight.grad.abs().sum() > 0
    old = one_head.weight.detach().clone()
    old_many = many_head.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(old, one_head.weight)
    assert not torch.equal(old_many, many_head.weight)
    print('one-to-many owner:', many_owner.tolist())
    print('global one-to-one owner:', one_owner.tolist())
    print(f'global quality={optimum:.2f}; greedy quality={greedy_value:.2f}')
    print('detached one-to-one branch trains its head; only many branch trains backbone')
    print(f'loss many={many_loss.item():.4f}, one={one_loss.item():.4f}')


if __name__ == '__main__':
    main()
