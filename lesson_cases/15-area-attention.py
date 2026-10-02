"""Compare full attention and contiguous-token area attention with shared Q,K,V."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def attend(tokens, projection, areas):
    b, n, c = tokens.shape
    assert areas > 0 and n % areas == 0
    q, k, v = projection(tokens).chunk(3, dim=-1)
    q, k, v = [t.reshape(b * areas, n // areas, c) for t in (q, k, v)]
    weights = (q @ k.transpose(-2, -1) / math.sqrt(c)).softmax(-1)
    return (weights @ v).reshape(b, n, c), weights


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    areas = 4  # Change only this value to 2 or 1 for the area-count exercise.
    changed_index = 15  # Change only this value to 3 for the near-token exercise.
    query_index = 0
    # H=4,W=4: area=4 gives four horizontal strips, not four square windows.
    tokens = torch.arange(16.).reshape(1, 16, 1).repeat(1, 1, 2) / 16
    projection = nn.Linear(2, 6, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(2).repeat(3, 1))
    full, full_weights = attend(tokens, projection, 1)
    area, area_weights = attend(tokens, projection, areas)
    changed = tokens.clone()
    changed[:, changed_index] += 5
    full_changed, _ = attend(changed, projection, 1)
    area_changed, _ = attend(changed, projection, areas)
    area_size = tokens.shape[1] // areas
    same_area = query_index // area_size == changed_index // area_size
    full_affected = not torch.allclose(full[:, query_index], full_changed[:, query_index])
    area_affected = not torch.allclose(area[:, query_index], area_changed[:, query_index])
    assert full_affected and area_affected == same_area
    assert full_weights.numel() == tokens.shape[0] * tokens.shape[1] ** 2
    assert area_weights.numel() == tokens.shape[0] * tokens.shape[1] ** 2 // areas
    print('full affinity shape/count:', tuple(full_weights.shape), full_weights.numel())
    print('area affinity shape/count:', tuple(area_weights.shape), area_weights.numel())
    print(f'first-token output full={full[0,0,0].item():.4f}, area={area[0,0,0].item():.4f}')
    print(f'changing token {changed_index} affects token {query_index}: full={full_affected}, area={area_affected}; same area={same_area}')
    optimizer = torch.optim.SGD(projection.parameters(), lr=.05)
    optimizer.zero_grad()
    loss = F.mse_loss(area, tokens)
    loss.backward()
    assert projection.weight.grad.abs().sum() > 0
    optimizer.step()
    print(f'area-attention backward/step verified; loss={loss.item():.4f}')


if __name__ == '__main__':
    main()
