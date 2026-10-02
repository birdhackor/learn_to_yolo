import math
import torch
from miniyolo.data import ShapeDataset, collate
from miniyolo.models import GridDetector
from miniyolo.targets import build_targets
from miniyolo.losses import grid_loss


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    dataset = ShapeDataset(n=4, size=64, seed=7, max_objects=1, allow_empty=False)
    images, anns = collate([dataset[i] for i in range(4)])
    target = build_targets(anns, 4, 64, 2)
    model = GridDetector(num_classes=2, grid_size=4, width=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    before = next(model.parameters()).detach().clone()
    model.train()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(images)
        assert prediction.shape == (4, 4, 4, 7)
        parts = grid_loss(prediction, target)
        assert torch.isfinite(parts['total'])
        parts['total'].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        assert math.isfinite(grad) and grad > 0
        optimizer.step()
        with torch.no_grad():
            obj = prediction[..., 4].sigmoid()
            pos_mean = obj[target['positive']].mean().item()
            neg_mean = obj[~target['positive']].mean().item()
        print('step', step, {k: round(v.item(), 4) for k, v in parts.items()},
              'positive', round(pos_mean, 4), 'negative', round(neg_mean, 4))
    assert not torch.equal(before, next(model.parameters()).detach())
    assert target['positive'].sum() == 4
    print('3 real CPU optimizer steps; parameters changed; no generalization claim')


if __name__ == '__main__':
    main()
