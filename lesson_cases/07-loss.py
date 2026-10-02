import math
import torch
from miniyolo.targets import build_targets
from miniyolo.losses import grid_loss


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    scene = {"boxes": torch.tensor([[8., 12., 24., 28.]]), "labels": torch.tensor([0])}
    target = build_targets([scene], 4, 64, 2)
    prediction = torch.zeros(1, 4, 4, 7, requires_grad=True)
    parts = grid_loss(prediction, target)
    expected_box = (.5**2 + .25**2 + .25**2 + .25**2) / 4
    assert abs(parts['box'].item() - expected_box) < 1e-7
    assert abs(parts['objectness'].item() - math.log(2)) < 1e-6
    assert abs(parts['classification'].item() - math.log(2)) < 1e-6
    assert abs(parts['total'].item() - (5*expected_box + 2*math.log(2))) < 1e-6
    parts['total'].backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, 0, 0, :4].abs().sum() == 0
    assert prediction.grad[0, 0, 0, 4] > 0  # gradient descent lowers background logit
    assert prediction.grad[0, 1, 1, 4] < 0  # raises positive objectness logit
    pos = target['positive']
    assert (prediction.grad[~pos][..., :4] == 0).all()
    assert (prediction.grad[~pos][..., 5:] == 0).all()
    assert torch.allclose(prediction.grad[..., 4][pos], torch.full((1,), -1/32))
    assert torch.allclose(prediction.grad[..., 4][~pos], torch.full((15,), 1/32))
    print({k: round(v.item(), 6) for k, v in parts.items()})
    print('positive/background objectness gradients',
          prediction.grad[0, 1, 1, 4].item(), prediction.grad[0, 0, 0, 4].item())
    empty = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
    empty_pred = torch.zeros(1, 4, 4, 7, requires_grad=True)
    empty_parts = grid_loss(empty_pred, build_targets([empty], 4, 64, 2))
    empty_parts['total'].backward()
    assert empty_parts['box'].item() == empty_parts['classification'].item() == 0
    assert torch.isfinite(empty_pred.grad).all()
    assert abs(empty_parts['objectness'].item() - math.log(2)) < 1e-6
    assert abs(empty_parts['total'].item() - math.log(2)) < 1e-6
    assert (empty_pred.grad[..., :4] == 0).all()
    assert (empty_pred.grad[..., 5:] == 0).all()
    assert torch.allclose(empty_pred.grad[..., 4], torch.full((1,4,4),1/32))
    print('empty image: finite backward, box/class=0')


if __name__ == '__main__':
    main()
