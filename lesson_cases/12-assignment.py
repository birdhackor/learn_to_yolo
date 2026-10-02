"""Simplified task-aligned top-k assignment, with conflict and empty-GT handling."""
import torch
from torch import nn
from torch.nn import functional as F


def box_iou(boxes_a, boxes_b):
    intersection = (torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
                    - torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])).clamp(min=0).prod(-1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).prod(-1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).prod(-1)
    return intersection / (area_a[:, None] + area_b[None] - intersection)


@torch.no_grad()
def assign(points, gt, scores, overlaps, k=2):
    # points [P,2], gt [G,4], scores/overlaps [G,P]; -1 denotes background.
    owner = torch.full((len(points),), -1, dtype=torch.long)
    if len(gt) == 0:
        return owner, torch.zeros_like(overlaps)
    inside = ((points[None] > gt[:, None, :2]) & (points[None] < gt[:, None, 2:])).all(-1)
    metric = scores * overlaps.square()  # teaching alpha=1,beta=2; not original defaults
    selected = torch.zeros_like(inside)
    for g in range(len(gt)):
        eligible = torch.where(inside[g] & (metric[g] > 0))[0]
        n = min(k, len(eligible))
        if n:
            selected[g, eligible[metric[g, eligible].topk(n).indices]] = True
    for p in range(len(points)):
        candidates = torch.where(selected[:, p])[0]
        if len(candidates):
            owner[p] = candidates[overlaps[candidates, p].argmax()]
    return owner, metric * inside


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    points = torch.tensor([[8., 8.], [16., 8.], [24., 8.], [40., 8.]])
    gt = torch.tensor([[0., 0., 20., 16.], [12., 0., 32., 16.]])
    scores = torch.tensor([[.9, .7, .1, .8], [.1, .8, .7, .9]])
    predicted_boxes = torch.tensor([[0., 0., 10., 16.], [4., 0., 24., 16.],
                                    [14., 0., 32., 16.], [40., 0., 50., 10.]])
    overlaps = box_iou(gt, predicted_boxes)  # actual shared boxes, not arbitrary incompatible IoUs
    owner, metric = assign(points, gt, scores, overlaps)
    assert owner.tolist() == [0, 0, 1, -1]
    empty, _ = assign(points, gt[:0], scores[:0], overlaps[:0])
    assert empty.tolist() == [-1] * 4
    logits = nn.Parameter(torch.zeros(4))
    optimizer = torch.optim.SGD([logits], lr=1.)
    target = (owner >= 0).float()  # simplified binary foreground, not TAL's quality targets
    optimizer.zero_grad()
    loss = F.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    assert torch.allclose(logits.grad, torch.tensor([-.125, -.125, -.125, .125]))
    optimizer.step()
    print('inside-masked score * IoU^2:', metric.round(decimals=3).tolist())
    print('candidate owner (-1 background):', owner.tolist())
    print('foreground target:', target.tolist())
    print('gradient:', logits.grad.tolist())
    print('empty image owner:', empty.tolist())
    moved_boxes = predicted_boxes.clone()
    moved_boxes[1] = gt[1]
    moved_owner, _ = assign(points, gt, scores, box_iou(gt, moved_boxes))
    assert moved_owner.tolist() == [0, 1, 1, -1]
    print('exercise: move predicted box p1 to GT B; owner:', moved_owner.tolist())


if __name__ == '__main__':
    main()
