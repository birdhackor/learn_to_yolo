"""The supervision must change before removing overlap suppression."""
import torch
from torch import nn
from torch.nn import functional as F


def iou(a, b):
    inter = (torch.minimum(a[2:], b[2:]) - torch.maximum(a[:2], b[:2])).clamp(min=0).prod()
    return inter / ((a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - inter)


def nms(boxes, scores, threshold=.5):
    order = scores.argsort(descending=True).tolist()
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) <= threshold]
    return keep


def fit_scores(target):
    logits = nn.Parameter(torch.zeros(4))
    optimizer = torch.optim.SGD([logits], lr=1.)
    for _ in range(100):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        optimizer.step()
    return logits.sigmoid().detach()


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 0., 11., 10.], [20., 0., 30., 10.], [40., 0., 50., 10.]])
    many = fit_scores(torch.tensor([1., 1., 1., 0.]))
    one = fit_scores(torch.tensor([1., 0., 1., 0.]))
    many_ids = torch.where(many > .5)[0].tolist()
    one_ids = torch.where(one > .5)[0].tolist()
    kept = nms(boxes[many_ids], many[many_ids])
    many_after_nms = [many_ids[i] for i in kept]
    assert len(many_ids) == 3 and len(many_after_nms) == 2
    assert one_ids == [0, 2]
    manual_scores = torch.tensor([.92, .90, .80, .05])
    top2 = manual_scores.topk(2).indices.tolist()
    assert top2 == [0, 1]  # top-k itself cannot detect duplication
    print('many-head scores:', many.round(decimals=3).tolist())
    print('one-head scores:', one.round(decimals=3).tolist())
    print('many/no NMS:', many_ids, 'many/NMS:', many_after_nms, 'one/no NMS:', one_ids)
    print('top-2 on duplicate-heavy scores:', top2, '(misses object at candidate 2)')
    print(f'duplicate IoU={iou(boxes[0], boxes[1]).item():.4f}')


if __name__ == '__main__':
    main()
