"""Separate branch gradients while retaining a shared backbone."""
import torch
from torch import nn
from torch.nn import functional as F


class DecoupledHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 8, 3, padding=1)
        self.box_branch = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 4, 1))
        self.class_branch = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 2, 1))

    def forward(self, x):
        f = F.relu(self.backbone(x))
        return self.box_branch(f), self.class_branch(f)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = DecoupledHead()
    x = torch.rand(2, 3, 4, 4)
    distances = torch.full((2, 4, 4, 4), 1.5)
    classes = torch.zeros(2, 2, 4, 4)
    classes[:, 0] = 1  # independent sigmoid classes; simplified all-positive example
    boxes, logits = model(x)
    box_loss = F.smooth_l1_loss(F.softplus(boxes), distances)
    cls_loss = F.binary_cross_entropy_with_logits(logits, classes)
    model.zero_grad()
    box_loss.backward(retain_graph=True)
    assert model.box_branch[0].weight.grad is not None
    assert model.class_branch[0].weight.grad is None
    box_backbone_grad = model.backbone.weight.grad.clone()
    model.zero_grad()
    cls_loss.backward(retain_graph=True)
    assert model.box_branch[0].weight.grad is None
    class_backbone_grad = model.backbone.weight.grad.clone()
    cosine = F.cosine_similarity(box_backbone_grad.flatten(), class_backbone_grad.flatten(), dim=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    model.zero_grad()
    (box_loss + cls_loss).backward()
    assert torch.allclose(model.backbone.weight.grad, box_backbone_grad + class_backbone_grad, atol=1e-7)
    old = model.backbone.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(old, model.backbone.weight)
    print('box / class shapes:', tuple(boxes.shape), tuple(logits.shape))
    print('classification-only backward leaves box branch grad=None')
    print(f'shared-backbone gradient cosine: {cosine.item():.4f}')
    print('total backbone gradient = box gradient + class gradient: verified')
    print('parameters:', sum(p.numel() for p in model.parameters()))


if __name__ == '__main__':
    main()
