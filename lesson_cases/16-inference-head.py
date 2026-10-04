"""Make a deployment module that actually excludes its auxiliary training head."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


def decode_ltrb(points, distance, stride):
    return torch.cat((points - distance[..., :2] * stride,
                      points + distance[..., 2:] * stride), -1)


class DualToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4))
        self.many = nn.Conv2d(8, 6, 1)
        self.one = nn.Conv2d(8, 6, 1)

    def forward(self, image):
        features = self.backbone(image)
        return {'many': self.many(features), 'one': self.one(features.detach())}


class DeployToy(nn.Module):
    def __init__(self, trained):
        super().__init__()
        self.backbone = copy.deepcopy(trained.backbone)
        self.head = copy.deepcopy(trained.one)

    def forward(self, image):
        raw = self.head(self.backbone(image)).flatten(2).transpose(1, 2)  # B,16,6
        distance = F.softplus(raw[..., :4])  # teaching choice, not official raw YOLO26 regression
        axis = (torch.arange(4, device=image.device, dtype=image.dtype) + .5) * 16
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        points = torch.stack((x, y), -1).reshape(1, 16, 2)
        boxes = decode_ltrb(points, distance, stride=16)
        scores, labels = raw[..., 4:].sigmoid().max(-1)  # single-label-per-candidate simplification
        top_scores, ids = scores.topk(3, dim=1)
        return boxes.gather(1, ids[..., None].expand(-1, -1, 4)), top_scores, labels.gather(1, ids)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    # Check l,t,r,b ordering and class-score semantics before using random head output.
    known_distance = torch.tensor([[1., .5, 2., 1.5]])
    known_raw = torch.cat((torch.log(torch.expm1(known_distance)), torch.tensor([[0., math.log(3)]])), dim=-1)
    manual_box = decode_ltrb(torch.tensor([[24., 40.]]), F.softplus(known_raw[..., :4]), stride=16)
    class_scores = known_raw[..., 4:].sigmoid()
    manual_score, manual_label = class_scores.max(-1)
    assert torch.allclose(manual_box, torch.tensor([[8., 32., 56., 64.]]))
    assert torch.allclose(class_scores, torch.tensor([[.5, .75]]))
    assert manual_label.tolist() == [1] and torch.allclose(manual_score, torch.tensor([.75]))
    print('manual ltrb decode box:', manual_box.round(decimals=4).tolist())
    print('manual class probabilities:', class_scores.round(decimals=4).tolist(),
          f'; winner label={manual_label.item()}, score={manual_score.round(decimals=4).item()}')
    model = DualToy()
    image = torch.rand(2, 3, 64, 64)
    optimizer = torch.optim.SGD(model.parameters(), lr=.02)
    optimizer.zero_grad()
    output = model(image)
    loss = output['many'].square().mean() + output['one'].square().mean()
    loss.backward()
    before = model.one.weight.detach().clone()
    optimizer.step()
    # The update must move the one head; otherwise a stale copy of it would pass the parity check below.
    assert not torch.equal(before, model.one.weight)
    deploy = DeployToy(model).eval()
    with torch.no_grad():
        reference = model(image)['one']
        deploy_raw = deploy.head(deploy.backbone(image))
        # deepcopy keeps every weight bit and the arithmetic is the same, so demand exact equality.
        assert torch.equal(reference, deploy_raw)
        boxes, scores, labels = deploy(image)
    assert boxes.shape == (2, 3, 4) and scores.shape == (2, 3)
    assert not any('many' in name for name, _ in deploy.named_parameters())
    print('training output keys:', list(output))
    print('deploy boxes / scores / labels:', tuple(boxes.shape), tuple(scores.shape), tuple(labels.shape))
    print('parameters training / deploy:', sum(p.numel() for p in model.parameters()), sum(p.numel() for p in deploy.parameters()))
    print('retained one-head raw output vs reference, max abs difference:', (deploy_raw - reference).abs().max().item())
    print('top-k has no pairwise IoU/NMS; random toy predictions are not accuracy evidence')


if __name__ == '__main__':
    main()
