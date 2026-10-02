"""One object/image, separate class/box heads; three-step CPU smoke test."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/miniyolo-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/miniyolo-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from torch import nn


def to_xyxy(boxes):
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), -1)


class Localizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.class_head = nn.Linear(4, 2)
        self.box_head = nn.Linear(4 * 16 * 16, 4)

    def forward(self, images):
        features = self.body(images)
        return self.class_head(features.mean((2, 3))), self.box_head(features.flatten(1)).sigmoid()


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    left, right = torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)
    left[0, 0, 1, 0], right[0, 0, 1, 3] = 1, 1
    assert torch.equal(left.mean((2, 3)), right.mean((2, 3)))
    print(f"two different positions -> same global mean={left.mean().item():.4f}")
    images = torch.zeros(2, 3, 32, 32)
    pixel_boxes = torch.tensor([[4., 6., 16., 18.], [16., 10., 28., 22.]])
    labels = torch.tensor([0, 1])
    for i, box in enumerate(pixel_boxes.long()):
        x1, y1, x2, y2 = box.tolist()
        images[i, 0 if i == 0 else 2, y1:y2, x1:x2] = 1
    centers = (pixel_boxes[:, :2] + pixel_boxes[:, 2:]) / 2
    sizes = pixel_boxes[:, 2:] - pixel_boxes[:, :2]
    targets = torch.cat((centers, sizes), 1) / 32
    assert torch.allclose(targets[0], torch.tensor([.3125, .375, .375, .375]))
    model = Localizer()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.box_head.weight.detach().clone()
    model.train()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits, boxes = model(images)
        class_loss = nn.functional.cross_entropy(logits, labels)
        box_loss = nn.functional.mse_loss(boxes, targets)
        loss = class_loss + 5 * box_loss
        loss.backward()
        assert model.box_head.weight.grad.abs().sum() > 0
        optimizer.step()
        print(f"step={step}, classification={class_loss.item():.4f}, "
              f"box={box_loss.item():.4f}, total={loss.item():.4f}")
    assert not torch.equal(before, model.box_head.weight)
    model.eval()
    with torch.no_grad():
        logits, normalized_boxes = model(images)
        predicted_boxes = to_xyxy(normalized_boxes) * 32
    assert logits.shape == (2, 2) and normalized_boxes.shape == (2, 4)
    print(f"first_target_cxcywh={targets[0].tolist()}, class_shape={tuple(logits.shape)}, box_shape={tuple(normalized_boxes.shape)}")
    print(f"predicted pixel xyxy={predicted_boxes.tolist()}")
    fig, axes = plt.subplots(1, 2, figsize=(7, 4), constrained_layout=True)
    for i, ax in enumerate(axes):
        ax.imshow(images[i].permute(1, 2, 0).numpy(), extent=(0, 32, 32, 0))
        for box, color, label in ((pixel_boxes[i], "lime", "GT"), (predicted_boxes[i], "gold", "pred")):
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, label=label))
        ax.set_xlim(0, 32)
        ax.set_ylim(32, 0)
        ax.legend()
        ax.set_title("Synthetic image; only 3 steps")
    path = Path("artifacts/04-localization.png")
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"overlay={path}; no held-out detection claim")


if __name__ == "__main__":
    main()
