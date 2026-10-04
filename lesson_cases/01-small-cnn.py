"""Three CPU training steps and an honest prediction panel, not a trained benchmark."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/miniyolo-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/miniyolo-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn


def make_batch():
    images = torch.zeros(8, 3, 32, 32)
    labels = torch.arange(8) % 2
    for i in range(8):
        # top follows (i // 2) % 2, not the label i % 2, so height says nothing about the class
        left, top = 4 + i % 3 * 4, 6 + (i // 2) % 2 * 5
        images[i, 0 if labels[i] == 0 else 2, top:top + 12, left:left + 12] = 1
    shapes = images.amax(1)  # [8,32,32]: where each rectangle is, color ignored
    tops = shapes.amax(2).argmax(1)  # first lit row of each image
    for c in (0, 1):
        assert sorted(tops[labels == c].tolist()) == [6, 6, 11, 11], tops  # 2 high, 2 low per class
    # A red and a blue rectangle cover the same pixels, so only color separates the classes.
    assert any(torch.equal(r, b) for r in shapes[labels == 0] for b in shapes[labels == 1])
    return images, labels


class SmallCNN(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(width * 2, 2)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    images, labels = make_batch()
    assert images.shape == (8, 3, 32, 32) and images.dtype == torch.float32
    assert images.min() >= 0 and images.max() <= 1
    model = SmallCNN()
    x = images
    macs = 0
    for layer in model.features:
        x = layer(x)
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            print(f"{type(layer).__name__}: {tuple(x.shape)}")
        if isinstance(layer, nn.Conv2d):
            macs += x[0].numel() * layer.in_channels * layer.kernel_size[0] ** 2
    macs += model.head.in_features * model.head.out_features
    params = sum(p.numel() for p in model.parameters())
    print(f"parameters={params}; multiply-accumulates/image={macs}")
    assert params == 1158 and macs == 479248
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.features[0].weight.detach().clone()
    model.train()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        assert torch.isfinite(model.features[0].weight.grad).all()
        optimizer.step()
        print(f"step={step}, loss={loss.item():.4f}")
    assert not torch.equal(before, model.features[0].weight)
    model.eval()
    with torch.no_grad():
        predictions = model(images).argmax(1)
    errors = (predictions != labels).nonzero().flatten().tolist()
    print(f"predictions={predictions.tolist()}, labels={labels.tolist()}, error_indices={errors}")
    print("Three-step smoke test only; accuracy here is not held-out performance.")
    fig, axes = plt.subplots(2, 4, figsize=(9, 6), constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        ax.set_title(f"{i}: GT={labels[i].item()} pred={predictions[i].item()}",
                     color="red" if i in errors else "black")
        ax.axis("off")
    fig.suptitle("Synthetic images; only 3 updates; red titles = mistakes")
    output = Path("artifacts/01-small-cnn.png")
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)
    print(f"panel={output}")


if __name__ == "__main__":
    main()
