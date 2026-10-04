"""Budget-matched small plain/residual comparison, not a ResNet benchmark."""
import copy
from time import perf_counter
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, residual):
        super().__init__()
        self.residual = residual
        self.branch = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1, bias=False), nn.ReLU(),
                                    nn.Conv2d(4, 4, 3, padding=1, bias=False))

    def forward(self, x):
        correction = self.branch(x)
        return x + correction if self.residual else correction


class Classifier(nn.Module):
    def __init__(self, residual):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(residual) for _ in range(3)])
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        x = self.blocks(nn.functional.relu(self.stem(x)))
        return self.head(x.mean((2, 3)))


def data(n, shift):
    images, labels = torch.zeros(n, 3, 16, 16), torch.arange(n) % 2
    for i in range(n):
        # left follows i // 2, not the label i % 2, so each red/blue pair shares one position
        left = 2 + ((i // 2) + shift) % 4
        top = 3 + shift
        images[i, 0 if labels[i] == 0 else 2, top:top + 8, left:left + 8] = 1
    shapes = images.amax(1)  # [n,16,16]: where each square is, color ignored
    tops, lefts = shapes.amax(2).argmax(1), shapes.amax(1).argmax(1)  # first lit row, first lit column
    corners = torch.stack((tops, lefts), 1)  # [n,2]: (top, left) of each square
    # Red and blue use the same (top, left) positions equally often (equal sorted lists); with every
    # square 8x8 (drawn above), only color separates the classes.
    assert sorted(corners[labels == 0].tolist()) == sorted(corners[labels == 1].tolist()), corners.tolist()
    return images, labels


def count_per_image(model, images):
    """Multiply-accumulates and shortcut additions per image, counted by forward hooks."""
    macs, shortcut_adds = [], []

    def hook(layer, inputs, output):
        values = output[0].numel()  # values this layer outputs for image 0
        if isinstance(layer, nn.Conv2d):  # each output value needs in_channels x 3 x 3 MACs
            macs.append(values * layer.in_channels * layer.kernel_size[0] ** 2)
        elif isinstance(layer, nn.Linear):  # each output value needs in_features MACs
            macs.append(values * layer.in_features)
        elif isinstance(layer, Block) and layer.residual:  # x + F(x): one add per output value
            shortcut_adds.append(values)

    handles = [layer.register_forward_hook(hook) for layer in model.modules()
               if isinstance(layer, (nn.Conv2d, nn.Linear, Block))]
    with torch.no_grad():
        model(images)
    for handle in handles:
        handle.remove()
    return sum(macs), sum(shortcut_adds)


def train_step(model, optimizer, images, labels):
    """One SGD update on the whole batch; returns the pre-update loss and stem-weight gradient L2 norm."""
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    grad_norm = model.stem.weight.grad.norm().item()
    assert grad_norm > 0 and torch.isfinite(loss)
    assert torch.isfinite(model.stem.weight.grad).all()
    optimizer.step()
    return loss.item(), grad_norm


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    train_x, train_y = data(8, 0)
    val_x, val_y = data(4, 2)
    assert not any(torch.equal(t, v) for t in train_x for v in val_x)
    plain, residual = Classifier(False), Classifier(True)
    residual.load_state_dict(plain.state_dict())
    for p, r in zip(plain.parameters(), residual.parameters()):
        assert torch.equal(p, r)
    for name, model in (("plain", plain), ("residual", residual)):
        params = sum(p.numel() for p in model.parameters())
        assert params == 986
        macs, shortcut_adds = count_per_image(model, train_x)
        assert macs == 248840
        assert shortcut_adds == (3072 if name == "residual" else 0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.stem.weight.detach().clone()
        # Untimed warm-up: the same step once on a throwaway copy, so the step's first-call
        # start-up work is not charged to whichever model happens to be timed first.
        throwaway = copy.deepcopy(model)
        train_step(throwaway, torch.optim.SGD(throwaway.parameters(), lr=0.1), train_x, train_y)
        assert torch.equal(before, model.stem.weight)  # the warm-up left the real model untouched
        start = perf_counter()
        model.train()
        for step in range(3):
            loss, grad_norm = train_step(model, optimizer, train_x, train_y)
            print(f"{name} step={step}, loss={loss:.4f}, stem_grad_norm={grad_norm:.6f}")
        elapsed = perf_counter() - start
        assert not torch.equal(before, model.stem.weight)
        model.eval()
        with torch.no_grad():
            acc = (model(val_x).argmax(1) == val_y).float().mean().item()
        print(f"{name}: params={params}, MACs/image={macs}, shortcut_adds/image={shortcut_adds}, "
              f"validation_accuracy={acc:.2f}, 3_step_seconds={elapsed:.4f}")
    print("Same initial weights/data/optimizer/steps; 3 steps and 4 validation images do not rank architectures.")


if __name__ == "__main__":
    main()
