"""Budget-matched small plain/residual comparison, not a ResNet benchmark."""
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
        left = 2 + (i + shift) % 4
        top = 3 + shift
        images[i, 0 if labels[i] == 0 else 2, top:top + 8, left:left + 8] = 1
    return images, labels


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
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.stem.weight.detach().clone()
        start = perf_counter()
        model.train()
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x)
            loss = nn.functional.cross_entropy(logits, train_y)
            loss.backward()
            grad_norm = model.stem.weight.grad.norm().item()
            assert grad_norm > 0 and torch.isfinite(loss)
            assert torch.isfinite(model.stem.weight.grad).all()
            optimizer.step()
            print(f"{name} step={step}, loss={loss.item():.4f}, stem_grad_norm={grad_norm:.6f}")
        elapsed = perf_counter() - start
        assert not torch.equal(before, model.stem.weight)
        model.eval()
        with torch.no_grad():
            acc = (model(val_x).argmax(1) == val_y).float().mean().item()
        print(f"{name}: params={params}, MACs/image=248840, shortcut_adds/image="
              f"{3072 if name == 'residual' else 0}, validation_accuracy={acc:.2f}, "
              f"3_step_seconds={elapsed:.4f}")
    print("Same initial weights/data/optimizer/steps; 3 steps and 4 validation images do not rank architectures.")


if __name__ == "__main__":
    main()
