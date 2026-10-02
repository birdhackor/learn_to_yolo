"""Identity is an exact statement about same-shape addition."""
import torch
from torch import nn


class IdentityBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.ReLU(),
                                    nn.Conv2d(channels, channels, 3, padding=1, bias=False))

    def forward(self, x):
        return x + self.branch(x)  # No post-add ReLU in this teaching block.


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    block = IdentityBlock(1)
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
    x = torch.tensor([[[[-2.0, -1.0], [0.0, 1.0]]]], requires_grad=True)
    y = block(x)
    assert torch.equal(x, y)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    print(f"x={x.detach().flatten().tolist()}, y={y.detach().flatten().tolist()}")
    print(f"input_gradient={x.grad.flatten().tolist()}; F=0 preserves negative values too")

    trained = IdentityBlock(4)
    inputs, target = torch.randn(2, 4, 8, 8), torch.zeros(2, 4, 8, 8)
    optimizer = torch.optim.SGD(trained.parameters(), lr=0.05)
    before = trained.branch[2].weight.detach().clone()
    trained.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = trained(inputs)
        loss = nn.functional.mse_loss(prediction, target)
        loss.backward()
        assert trained.branch[2].weight.grad.abs().sum() > 0
        optimizer.step()
        print(f"step={step}, shape={tuple(prediction.shape)}, loss={loss.item():.4f}")
    assert not torch.equal(before, trained.branch[2].weight)
    print("random branch updated; this does not measure ResNet classification quality")


if __name__ == "__main__":
    main()
