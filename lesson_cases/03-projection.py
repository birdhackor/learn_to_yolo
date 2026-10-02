"""A projection aligns shapes and mixes channels; it is a learned transform."""
import torch
from torch import nn


class ProjectionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = nn.Sequential(nn.Conv2d(3, 6, 3, stride=2, padding=1, bias=False),
                                    nn.ReLU(), nn.Conv2d(6, 6, 3, padding=1, bias=False))
        self.projection = nn.Conv2d(3, 6, 1, stride=2, bias=False)

    def forward(self, x):
        main, shortcut = self.branch(x), self.projection(x)
        assert main.shape == shortcut.shape
        return main + shortcut


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    block = ProjectionBlock()
    with torch.no_grad():
        for p in block.branch.parameters():
            p.zero_()
        block.projection.weight.zero_()
        block.projection.weight[0, :, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    image = torch.tensor([1.0, 10.0, 100.0]).view(1, 3, 1, 1).expand(1, 3, 4, 4)
    output = block(image)
    assert output.shape == (1, 6, 2, 2)
    assert torch.equal(output[0, 0], torch.full((2, 2), 321.0))
    print(f"input={tuple(image.shape)}, main={tuple(block.branch(image).shape)}, "
          f"projection={tuple(block.projection(image).shape)}, output={tuple(output.shape)}")
    print(f"first output channel={output[0, 0].detach().tolist()}; 1*1 + 2*10 + 3*100 = 321")

    learned = ProjectionBlock()
    inputs = torch.randn(2, 3, 4, 4)
    target = torch.zeros(2, 6, 2, 2)
    optimizer = torch.optim.SGD(learned.parameters(), lr=0.05)
    before = learned.projection.weight.detach().clone()
    learned.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(learned(inputs), target)
        loss.backward()
        assert learned.projection.weight.grad.abs().sum() > 0
        optimizer.step()
        print(f"step={step}, loss={loss.item():.4f}")
    assert not torch.equal(before, learned.projection.weight)
    assert sum(p.numel() for p in learned.parameters()) == 504
    print("projection changed; block parameters=504; no classification-quality claim")


if __name__ == "__main__":
    main()
