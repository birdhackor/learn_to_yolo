"""C3k2-inspired split-transform-concatenate, not the full YOLO11 architecture."""
import torch
from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.layers(x)


class SplitAggregate(nn.Module):
    def __init__(self, channels=8, hidden=4, blocks=2):
        super().__init__()
        self.project = nn.Conv2d(channels, 2 * hidden, 1)
        self.blocks = nn.ModuleList(Bottleneck(hidden) for _ in range(blocks))
        self.fuse = nn.Conv2d((2 + blocks) * hidden, channels, 1)

    def forward(self, x, inspect=False):
        a, b = self.project(x).chunk(2, dim=1)
        paths = [a, b]
        for block in self.blocks:
            paths.append(block(paths[-1]))
        joined = torch.cat(paths, dim=1)
        output = self.fuse(joined)
        return (output, paths, joined) if inspect else output


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    x = torch.randn(2, 8, 8, 8)
    plain = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1))
    module = SplitAggregate()
    optimizer = torch.optim.SGD(module.parameters(), lr=.02)
    target = x.roll(1, dims=-1)  # known local transformation; no detection-accuracy claim
    before = F.mse_loss(module(x), target).item()
    for _ in range(30):
        optimizer.zero_grad()
        loss = F.mse_loss(module(x), target)
        loss.backward()
        optimizer.step()
    after = F.mse_loss(module(x), target).item()
    assert module(x).shape == x.shape and after < before
    # Inspect direct concat segments separately from each path's total gradient.
    # b1's total gradient also includes its downstream use as the input to b2.
    optimizer.zero_grad()
    output, paths, joined = module(x, inspect=True)
    joined.retain_grad()
    for path in paths:
        path.retain_grad()
    F.mse_loss(output, target).backward()
    hidden = module.project.out_channels // 2
    assert len(paths) == 2 + len(module.blocks)
    for i, path in enumerate(paths):
        segment = slice(i * hidden, (i + 1) * hidden)
        assert torch.equal(joined[:, segment], path)
        assert path.grad is not None and path.grad.abs().sum() > 0
        assert joined.grad[:, segment].abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    print('input/output:', tuple(x.shape), tuple(module(x).shape))
    channels = ' + '.join(str(hidden) for _ in paths)
    print(f'concatenated channels: {channels} = {module.fuse.in_channels}; fuse {module.fuse.in_channels} -> {module.fuse.out_channels}')
    print('parameters plain / split:', sum(p.numel() for p in plain.parameters()), sum(p.numel() for p in module.parameters()))
    print(f'local transformation MSE {before:.4f} -> {after:.4f}')
    print('concat order, each direct concat-segment gradient, and each path total gradient: verified')
    print('direct concat-segment gradient L1:', [round(float(joined.grad[:, i*hidden:(i+1)*hidden].abs().sum()), 4) for i in range(len(paths))])


if __name__ == '__main__':
    main()
