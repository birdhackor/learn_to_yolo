import torch
from torch import nn
import torch.nn.functional as F


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv2d(16,8,1)
        self.mix = nn.Conv2d(16,8,3,padding=1)

    def forward(self,shallow,deep):
        reduced = self.reduce(deep)
        up = F.interpolate(reduced,size=shallow.shape[-2:],mode='nearest')
        return self.mix(torch.cat([shallow,up],dim=1))


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    shallow = torch.randn(1,8,8,8,requires_grad=True)
    deep = torch.randn(1,16,4,4,requires_grad=True)
    model = Fusion()
    optimizer = torch.optim.SGD(model.parameters(),lr=.01)
    before = model.mix.weight.detach().clone()
    output = model(shallow,deep)
    assert output.shape == (1,8,8,8)
    loss = output.square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert torch.isfinite(shallow.grad).all() and torch.isfinite(deep.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert shallow.grad.abs().sum() > 0 and deep.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before,model.mix.weight.detach())
    # Different channels can legally concatenate; only spatial axes must match.
    unreduced = F.interpolate(deep.detach(),size=(8,8),mode='nearest')
    joined = torch.cat([shallow.detach(),unreduced],dim=1)
    assert joined.shape == (1,24,8,8)
    assert nn.Conv2d(24,8,3,padding=1)(joined).shape == (1,8,8,8)
    matched = model.reduce(deep.detach())
    matched = F.interpolate(matched,size=(8,8),mode='nearest')
    assert torch.cat([shallow.detach(),matched],dim=1).shape == (1,16,8,8)
    assert (shallow.detach()+matched).shape == (1,8,8,8)
    # A fully hand-checkable nearest-neighbor interpolation example.
    toy = torch.tensor([[[[1.,2.],[3.,4.]]]],requires_grad=True)
    enlarged = F.interpolate(toy,size=(4,4),mode='nearest')
    expected = torch.tensor([[[[1.,1.,2.,2.],[1.,1.,2.,2.],
                               [3.,3.,4.,4.],[3.,3.,4.,4.]]]])
    assert torch.equal(enlarged,expected)
    enlarged.sum().backward()
    assert torch.equal(toy.grad,torch.full_like(toy,4))
    print('nearest example',enlarged[0,0].tolist(),'source gradient',toy.grad[0,0].tolist())
    print('8ch shallow + 8ch upsampled deep -> 16ch concat -> 8ch mixed')
    print('parameters',sum(p.numel() for p in model.parameters()),'both branches backward and one step')


if __name__ == '__main__':
    main()
