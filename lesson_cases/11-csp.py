import torch
from torch import nn


class CSP(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = nn.Sequential(nn.Conv2d(4,4,3,padding=1),nn.ReLU(),
                                    nn.Conv2d(4,4,3,padding=1),nn.ReLU())
        self.fuse = nn.Conv2d(8,8,1)

    def forward(self,x):
        bypass, transformed = x.chunk(2,dim=1)
        return self.fuse(torch.cat([bypass,self.branch(transformed)],dim=1))


class Full(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = nn.Sequential(nn.Conv2d(8,8,3,padding=1),nn.ReLU(),
                                    nn.Conv2d(8,8,3,padding=1),nn.ReLU())
        self.fuse = nn.Conv2d(8,8,1)

    def forward(self,x):
        return self.fuse(self.branch(x))


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    a = torch.tensor([[[[1.]],[[2.]]]])
    b = torch.tensor([[[[10.]],[[20.]]]])
    joined, added = torch.cat([a,b],dim=1), a+b
    assert joined.shape == (1,4,1,1) and joined.flatten().tolist() == [1,2,10,20]
    assert added.shape == (1,2,1,1) and added.flatten().tolist() == [11,22]
    print('concat',tuple(joined.shape),joined.flatten().tolist(),
          'add',tuple(added.shape),added.flatten().tolist())
    full,csp = Full(),CSP()
    counts = [sum(p.numel() for p in m.parameters()) for m in [full,csp]]
    assert counts == [1240,368]
    for name,model in [('full',full),('CSP',csp)]:
        x = torch.randn(2,8,8,8,requires_grad=True)
        optimizer = torch.optim.SGD(model.parameters(),lr=.01)
        before = model.fuse.weight.detach().clone()
        output = model(x)
        assert output.shape == x.shape
        loss = output.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in model.parameters())
        assert x.grad[:,:4].abs().sum() > 0
        assert x.grad[:,4:].abs().sum() > 0
        optimizer.step()
        assert not torch.equal(before,model.fuse.weight.detach())
        print(name,'input gradient abs sums first4/last4',
              round(x.grad[:,:4].abs().sum().item(),6),round(x.grad[:,4:].abs().sum().item(),6))
        print(name,'output',tuple(output.shape),'finite backward and step')
    print('parameters full/CSP',counts,'capacity differs; no accuracy comparison')


if __name__ == '__main__':
    main()
