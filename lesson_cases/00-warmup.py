"""A scalar update with a hand-computable answer; CPU only."""
import torch


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x, target = torch.tensor([[2.0]]), torch.tensor([[4.0]])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)
    loss = ((prediction - target) ** 2).mean()
    loss.backward()
    gradient = model.weight.grad.item()
    before = model.weight.item()
    optimizer.step()
    after = model.weight.item()
    model.eval()
    with torch.no_grad():
        new_loss = ((model(x) - target) ** 2).mean().item()
    print(f"x_shape={tuple(x.shape)}, weight_shape={tuple(model.weight.shape)}")
    print(f"prediction={prediction.item():.2f}, loss={loss.item():.2f}, gradient={gradient:.2f}")
    print(f"weight: {before:.2f} -> {after:.2f}; new_loss={new_loss:.2f}")
    assert abs(gradient + 8.0) < 1e-6
    assert abs(after - 1.8) < 1e-6 and abs(new_loss - 0.16) < 1e-5

    # eval() changes layer behavior; it does not disable autograd.
    assert model.training is False
    assert model(x).requires_grad
    # A newly created model needs an optimizer that references its parameters.
    replacement = torch.nn.Linear(1, 1, bias=False)
    new_optimizer = torch.optim.SGD(replacement.parameters(), lr=0.1)
    assert new_optimizer.param_groups[0]["params"][0] is replacement.weight
    print("eval still tracks gradients; replacement optimizer points to replacement model")


if __name__ == "__main__":
    main()
