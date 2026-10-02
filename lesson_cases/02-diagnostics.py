"""Controlled failures: disconnected gradient, invalid label, spurious correlation."""
import torch
from torch import nn


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    body, head = nn.Linear(2, 3), nn.Linear(3, 2)
    optimizer = torch.optim.SGD(list(body.parameters()) + list(head.parameters()), lr=0.2)
    x, labels = torch.tensor([[-0.2, -1.0], [0.2, 1.0]]), torch.tensor([0, 1])
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(head(body(x).detach()), labels)
    loss.backward()
    assert body.weight.grad is None and head.weight.grad is not None
    print("broken graph: body.weight.grad=None, head.weight.grad exists")
    optimizer.zero_grad(set_to_none=True)
    before = body.weight.detach().clone()
    loss = nn.functional.cross_entropy(head(body(x)), labels)
    loss.backward()
    assert body.weight.grad is not None and body.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, body.weight)
    print("repaired graph: body gradient nonzero and body parameter changed")

    bad_labels = torch.tensor([0, 2])
    valid_labels = ((bad_labels >= 0) & (bad_labels < 2)).all()
    assert not valid_labels
    print("label preflight: [0, 2] invalid for two classes; expected IDs 0 or 1")

    # Feature 0 is a weak true cue; feature 1 is a strong incidental corner cue.
    train_x, train_y = x.repeat(4, 1), labels.repeat(4)
    validation_x = torch.tensor([[-0.2, 1.0], [0.2, -1.0]]).repeat(4, 1)
    validation_y = labels.repeat(4)
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    for step in range(20):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = nn.functional.cross_entropy(logits, train_y)
        loss.backward()
        assert torch.isfinite(model.weight.grad).all()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = nn.functional.cross_entropy(model(train_x), train_y).item()
            val_loss = nn.functional.cross_entropy(model(validation_x), validation_y).item()
        if step in (0, 9, 19):
            print(f"step={step}: train_loss={train_loss:.4f}, validation_loss={val_loss:.4f}")
    with torch.no_grad():
        train_acc = (model(train_x).argmax(1) == train_y).float().mean().item()
        val_acc = (model(validation_x).argmax(1) == validation_y).float().mean().item()
    print(f"train_accuracy={train_acc:.2f}; validation_accuracy={val_acc:.2f}")
    print(f"learned weights={model.weight.detach().tolist()}")
    assert train_acc == 1.0 and val_acc == 0.0
    print("Synthetic distribution shift demonstrates: tiny-set overfit is not generalization.")


if __name__ == "__main__":
    main()
