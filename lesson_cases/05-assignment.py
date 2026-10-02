"""From variable-length annotations to a one-slot grid target; collisions are explicit."""
import torch
from torch import nn


def build(targets, grid=4, image_size=64, classes=2):
    batch = len(targets)
    boxes = torch.zeros(batch, grid, grid, 4)
    positive = torch.zeros(batch, grid, grid, dtype=torch.bool)
    class_ids = torch.full((batch, grid, grid), -1, dtype=torch.long)
    for b, target in enumerate(targets):
        for box, label in zip(target["boxes"], target["labels"]):
            assert (box[2:] > box[:2]).all() and box.min() >= 0 and box.max() <= image_size
            assert 0 <= label < classes
            center, size = (box[:2] + box[2:]) / 2, box[2:] - box[:2]
            grid_center = center / image_size * grid
            col, row = grid_center.floor().long().tolist()
            if positive[b, row, col]:
                raise ValueError(f"same-cell collision at image={b}, row={row}, col={col}")
            boxes[b, row, col] = torch.cat((grid_center - grid_center.floor(), size / image_size))
            positive[b, row, col] = True
            class_ids[b, row, col] = label
    return boxes, positive.float(), class_ids, positive


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    two = {"boxes": torch.tensor([[4., 4., 20., 20.], [36., 36., 52., 52.]]),
           "labels": torch.tensor([0, 1])}
    empty = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
    box_targets, objectness, class_ids, positive = build([two, empty])
    assert positive.sum() == 2 and (~positive).sum() == 30
    assert positive[0, 0, 0] and positive[0, 2, 2] and not positive[1].any()
    assert torch.equal(box_targets[0, 0, 0], torch.tensor([.75, .75, .25, .25]))
    print(f"box_shape={tuple(box_targets.shape)}, objectness_shape={tuple(objectness.shape)}, "
          f"class_shape={tuple(class_ids.shape)}")
    print(f"positive_cells(row,col)={positive[0].nonzero().tolist()}, "
          f"first_target={box_targets[0, 0, 0].tolist()}, first_image_negative=14")
    print(f"batch positives={positive.sum().item()}, negatives={(~positive).sum().item()}, ignores=0")

    collision = {"boxes": torch.tensor([[4., 4., 20., 20.], [2., 2., 10., 10.]]),
                 "labels": torch.tensor([0, 0])}
    try:
        build([collision])
    except ValueError as error:
        print(f"capacity limit correctly raised: {error}")
    else:
        raise AssertionError("A one-slot grid must not silently lose the second object.")

    head = nn.Conv2d(4, 7, 1)
    features = torch.randn(2, 4, 4, 4)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    before = head.weight.detach().clone()
    head.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = head(features).permute(0, 2, 3, 1)
        prediction.retain_grad()
        box_loss = nn.functional.mse_loss(prediction[..., :4].sigmoid()[positive], box_targets[positive])
        object_loss = nn.functional.binary_cross_entropy_with_logits(prediction[..., 4], objectness)
        class_loss = nn.functional.cross_entropy(prediction[..., 5:][positive], class_ids[positive])
        loss = box_loss + object_loss + class_loss
        loss.backward()
        assert prediction.shape == (2, 4, 4, 7)
        assert prediction.grad[..., :4][~positive].abs().sum() == 0
        assert prediction.grad[..., 5:][~positive].abs().sum() == 0
        assert prediction.grad[..., 4][~positive].abs().sum() > 0
        assert torch.isfinite(head.weight.grad).all()
        optimizer.step()
        print(f"step={step}, box={box_loss.item():.4f}, obj={object_loss.item():.4f}, cls={class_loss.item():.4f}")
    assert not torch.equal(before, head.weight)
    print("negative positions supervise objectness only; artificial-feature head update verified")


if __name__ == "__main__":
    main()
