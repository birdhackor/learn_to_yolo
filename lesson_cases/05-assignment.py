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
    # Read each cell's state from the targets: a positive cell has objectness 1, a negative
    # cell objectness 0 ("no object"); any other cell would need a third state such as ignore.
    negative = objectness == 0
    ignore = ~positive & ~negative
    assert positive.sum() == 2 and negative.sum() == 30 and not ignore.any()
    assert positive[0, 0, 0] and positive[0, 2, 2] and not positive[1].any()
    assert torch.equal(box_targets[0, 0, 0], torch.tensor([.75, .75, .25, .25]))
    print(f"box_shape={tuple(box_targets.shape)}, objectness_shape={tuple(objectness.shape)}, "
          f"class_shape={tuple(class_ids.shape)}")
    cells = positive[0].nonzero().tolist()  # [row, col] of each positive cell, row by row
    row, col = cells[0]
    print(f"positive_cells(row,col)={cells}, first_target={box_targets[0, row, col].tolist()}, "
          f"first_image_negative={negative[0].sum().item()}")
    print(f"batch positives={positive.sum().item()}, negatives={negative.sum().item()}, "
          f"ignores={ignore.sum().item()}")

    # Both boxes above have center x = y and width = height, so swapping row/col, the x/y
    # offsets or w/h in build() would go unnoticed. Here the first box has unequal offsets and
    # sizes, and the second sits at row 1, col 2 (a row/col swap would give row 2, col 1); it is
    # not the page exercise's box, so the printed result does not give away the exercise's answers.
    odd = {"boxes": torch.tensor([[2., 4., 22., 16.], [36., 12., 52., 28.]]),
           "labels": torch.tensor([0, 1])}
    odd_targets, _, odd_classes, odd_positive = build([odd])
    odd_cells = odd_positive[0].nonzero().tolist()
    odd_cell_targets = odd_targets[0][odd_positive[0]].tolist()  # same order as odd_cells
    odd_cell_classes = odd_classes[0][odd_positive[0]].tolist()
    assert odd_cells == [[0, 0], [1, 2]]
    assert odd_cell_targets == [[.75, .625, .3125, .1875], [.75, .25, .25, .25]]
    assert odd_cell_classes == [0, 1]
    print(f"asymmetric boxes: positive_cells(row,col)={odd_cells}, targets={odd_cell_targets}, "
          f"classes={odd_cell_classes}")

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
