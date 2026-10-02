import torch
from miniyolo.targets import build_targets


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    scene = {"boxes": torch.tensor([[8., 12., 24., 28.], [40., 36., 56., 52.]]),
             "labels": torch.tensor([0, 1])}
    empty = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
    target = build_targets([scene, empty], grid_size=4, image_size=64, num_classes=2)
    assert target["box"].shape == (2, 4, 4, 4)
    assert target["positive"].sum().item() == 2
    assert torch.allclose(target["box"][0, 1, 1], torch.tensor([0., .25, .25, .25]))
    assert torch.allclose(target["box"][0, 2, 3], torch.tensor([0., .75, .25, .25]))
    assert target["class_ids"][0, 1, 1].item() == 0
    assert target["class_ids"][0, 2, 3].item() == 1
    assert not target["positive"][1].any()
    collision = {"boxes": torch.tensor([[8., 12., 24., 28.], [10., 14., 26., 30.]]),
                 "labels": torch.tensor([0, 0])}
    try:
        build_targets([collision], grid_size=4, image_size=64, num_classes=2)
    except ValueError:
        print("same-cell collision rejected")
    else:
        raise AssertionError("Two objects silently overwrote one slot")
    print("positive indices (b,y,x)", target["positive"].nonzero().tolist())
    print("red target", target["box"][0, 1, 1].tolist())
    print("blue target", target["box"][0, 2, 3].tolist())
    print("positive/negative counts", 2, int((~target["positive"]).sum()))


if __name__ == '__main__':
    main()
