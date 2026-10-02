"""One deterministic scene, variable annotation counts, no downloads."""
import torch
from miniyolo.data import ShapeDataset, collate


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    image = torch.zeros(3, 64, 64)
    image[0, 12:28, 8:24] = 1
    image[2, 36:52, 40:56] = 1
    target = {"boxes": torch.tensor([[8., 12., 24., 28.], [40., 36., 56., 52.]]),
              "labels": torch.tensor([0, 1], dtype=torch.long)}
    expected_pixels = torch.zeros_like(image)
    for box, label in zip(target["boxes"], target["labels"]):
        x1, y1, x2, y2 = box.to(torch.long).tolist()
        channel = {0: 0, 1: 2}[label.item()]
        expected_pixels[channel, y1:y2, x1:x2] = 1
    assert torch.equal(image, expected_pixels), "Annotation and colored pixels disagree"
    empty = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
    images, targets = collate([(image, target), (torch.zeros_like(image), empty)])
    assert images.shape == (2, 3, 64, 64)
    assert targets[1]["boxes"].shape == (0, 4)
    assert images[0, 0, 12:28, 8:24].sum().item() == 256
    assert images[0, 2, 36:52, 40:56].sum().item() == 256
    generated = ShapeDataset(n=8, size=64, seed=7)
    for im, ann in generated:
        assert im.dtype == torch.float32 and im.shape == (3, 64, 64)
        assert ann["boxes"].shape == (len(ann["labels"]), 4)
        boxes = ann["boxes"]
        assert bool(((boxes >= 0) & (boxes <= 64)).all())
        assert bool((boxes[:, 2:] > boxes[:, :2]).all())
    print("batch", tuple(images.shape), "counts", [len(t["boxes"]) for t in targets])
    print("red/blue channel sums", images[0, 0].sum().item(), images[0, 2].sum().item())
    print("dataset contract checked: 8 images")


if __name__ == '__main__':
    main()
