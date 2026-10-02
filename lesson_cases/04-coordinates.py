"""Box/image transforms with recorded actual resize factors and exact round trip."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/miniyolo-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/miniyolo-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch


def letterbox(image, boxes, size=64):
    if not boxes.is_floating_point():
        boxes = boxes.to(torch.float32)
    _, height, width = image.shape
    ideal_scale = min(size / width, size / height)
    new_w, new_h = round(width * ideal_scale), round(height * ideal_scale)
    left, top = (size - new_w) // 2, (size - new_h) // 2
    resized = torch.nn.functional.interpolate(image[None], size=(new_h, new_w),
                                               mode="bilinear", align_corners=False)[0]
    canvas = image.new_zeros(3, size, size)
    canvas[:, top:top + new_h, left:left + new_w] = resized
    scales = boxes.new_tensor([new_w / width, new_h / height] * 2)
    padding = boxes.new_tensor([left, top] * 2)
    metadata = {"scale_xyxy": scales, "padding_xyxy": padding,
                "original_hw": (height, width), "resized_hw": (new_h, new_w)}
    return canvas, boxes * scales + padding, metadata


def undo(boxes, metadata):
    return (boxes - metadata["padding_xyxy"]) / metadata["scale_xyxy"]


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    image = torch.zeros(3, 40, 80)
    image[0, 5:25, 10:50] = 1
    boxes = torch.tensor([[10., 5., 50., 25.]])
    canvas, transformed, metadata = letterbox(image, boxes)
    restored = undo(transformed, metadata)
    assert torch.allclose(transformed, torch.tensor([[8., 20., 40., 36.]]))
    assert torch.allclose(restored, boxes, atol=1e-5)
    _, integer_transformed, integer_metadata = letterbox(image, boxes.to(torch.long))
    integer_restored = undo(integer_transformed, integer_metadata)
    assert torch.isfinite(integer_restored).all()
    assert torch.allclose(integer_restored, boxes, atol=1e-5)
    print("integer annotation converted to float; finite exact round trip passed")
    stretch = torch.nn.functional.interpolate(image[None], size=(64, 64),
                                               mode="bilinear", align_corners=False)[0]
    stretch_boxes = boxes * torch.tensor([.8, 1.6, .8, 1.6])
    assert torch.allclose(stretch_boxes, torch.tensor([[8., 8., 40., 40.]]))
    print(f"original_hw={metadata['original_hw']}, resized_hw={metadata['resized_hw']}")
    print(f"original={boxes.tolist()}, letterbox={transformed.tolist()}, restored={restored.tolist()}")
    print(f"stretch={stretch_boxes.tolist()}, padding={metadata['padding_xyxy'].tolist()}")
    odd_image, odd_boxes = torch.zeros(3, 37, 83), torch.tensor([[7., 3., 70., 31.]])
    _, odd_transformed, odd_metadata = letterbox(odd_image, odd_boxes)
    assert torch.allclose(undo(odd_transformed, odd_metadata), odd_boxes, atol=1e-5)
    _, empty_boxes, _ = letterbox(image, torch.empty(0, 4))
    assert empty_boxes.shape == (0, 4)
    print(f"odd rounding: resized_hw={odd_metadata['resized_hw']}, scales={odd_metadata['scale_xyxy'].tolist()}")
    print("odd-size and empty-box round trips passed; geometry only, no training required")
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)
    for ax, current, box, name in zip(axes, (image, stretch, canvas),
                                     (boxes[0], stretch_boxes[0], transformed[0]),
                                     ("Original 80x40", "Stretch 64x64", "Letterbox 64x64")):
        height, width = current.shape[1:]
        ax.imshow(current.permute(1, 2, 0).numpy(), extent=(0, width, height, 0))
        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="lime"))
        ax.set_title(name)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
    path = Path("artifacts/04-coordinates.png")
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"transform_panel={path}")


if __name__ == "__main__":
    main()
