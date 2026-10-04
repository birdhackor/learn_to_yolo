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
    # The box must still frame the object: the red pixels on the canvas ([y, x] indexing),
    # written as a half-open box (last pixel index + 1), must equal the transformed box.
    # Exact equality works here because every edge of this box lands on a whole pixel.
    ys, xs = torch.where(canvas[0] > .5)
    red_pixel_box = [xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1]
    assert red_pixel_box == transformed[0].tolist()
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
    print(f"red_pixel_box={red_pixel_box}, equal to the letterbox box")
    print(f"stretch={stretch_boxes.tolist()}, padding={metadata['padding_xyxy'].tolist()}")
    odd_image, odd_boxes = torch.zeros(3, 37, 83), torch.tensor([[7., 3., 70., 31.]])
    _, odd_transformed, odd_metadata = letterbox(odd_image, odd_boxes)
    odd_restored = undo(odd_transformed, odd_metadata)
    assert torch.allclose(odd_restored, odd_boxes, atol=1e-5)
    _, empty_boxes, empty_metadata = letterbox(image, torch.empty(0, 4))
    assert empty_boxes.shape == (0, 4)
    empty_restored = undo(empty_boxes, empty_metadata)
    assert empty_restored.shape == (0, 4) and torch.isfinite(empty_restored).all()
    print(f"odd rounding: resized_hw={odd_metadata['resized_hw']}, scales={odd_metadata['scale_xyxy'].tolist()}")
    print(f"odd original={odd_boxes.tolist()}, restored={odd_restored.tolist()}")
    print("odd-size and empty-box round trips passed; geometry only, no training required")
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)
    for ax, current, box, name in zip(axes, (image, stretch, canvas),
                                     (boxes[0], stretch_boxes[0], transformed[0]),
                                     ("Original", "Stretch", "Letterbox")):
        height, width = current.shape[1:]
        ax.imshow(current.permute(1, 2, 0).numpy(), extent=(0, width, height, 0))
        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="lime"))
        ax.set_title(f"{name} W{width}×H{height}")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
    # Display only: the padding is 0 (black), like the image background, so outline the
    # resized content on the letterbox panel; the rest of that canvas is padding.
    left, top = metadata["padding_xyxy"][:2].tolist()
    new_h, new_w = metadata["resized_hw"]
    axes[2].add_patch(Rectangle((left, top), new_w, new_h, fill=False,
                                edgecolor="white", linestyle="--"))
    axes[2].set_xlabel(f"dashed box: resized original W{new_w}×H{new_h}\n"
                       "outside the dashed box: padding")
    path = Path("artifacts/04-coordinates.png")
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"transform_panel={path}")


if __name__ == "__main__":
    main()
