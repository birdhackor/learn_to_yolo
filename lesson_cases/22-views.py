"""同一張圖的 global/local views；bbox 只用來看 crop，從不餵給 SSL。"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from miniyolo.figures import save_svg, use_svg_text
from miniyolo.self_distillation import random_view
from miniyolo.vision_data import ColorRectangles


def main():
    torch.set_num_threads(2)
    data = ColorRectangles(samples=1, seed=101)
    image, bbox = data.images[:1], data.boxes[0]
    generator = torch.Generator().manual_seed(22)
    global_a, crop_a = random_view(image, generator)
    global_b, crop_b = random_view(image, generator)
    local, crop_local = random_view(image, generator, scale=(.15, .25))
    repeated_generator = torch.Generator().manual_seed(22)
    repeated = [random_view(image, repeated_generator, scale=scale)[0]
                for scale in ((.65, 1.), (.65, 1.), (.15, .25))]
    assert all(torch.equal(a, b) for a, b in zip((global_a, global_b, local), repeated))
    # 明確的反例：從四角選一個完全沒碰到物件的 8×8 crop。
    candidates = torch.tensor([[0, 0, 8, 8], [24, 0, 32, 8], [0, 24, 8, 32], [24, 24, 32, 32]])
    overlaps = (torch.minimum(candidates[:, 2:], bbox[2:]) -
                torch.maximum(candidates[:, :2], bbox[:2])).clamp_min(0).prod(-1)
    empty_crop = candidates[(overlaps == 0).nonzero()[0, 0]]
    x1, y1, x2, y2 = empty_crop.tolist()
    empty = F.interpolate(image[:, :, y1:y2, x1:x2], (32, 32), mode="bilinear", align_corners=False)
    result = {"source_seed": 101, "source_sample_index": 0, "source_samples": 1,
              "view_generator_seed": 22, "same_rng_reproduces_views": True,
              "input_shape": list(image.shape), "source_bbox_xyxy_pixel": bbox.tolist(),
              "global_a_crop_xyxy_pixel": crop_a[0].tolist(), "global_b_crop_xyxy_pixel": crop_b[0].tolist(),
              "local_crop_xyxy_pixel": crop_local[0].tolist(), "empty_crop_xyxy_pixel": empty_crop.tolist(),
              "empty_crop_object_intersection_pixel2": 0, "view_shape": list(global_a.shape),
              "ssl_inputs": "images only; labels/bboxes excluded",
              "role": "official DINO: teacher gets global; student gets global+local; later tiny trainer uses only 2 globals",
              "limitation": "same source does not guarantee a crop contains the object; views are not labeled detection examples"}
    output = Path("artifacts/runs/dino")
    output.mkdir(parents=True, exist_ok=True)
    use_svg_text()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    figure, axes = plt.subplots(1, 5, figsize=(13, 3.7))
    panels = [(image, "原圖：bbox 只供說明", None), (global_a, "global A", crop_a[0]),
              (global_b, "global B", crop_b[0]), (local, "local 示範", crop_local[0]),
              (empty, "裁走物件的反例", empty_crop)]
    for axis, (view, name, crop) in zip(axes, panels):
        axis.imshow(view[0].permute(1, 2, 0).numpy(), vmin=0, vmax=1)
        axis.set_title(name)
        axis.axis("off")
        if crop is not None:
            axis.text(.5, -.05, f"原圖 crop={crop.tolist()}", transform=axis.transAxes, ha="center", fontsize=9)
    axes[0].add_patch(Rectangle((int(bbox[0]) - .5, int(bbox[1]) - .5),
                               int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]),
                               edgecolor="white", fill=False, linewidth=2))
    figure.suptitle("同一張圖片可形成不同 view；每個 crop 都 resize 回 32×32")
    figure.tight_layout()
    save_svg(figure, output / "22-views.svg", "實際 global 與 local views", "原圖、global A、global B、local 與裁走物件的 crop")
    plt.close(figure)
    (output / "22-views.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"actual views: {output / '22-views.svg'}")


if __name__ == "__main__":
    main()
