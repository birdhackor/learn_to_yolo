"""把變動長度 pixel 標註分配到中心所在 cell；不默默丟掉物件。"""

import torch
from .geometry import _boxes, _image_hw, xyxy_to_cxcywh


def build_targets(targets, grid_size=4, image_size=64, num_classes=2):
    if grid_size < 1 or num_classes < 1:
        raise ValueError("grid_size 與 num_classes 必須為正整數")
    h, w = _image_hw(image_size)
    batch = len(targets)
    device = targets[0]["boxes"].device if batch else torch.device("cpu")
    shape = (batch, grid_size, grid_size)
    output = {
        "box": torch.zeros((*shape, 4), dtype=torch.float32, device=device),
        "objectness": torch.zeros(shape, device=device),
        "class_ids": torch.zeros(shape, dtype=torch.long, device=device),
        "positive": torch.zeros(shape, dtype=torch.bool, device=device),
    }
    for b, target in enumerate(targets):
        boxes, labels = _boxes(target["boxes"]), target["labels"]
        if labels.dtype != torch.long or labels.ndim != 1 or len(labels) != len(boxes):
            raise ValueError("labels 必須是與 boxes 同長度的 Long[N]")
        if boxes.device != device or labels.device != device:
            raise ValueError("所有 boxes 與 labels 必須在相同 device")
        if ((labels < 0) | (labels >= num_classes)).any():
            raise ValueError("class id 超出 num_classes")
        if ((boxes[:, 2:] <= boxes[:, :2]).any()
                or (boxes < 0).any()
                or (boxes > boxes.new_tensor([w, h, w, h])).any()):
            raise ValueError("標註框必須有正面積，且位於 image_size 範圍內")
        normalized = xyxy_to_cxcywh(boxes) / boxes.new_tensor([w, h, w, h])
        for obj, label in zip(normalized, labels):
            grid_xy = obj[:2] * grid_size
            col, row = grid_xy.floor().long().tolist()
            if output["positive"][b, row, col]:
                raise ValueError(f"same-cell collision: image {b}, cell ({row},{col})")
            output["box"][b, row, col] = torch.cat((grid_xy - grid_xy.floor(), obj[2:]))
            output["objectness"][b, row, col] = 1
            output["class_ids"][b, row, col] = label
            output["positive"][b, row, col] = True
    return output
