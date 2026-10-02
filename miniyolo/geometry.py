"""半開區間 xyxy 幾何；面積不加 1，空框保持 [0,4]。"""

import torch
from torch.nn import functional as F


def _boxes(boxes):
    if not isinstance(boxes, torch.Tensor) or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes 必須是 [N,4] tensor")
    boxes = boxes if boxes.is_floating_point() else boxes.float()
    if not torch.isfinite(boxes).all():
        raise ValueError("boxes 含非有限值")
    return boxes


def _image_hw(size):
    h, w = (size, size) if isinstance(size, (int, float)) else size
    if h <= 0 or w <= 0:
        raise ValueError("image_size 必須為正數或 (height,width)")
    return h, w


def xyxy_to_cxcywh(boxes):
    boxes = _boxes(boxes)
    return torch.cat(((boxes[:, :2] + boxes[:, 2:]) / 2,
                      boxes[:, 2:] - boxes[:, :2]), dim=-1)


def cxcywh_to_xyxy(boxes):
    boxes = _boxes(boxes)
    half = boxes[:, 2:] / 2
    return torch.cat((boxes[:, :2] - half, boxes[:, :2] + half), dim=-1)


def box_iou(boxes1, boxes2):
    """[N,M] IoU；退化框的面積為零，union=0 時回傳零。"""
    boxes1, boxes2 = _boxes(boxes1), _boxes(boxes2)
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(-1)
    union = area1[:, None] + area2[None, :] - intersection
    return torch.where(union > 0, intersection / union.clamp(min=torch.finfo(union.dtype).tiny), 0)


def nms(boxes, scores, iou_threshold=.5):
    """保留高分候選；只抑制 IoU > 門檻，同分按輸入順序決定。"""
    boxes = _boxes(boxes)
    if scores.ndim != 1 or len(scores) != len(boxes) or not torch.isfinite(scores).all():
        raise ValueError("scores 必須是與 boxes 同長度的有限值 [N]")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold 必須在 [0,1]")
    order = torch.argsort(scores, descending=True, stable=True)
    kept = []
    while order.numel():
        current = order[0]
        kept.append(current)
        remaining = order[1:]
        overlaps = box_iou(boxes[current].unsqueeze(0), boxes[remaining]).squeeze(0)
        order = remaining[overlaps <= iou_threshold]
    return torch.stack(kept).long() if kept else torch.empty(0, dtype=torch.long, device=boxes.device)


def letterbox(image, boxes, size=64):
    """CHW tensor 等比縮放後補黑邊；metadata 記錄整數取整後的實際比例。"""
    if image.ndim != 3 or min(image.shape) < 1 or not image.is_floating_point():
        raise ValueError("image 必須是非空浮點 CHW tensor")
    if not torch.isfinite(image).all():
        raise ValueError("image 含非有限值")
    boxes = _boxes(boxes)
    h, w = image.shape[-2:]
    out_h, out_w = map(int, _image_hw(size))
    scale = min(out_h / h, out_w / w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (out_h - new_h) // 2, (out_w - new_w) // 2
    resized = F.interpolate(image.unsqueeze(0), size=(new_h, new_w),
                            mode="bilinear", align_corners=False).squeeze(0)
    output = F.pad(resized, (left, out_w - new_w - left, top, out_h - new_h - top))
    scale_xy = (new_w / w, new_h / h)
    transformed = boxes * boxes.new_tensor([*scale_xy, *scale_xy])
    transformed += boxes.new_tensor([left, top, left, top])
    metadata = {"original_size": (h, w), "scale": scale, "scale_xy": scale_xy,
                "padding": (left, top), "resized_size": (new_h, new_w)}
    return output, transformed, metadata


def undo_letterbox(boxes, metadata):
    """回到原圖 pixel xyxy，並裁切補邊上的候選到原圖範圍。"""
    boxes = _boxes(boxes)
    h, w = metadata["original_size"]
    left, top = metadata["padding"]
    sx, sy = metadata.get("scale_xy", (metadata["scale"], metadata["scale"]))
    if sx <= 0 or sy <= 0:
        raise ValueError("letterbox scale 必須為正")
    restored = (boxes - boxes.new_tensor([left, top, left, top]))
    restored = restored / boxes.new_tensor([sx, sy, sx, sy])
    return torch.stack((restored[:, 0].clamp(0, w), restored[:, 1].clamp(0, h),
                        restored[:, 2].clamp(0, w), restored[:, 3].clamp(0, h)), dim=-1)
