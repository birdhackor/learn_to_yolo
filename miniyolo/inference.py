"""Grid 解碼、objectness×class score 與分別按類別做 NMS。"""

import torch
from .geometry import _image_hw, cxcywh_to_xyxy, nms


@torch.no_grad()
def decode_grid(prediction, image_size=64, score_threshold=.25, nms_iou=.5):
    """每格選 softmax 最大類別；score 不是 precision，也不是單獨 obj。"""
    if prediction.ndim != 4 or prediction.shape[-1] < 6:
        raise ValueError("prediction 必須為 [B,S,S,5+C]，C>=1")
    s = prediction.shape[1]
    if s < 1 or prediction.shape[2] != s or not torch.isfinite(prediction).all():
        raise ValueError("grid 必須非空且正方形，prediction 必須為有限值")
    if not 0 <= score_threshold <= 1 or not 0 <= nms_iou <= 1:
        raise ValueError("score_threshold 與 nms_iou 必須在 [0,1]")
    h, w = _image_hw(image_size)
    rows, cols = torch.meshgrid(torch.arange(s, device=prediction.device),
                                torch.arange(s, device=prediction.device), indexing="ij")
    grid = torch.stack((cols, rows), dim=-1)
    offsets = prediction[..., :4].sigmoid()
    centers = (offsets[..., :2] + grid) / s
    sizes = offsets[..., 2:4]
    encoded = torch.cat((centers, sizes), dim=-1) * prediction.new_tensor([w, h, w, h])
    boxes = cxcywh_to_xyxy(encoded.reshape(-1, 4)).reshape(*prediction.shape[:-1], 4)
    boxes[..., 0::2].clamp_(0, w)
    boxes[..., 1::2].clamp_(0, h)
    probabilities, labels = prediction[..., 5:].softmax(-1).max(-1)
    scores = prediction[..., 4].sigmoid() * probabilities
    results = []
    for batch_boxes, batch_scores, batch_labels in zip(boxes, scores, labels):
        keep = (batch_scores >= score_threshold) & (batch_boxes[..., 2:] > batch_boxes[..., :2]).all(-1)
        b, sc, lab = batch_boxes[keep], batch_scores[keep], batch_labels[keep]
        selected = []
        for class_id in lab.unique(sorted=True):
            indices = torch.where(lab == class_id)[0]
            selected.append(indices[nms(b[indices], sc[indices], nms_iou)])
        indices = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long, device=prediction.device)
        indices = indices[torch.argsort(sc[indices], descending=True, stable=True)]
        results.append({"boxes": b[indices], "scores": sc[indices], "labels": lab[indices]})
    return results
