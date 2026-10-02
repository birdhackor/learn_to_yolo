"""跨圖按類別、分數排序的 all-points interpolated AP，GT 最多配對一次。"""

import torch
from .geometry import _boxes, box_iou


def _validate(entry, num_classes, prediction=False):
    boxes, labels = _boxes(entry["boxes"]), entry["labels"]
    if labels.dtype != torch.long or labels.ndim != 1 or len(labels) != len(boxes):
        raise ValueError("labels 必須是與 boxes 同長度的 Long[N]")
    if ((labels < 0) | (labels >= num_classes)).any():
        raise ValueError("class id 超出 num_classes")
    if (boxes[:, 2:] < boxes[:, :2]).any():
        raise ValueError("xyxy 順序錯誤")
    if prediction:
        scores = entry["scores"]
        if scores.ndim != 1 or len(scores) != len(boxes) or not torch.isfinite(scores).all():
            raise ValueError("scores 必須是與 boxes 同長度的有限值 [N]")
    return boxes.detach().cpu().double(), labels.detach().cpu()


def _interpolated_ap(true_positives, n_gt):
    tp, fp, recalls, precisions = 0, 0, [0.0], [0.0]
    for matched in true_positives:
        tp += int(matched)
        fp += int(not matched)
        recalls.append(tp / n_gt)
        precisions.append(tp / (tp + fp))
    recalls.append(1.0)
    precisions.append(0.0)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    return sum((recalls[i] - recalls[i - 1]) * precisions[i]
               for i in range(1, len(recalls)) if recalls[i] > recalls[i - 1])


@torch.no_grad()
def evaluate_ap(predictions, targets, num_classes=2, iou_threshold=.5):
    """AP 只平均有 GT 的類別；precision/recall 是所有輸入候選的 micro 指標。

    每個候選與同圖同類尚未使用的 GT 中最大 IoU 者配對，IoU>=門檻為 TP。
    無 GT 類別的 AP 是 None；全部無 GT 時 map=0、recall=0。
    此為單一 IoU 門檻 AP，並非 COCO AP@[.50:.95]。
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions 與 targets 的圖片數必須相同")
    if num_classes < 1 or not 0 <= iou_threshold <= 1:
        raise ValueError("num_classes>=1，iou_threshold 必須在 [0,1]")
    gt = [_validate(target, num_classes) for target in targets]
    pred = [_validate(entry, num_classes, prediction=True) for entry in predictions]
    ap_per_class, all_tp, all_predictions, all_gt = {}, 0, 0, 0
    for class_id in range(num_classes):
        class_gt = [boxes[labels == class_id] for boxes, labels in gt]
        used = [torch.zeros(len(boxes), dtype=torch.bool) for boxes in class_gt]
        n_gt = sum(len(boxes) for boxes in class_gt)
        candidates = []
        for image_id, ((boxes, labels), entry) in enumerate(zip(pred, predictions)):
            for index in torch.where(labels == class_id)[0].tolist():
                candidates.append((float(entry["scores"][index]), image_id, index, boxes[index]))
        # Python sort 穩定：同分按 image_id、原始候選順序。
        candidates.sort(key=lambda item: -item[0])
        matches = []
        for _, image_id, _, box in candidates:
            available = torch.where(~used[image_id])[0]
            matched = False
            if available.numel():
                ious = box_iou(box.unsqueeze(0), class_gt[image_id][available]).squeeze(0)
                overlap, position = ious.max(0)
                if float(overlap) >= iou_threshold:
                    used[image_id][available[position]] = True
                    matched = True
            matches.append(matched)
        all_tp += sum(matches)
        all_predictions += len(candidates)
        all_gt += n_gt
        ap_per_class[class_id] = _interpolated_ap(matches, n_gt) if n_gt else None
    valid_ap = [ap for ap in ap_per_class.values() if ap is not None]
    return {"ap_per_class": ap_per_class, "map": sum(valid_ap) / len(valid_ap) if valid_ap else 0.0,
            "precision": all_tp / all_predictions if all_predictions else 0.0,
            "recall": all_tp / all_gt if all_gt else 0.0}
