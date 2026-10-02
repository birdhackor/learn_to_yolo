"""明確的分項 loss；空圖片仍提供背景梯度。"""

import torch
from torch.nn import functional as F


def grid_loss(prediction, targets):
    """total=5*box+objectness+classification。

    box: 正格 sigmoid 座標的 MSE，平均正格×4 座標。
    objectness: 全格 BCEWithLogits，平均 B×S×S。
    classification: 正格 class logits 的 CE，平均正格數。
    無正格時 box/class 為與 prediction 相連的零，避免空 mean 的 NaN。
    """
    if prediction.ndim != 4 or prediction.shape[-1] < 6 or prediction.shape[0] < 1:
        raise ValueError("prediction 必須是非空 [B,S,S,5+C]，C>=1")
    if prediction.shape[1] != prediction.shape[2] or prediction.shape[1] < 1:
        raise ValueError("prediction 必須有非空正方形 grid")
    if not torch.isfinite(prediction).all():
        raise ValueError("prediction 含非有限值")
    positive = targets["positive"]
    if positive.shape != prediction.shape[:-1] or positive.dtype != torch.bool:
        raise ValueError("positive 必須是與 prediction grid 相符的 Bool[B,S,S]")
    if targets["box"].shape != (*positive.shape, 4):
        raise ValueError("box target shape 錯誤")
    if targets["objectness"].shape != positive.shape or targets["class_ids"].shape != positive.shape:
        raise ValueError("objectness/class_ids target shape 錯誤")
    if not torch.isfinite(targets["box"]).all() or not torch.isfinite(targets["objectness"]).all():
        raise ValueError("target 含非有限值")
    if positive.any():
        box = F.mse_loss(prediction[..., :4].sigmoid()[positive], targets["box"][positive])
        classification = F.cross_entropy(prediction[..., 5:][positive], targets["class_ids"][positive])
    else:
        box = (prediction[..., :4] * 0).sum()
        classification = (prediction[..., 5:] * 0).sum()
    objectness = F.binary_cross_entropy_with_logits(prediction[..., 4], targets["objectness"])
    total = 5 * box + objectness + classification
    return {"total": total, "box": box, "objectness": objectness, "classification": classification}
