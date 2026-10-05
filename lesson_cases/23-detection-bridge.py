"""獨立從零 SSL，再凍結 patch backbone，訓練每圖一物件的 cls+bbox head。"""
import json
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F

from miniyolo.figures import save_svg, use_svg_text
from miniyolo.self_distillation import DINOTrainer, extract_features, patch_feature_map
from miniyolo.vision_data import make_color_splits


class SingleObjectHead(nn.Module):
    """[B,32,4,4] -> hidden64 -> class logits2 + normalized cxcywh4。"""

    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.GELU())
        self.classifier = nn.Linear(64, 2)
        self.box = nn.Linear(64, 4)

    def forward(self, feature_map):
        hidden = self.shared(feature_map)
        raw = self.box(hidden).sigmoid()
        center, size = raw[:, :2], .05 + .7 * raw[:, 2:]
        boxes = torch.cat((center - size / 2, center + size / 2), -1).clamp(0, 1)
        return self.classifier(hidden), boxes  # xyxy normalized to 32px original image


def paired_iou(predictions, targets):
    """一張圖一個真值：按圖片配對 xyxy，無候選選擇或 AP 計算。"""
    intersection = (torch.minimum(predictions[:, 2:], targets[:, 2:]) -
                    torch.maximum(predictions[:, :2], targets[:, :2])).clamp_min(0).prod(-1)
    predicted_area = (predictions[:, 2:] - predictions[:, :2]).clamp_min(0).prod(-1)
    target_area = (targets[:, 2:] - targets[:, :2]).clamp_min(0).prod(-1)
    return intersection / (predicted_area + target_area - intersection).clamp_min(1e-9)


def main(ssl_steps=160, head_steps=200):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    data = make_color_splits()
    start = time.perf_counter()
    trainer = DINOTrainer(seed=7)
    ssl_history = trainer.train_steps(data["train"].images, ssl_steps)  # labels/bboxes excluded
    backbone = trainer.teacher.backbone.requires_grad_(False).eval()
    before = {name: value.clone() for name, value in backbone.state_dict().items()}
    feature_maps = {name: patch_feature_map(extract_features(backbone, split.images, kind="patches"))
                    for name, split in data.items()}
    # 標準化只估 train 每個 grid 位置/通道的統計量，val/test 不提供統計量。
    mean = feature_maps["train"].mean(0, keepdim=True)
    std = feature_maps["train"].std(0, keepdim=True, unbiased=False).clamp_min(.05)
    feature_maps = {name: (features - mean) / std for name, features in feature_maps.items()}
    torch.manual_seed(901)
    head = SingleObjectHead()
    optimizer = torch.optim.Adam(head.parameters(), lr=.003)
    generator = torch.Generator().manual_seed(902)
    initial_head = {name: tensor.clone() for name, tensor in head.state_dict().items()}
    history = []
    for step in range(head_steps):
        indices = torch.randint(len(data["train"]), (32,), generator=generator)
        logits, boxes = head(feature_maps["train"][indices])
        class_loss = F.cross_entropy(logits, data["train"].labels[indices])
        box_loss = F.smooth_l1_loss(boxes, data["train"].boxes[indices].float() / 32)
        value = class_loss + 10 * box_loss
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        gradients = [parameter.grad for parameter in head.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.square().sum()) for gradient in gradients) > 0
        optimizer.step()
        history.append({"step": step + 1, "loss": float(value.detach()),
                        "class_loss": float(class_loss.detach()), "box_loss": float(box_loss.detach())})
    unchanged = all(torch.equal(before[name], value) for name, value in backbone.state_dict().items())
    assert unchanged and all(parameter.grad is None for parameter in backbone.parameters())
    assert any(not torch.equal(initial_head[name], value) for name, value in head.state_dict().items())
    head.eval()
    metrics, predictions = {}, {}
    with torch.no_grad():
        for name, split in data.items():
            logits, boxes = head(feature_maps[name])
            predicted_labels = logits.argmax(-1)
            correct = predicted_labels == split.labels
            ious = paired_iou(boxes, split.boxes.float() / 32)
            metrics[name] = {"count": len(split), "class_correct": int(correct.sum()),
                             "class_accuracy": float(correct.float().mean()), "mean_iou": float(ious.mean()),
                             "iou_ge_0_5_count": int((ious >= .5).sum()),
                             "class_correct_and_iou_ge_0_5_count": int((correct & (ious >= .5)).sum())}
            predictions[name] = {"boxes": boxes * 32, "labels": predicted_labels, "ious": ious}
    examples = [{"test_index": index, "source_seed": data["test"].seed,
                 "truth_xyxy_pixel": data["test"].boxes[index].tolist(),
                 "predicted_xyxy_pixel": predictions["test"]["boxes"][index].tolist(),
                 "truth_class": data["test"].class_names[int(data["test"].labels[index])],
                 "predicted_class": data["test"].class_names[int(predictions["test"]["labels"][index])],
                 "iou": float(predictions["test"]["ious"][index])} for index in range(4)]
    result = {"ssl_steps": ssl_steps, "head_steps": head_steps, "device": "cpu",
              "split": {name: {"samples": len(split), "seed": split.seed} for name, split in data.items()},
              "ssl_first_loss": ssl_history[0]["loss"], "ssl_last_loss": ssl_history[-1]["loss"],
              "feature_shapes": {"patches": [128, 16, 32], "map": list(feature_maps["train"].shape)},
              "head": "Flatten512 -> Linear64/GELU -> class Linear2 + box Linear4; sigmoid center, size=.05+.7*sigmoid; clamp decoded xyxy to [0,1]",
              "head_learning_rate": .003, "head_batch_size": 32, "head_seed": 901,
              "loss": "cross_entropy(class) + 10*smooth_l1(normalized_xyxy)",
              "first_head_step": history[0], "last_head_step": history[-1], "head_history": history,
              "backbone_unchanged": unchanged, "backbone_has_no_grad": True,
              "head_weights_changed": True, "metrics": metrics, "examples": examples,
              "elapsed_seconds": time.perf_counter() - start,
              "evaluation": "one prediction paired with one truth per image; class argmax, paired IoU, joint class-correct and IoU>=.5; no AP/mAP",
              "limitation": "tiny frozen self-supervised ViT plus supervised single-object head; not DINO detector, not official DINOv2 backbone, no architecture ranking or natural-image transfer claim"}
    output = Path("artifacts/runs/dino")
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(), "feature_mean": mean,
                "feature_std": std, "class_names": data["train"].class_names,
                "backbone_config": trainer.backbone_config}, output / "23-detection-bridge.pt")
    (output / "23-detection-bridge.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    use_svg_text()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    figure, axes = plt.subplots(1, 4, figsize=(11, 3.5))
    for axis, example in zip(axes, examples):
        axis.imshow(data["test"].images[example["test_index"]].permute(1, 2, 0).numpy())
        for field, color, style in (("truth_xyxy_pixel", "#22c55e", "-"),
                                    ("predicted_xyxy_pixel", "#f59e0b", "--")):
            x1, y1, x2, y2 = example[field]
            axis.add_patch(Rectangle((x1 - .5, y1 - .5), x2 - x1, y2 - y1,
                                     fill=False, edgecolor=color, linestyle=style, linewidth=2))
        axis.set_title(f"test {example['test_index']}: {example['predicted_class']}\nIoU={example['iou']:.3f}")
        axis.axis("off")
    figure.suptitle("實際 test 圖：綠實線是真值，橘虛線是預測；每張圖一個框")
    figure.tight_layout()
    save_svg(figure, output / "23-detection-bridge.svg", "SSL backbone 加單物件偵測頭", "四張測試圖疊上真值與實際預測框")
    plt.close(figure)
    print(json.dumps({key: value for key, value in result.items() if key != "head_history"}, ensure_ascii=False, indent=2))
    print(f"actual boxes: {output / '23-detection-bridge.svg'}")


if __name__ == "__main__":
    main()
