"""可選 CPU/CUDA 的合成矩形訓練：python -m miniyolo.train --steps 160。

預設從零初始化；train/validation/test 以不同 seed 產生。CLI 不下載資料。
固定 step 實驗與長 epoch 實驗共用一條資料→監督→loss→解碼→評估管線。
"""

import argparse
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .data import ShapeDataset, collate
from .models import GridDetector
from .targets import build_targets
from .losses import grid_loss
from .inference import decode_grid
from .metrics import evaluate_ap


CLASS_NAMES = ["red rectangle", "blue rectangle"]


def _save_json(path, contents):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hardware(device):
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if not cpu and cpuinfo.is_file():
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), "unknown")
    return {"device": str(device), "cpu": cpu, "python": platform.python_version(),
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


@torch.no_grad()
def _evaluate(model, images, targets, config):
    model.eval()
    predictions = decode_grid(model(images), image_size=config["image_size"],
                              score_threshold=config["score_threshold"], nms_iou=config["nms_iou"])
    return evaluate_ap(predictions, targets, num_classes=2, iou_threshold=config["eval_iou"]), predictions


def train(config, output="artifacts/runs/grid-learning", report="artifacts/checks/grid-learning.json"):
    """執行固定順序、完整 batch 的學習；不足一個 batch 時循環索引補齊。"""
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("要求 CUDA，但此環境無可用 CUDA；CPU 驗證請用 --device cpu")
    run = Path(output)
    run.mkdir(parents=True, exist_ok=True)
    dataset = ShapeDataset(n=config["samples"], size=config["image_size"], seed=config["train_seed"])
    validation = ShapeDataset(n=config["validation_samples"], size=config["image_size"], seed=config["validation_seed"])
    test = ShapeDataset(n=config["test_samples"], size=config["image_size"], seed=config["test_seed"])
    # CPU 標註保留原始 pixel 契約；GPU 的訓練標註另搬到同一 device。
    train_images, train_targets = collate([dataset[i] for i in range(len(dataset))])
    val_images, val_targets = collate([validation[i] for i in range(len(validation))])
    test_images, test_targets = collate([test[i] for i in range(len(test))])
    train_images, val_device, test_device = train_images.to(device), val_images.to(device), test_images.to(device)
    train_targets = [{key: value.to(device) for key, value in target.items()} for target in train_targets]
    model = GridDetector(num_classes=2, grid_size=config["grid_size"], width=config["width"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    initial, _ = _evaluate(model, val_device, val_targets, config)
    print(json.dumps({"event": "initial_validation", **initial}))
    steps_per_epoch = math.ceil(config["samples"] / config["batch_size"])
    total_steps = config["steps"] if config["steps"] is not None else config["epochs"] * steps_per_epoch
    history, epoch_values = [], []
    start = time.perf_counter()
    for step in range(total_steps):
        indices = [(step * config["batch_size"] + i) % len(dataset) for i in range(config["batch_size"])]
        images = train_images[indices]
        targets = build_targets([train_targets[i] for i in indices], grid_size=config["grid_size"],
                                image_size=config["image_size"], num_classes=2)
        model.train()
        losses = grid_loss(model(images), targets)
        if not all(torch.isfinite(value) for value in losses.values()):
            raise RuntimeError(f"step {step + 1} loss 非有限值")
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
        values = {key: float(value.detach()) for key, value in losses.items()}
        history.append({"step": step + 1, **values})
        epoch_values.append(values)
        if (step + 1) % steps_per_epoch == 0 or step + 1 == total_steps:
            epoch = step // steps_per_epoch + 1
            summary = {key: sum(v[key] for v in epoch_values) / len(epoch_values) for key in values}
            print(json.dumps({"event": "train_epoch", "epoch": epoch, "steps": step + 1, **summary}))
            epoch_values = []
    # CUDA 的非同步運算先同步，計時才包含已排入的實際工作。
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    final_validation, predictions = _evaluate(model, val_device, val_targets, config)
    final_test, _ = _evaluate(model, test_device, test_targets, config)
    checkpoint = run / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "config": config, "class_names": CLASS_NAMES, "steps_completed": total_steps}, checkpoint)
    _save_json(run / "history.json", history)
    # 曲線只呈現這次實際 loss，不用人工假造訓練結果。
    cache = Path(__file__).resolve().parents[1] / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axis = plt.subplots(figsize=(7, 4))
    for key in ("total", "box", "objectness", "classification"):
        axis.plot([row["step"] for row in history], [row[key] for row in history], label=key)
    axis.set(xlabel="optimizer steps", ylabel="training loss", title="Synthetic grid MiniYOLO")
    axis.legend()
    fig.tight_layout()
    fig.savefig(run / "loss.png", dpi=140)
    plt.close(fig)
    from PIL import Image
    examples = []
    for i in range(min(4, len(validation))):
        image_path = run / f"validation-{i:02d}.png"
        Image.fromarray((val_images[i].permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(image_path)
        examples.append({"index": i, "image_path": str(image_path),
                         "target": {key: value.tolist() for key, value in val_targets[i].items()},
                         "prediction": {key: value.cpu().tolist() for key, value in predictions[i].items()}})
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(), "config": config,
        "steps_completed": total_steps, "hardware": _hardware(device),
        "training_seconds": elapsed, "checkpoint": str(checkpoint), "class_names": CLASS_NAMES,
        "history_path": str(run / "history.json"), "curve_path": str(run / "loss.png"),
        "initial_validation": initial, "final_validation": final_validation, "final_test": final_test,
        "final_train_loss": {key: history[-1][key] for key in ("total", "box", "objectness", "classification")},
        "metric_definitions": {
            "map": "mean all-points interpolated AP over classes having GT, at eval_iou only",
            "precision": "micro TP/(TP+FP) over candidates after score filtering and class-wise NMS",
            "recall": "micro TP/(TP+FN); each same-image same-class GT matched at most once",
            "score": "sigmoid(objectness) * max softmax(class logits)",
            "loss": "5*positive-cell sigmoid-coordinate MSE + all-cell BCEWithLogits + positive-cell class CE",
        },
        "limitations": "Early observation on synthetic colored rectangles; no real-image detection claim. Runtime describes this hardware/run only.",
        "validation_examples": examples,
    }
    _save_json(report, result)
    print(json.dumps({"event": "final_validation", **final_validation}))
    print(json.dumps({"event": "heldout_test", **final_test}))
    print(f"checkpoint={checkpoint} report={report}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--epochs", type=int, help="完整 epoch 數；預設 1")
    budget.add_argument("--steps", type=int, help="固定 optimizer step 預算，例如 160")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--validation-seed", type=int, default=700)
    parser.add_argument("--test-seed", type=int, default=7000)
    parser.add_argument("--score-threshold", type=float, default=.05)
    parser.add_argument("--nms-iou", type=float, default=.5)
    parser.add_argument("--eval-iou", type=float, default=.5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", default="artifacts/runs/grid-learning", help="checkpoint、曲線與範例圖目錄")
    parser.add_argument("--report", default="artifacts/checks/grid-learning.json")
    args = parser.parse_args()
    config = vars(args).copy()
    output, report = config.pop("output"), config.pop("report")
    config["epochs"] = args.epochs if args.epochs is not None else 1
    for key in ("epochs", "samples", "validation_samples", "test_samples", "batch_size", "width", "grid_size", "threads"):
        if config[key] < 1:
            parser.error(f"--{key.replace('_', '-')} 必須至少為 1")
    if config["steps"] is not None and config["steps"] < 1:
        parser.error("--steps 必須至少為 1")
    if config["grid_size"] != 4:
        parser.error("目前 ShapeDataset 的中心分配為 4×4；訓練 CLI 請用 --grid-size 4")
    if len({config["train_seed"], config["validation_seed"], config["test_seed"]}) != 3:
        parser.error("train/validation/test seed 必須互異")
    train(config, output, report)


if __name__ == "__main__":
    main()
