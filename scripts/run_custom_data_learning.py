"""CPU-only JSON + PNG -> learning -> held-out detection -> checkpoint reload.

Example fixture (synthetic rectangles, not frames from a real video)::

    .venv-model/bin/python scripts/run_custom_data_learning.py --fixture

Own annotations use the existing JsonDetectionDataset pixel-xyxy schema::

    .venv-model/bin/python scripts/run_custom_data_learning.py \
        --annotations /path/annotations.json --root /path/images --steps 160

A follow-up keeps a fixture diagnostic's train/validation images and settings and draws a fresh test.
It regenerates the diagnostic's fixture from its seeds and checks it against that report's hashes::

    .venv-model/bin/python scripts/run_custom_data_learning.py --fixture --steps 1600 \
        --fixture-test-seed 7001 --prior-diagnostic artifacts/checks/curriculum/custom-data-160-step.json

The architecture, optimizer, seed, and evaluation thresholds stay fixed. A low
held-out AP is a result to inspect; this script never searches for a better seed.
"""

import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from xml.sax.saxutils import escape

# Prevent the checkpoint RNG helper from initializing CUDA in this CPU exercise.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from PIL import Image, __version__ as pillow_version
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from miniyolo.checkpoint import capture_rng, load_checkpoint, save_checkpoint
from miniyolo.custom_data import JsonDetectionDataset
from miniyolo.data import collate
from miniyolo.figures import FONT_STACK
from miniyolo.inference import decode_grid
from miniyolo.geometry import box_iou, letterbox, undo_letterbox
from miniyolo.losses import grid_loss
from miniyolo.metrics import evaluate_ap
from miniyolo.models import GridDetector
from miniyolo.targets import build_targets
from miniyolo.train import optimizer_step
from miniyolo.provenance import machine_info, repo_dependencies
from scripts.detect_image import detect_image

SPLITS = ("train", "validation", "test")
# Fixture background noise takes values 0..12 in every channel; every painted color has a channel of 13 or more.
FIXTURE_NOISE = 13
# Short Traditional Chinese names for the fixture classes in the figure; other classes keep their own.
FIGURE_NAMES = {"red_rectangle": "紅", "blue_rectangle": "藍", "yellow_rectangle": "黃"}


def json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(value), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_path(path):
    """Repository-relative path inside the checkout, so records carry no machine-specific prefix."""
    path = Path(path).resolve()
    return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)


def repository_commit():
    """Read provenance without invoking Git or changing the checkout."""
    git_dir = REPO / ".git"
    if not git_dir.is_dir():
        return None
    head = (git_dir / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    reference = head.removeprefix("ref: ")
    loose = git_dir / reference
    if loose.is_file():
        return loose.read_text().strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for row in packed.read_text().splitlines():
            if row.endswith(" " + reference):
                return row.split()[0]
    return None


def create_fixture(folder, test_seed=7000):
    """Independent seeded synthetic sources; all originals are non-square."""
    classes = ["red_rectangle", "blue_rectangle", "yellow_rectangle"]
    colors = [(235, 35, 30), (30, 55, 235), (235, 215, 30)]
    records = []
    for split, count, seed in (("train", 24, 7), ("validation", 12, 700), ("test", 12, test_seed)):
        rng = np.random.default_rng(seed)
        positive_index = 0
        for index in range(count):
            width, height = (120, 80) if index % 2 == 0 else (80, 120)
            pixels = rng.integers(0, FIXTURE_NOISE, (height, width, 3), dtype=np.uint8)
            boxes, labels = [], []
            empty_every = 8 if split == "train" else 4
            if index % empty_every:
                label = positive_index % len(classes)
                positive_index += 1
                box_w = int(rng.integers(20, 35))
                box_h = int(rng.integers(18, 31))
                x1 = int(rng.integers(5, width - box_w - 4))
                y1 = int(rng.integers(5, height - box_h - 4))
                pixels[y1:y1 + box_h, x1:x1 + box_w] = colors[label]
                boxes = [[x1, y1, x1 + box_w, y1 + box_h]]
                labels = [label]
            relative = f"{split}/scene-{index:03d}.png"
            destination = folder / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(destination)
            records.append({"path": relative, "width": width, "height": height,
                            "source_id": f"synthetic-source-{split}-{index // 4}" +
                                         (f"-seed-{test_seed}" if split == "test" and test_seed != 7000 else ""),
                            "split": split, "boxes": boxes, "labels": labels})
    annotation_path = folder / "annotations.json"
    write_json(annotation_path, {"classes": classes, "images": records,
                                "provenance": "Seeded synthetic rectangles; not actual video frames.",
                                "split_seeds": {"train": 7, "validation": 700, "test": test_seed}})
    return annotation_path


def fixture_foreground_boxes(path):
    """Pixel bounds of the painted fixture rectangle (empty list for a background-only image)."""
    with Image.open(path) as image:
        painted = np.asarray(image.convert("RGB")).max(axis=2) >= FIXTURE_NOISE
    ys, xs = np.nonzero(painted)
    return [[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]] if len(xs) else []


def rgb_sha256(path):
    with Image.open(path) as image:
        # Include dimensions: identical bytes interpreted with a different
        # width are not identical images. Check decoded RGB as well as PNG.
        rgb = image.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def image_inventory(records, root):
    """PNG and decoded-RGB SHA-256 of every image, grouped by split in manifest order."""
    return [{"path": r["path"], "split": split, "sha256": sha256(Path(root) / r["path"]),
             "rgb_sha256": rgb_sha256(Path(root) / r["path"])}
            for split in SPLITS for r in records if r["split"] == split]


def rgb_inventory_sha256(inventory):
    """Data identity that survives a different PNG encoder build: paths, splits and decoded pixels."""
    rows = [{key: row[key] for key in ("path", "split", "rgb_sha256")} for row in inventory]
    return hashlib.sha256(json_text(rows).encode()).hexdigest()


def audit_data(datasets):
    sources, counts = {}, {}
    for split, dataset in datasets.items():
        empty = sum(not record["boxes"] for record in dataset.records)
        class_counts = {name: 0 for name in dataset.classes}
        for record in dataset.records:
            sources.setdefault(record["source_id"], set()).add(split)
            for label in record["labels"]:
                class_counts[dataset.classes[label]] += 1
        counts[split] = {"images": len(dataset), "empty_images": empty,
                         "objects": sum(class_counts.values()), "objects_per_class": class_counts,
                         "source_ids": sorted({r["source_id"] for r in dataset.records}),
                         "non_square_images": sum(r["width"] != r["height"] for r in dataset.records)}
    # Fingerprint both the annotation manifest and every actual PNG, so editing
    # pixels without changing boxes changes the dataset identity.
    inventory = image_inventory([r for dataset in datasets.values() for r in dataset.records],
                                datasets["train"].root)
    file_hashes, pixel_hashes = {}, {}
    for row in inventory:
        file_hashes.setdefault(row["sha256"], set()).add(row["split"])
        pixel_hashes.setdefault(row["rgb_sha256"], set()).add(row["split"])
    leaks = sum(len(splits) > 1 for splits in sources.values())
    file_duplicates = sum(len(splits) > 1 for splits in file_hashes.values())
    pixel_duplicates = sum(len(splits) > 1 for splits in pixel_hashes.values())
    if leaks or file_duplicates or pixel_duplicates:
        raise ValueError("Cross-split source or exact image duplication; split original sources before training")
    return counts, inventory, {"source_leakage_groups": leaks,
                              "cross_split_png_duplicates": file_duplicates,
                              "cross_split_decoded_rgb_duplicates": pixel_duplicates}


@torch.no_grad()
def full_loss(model, images, targets):
    model.eval()
    return {key: float(value) for key, value in grid_loss(model(images), targets).items()}


@torch.no_grad()
def predict(model, images, config):
    model.eval()
    raw = model(images)
    return raw, decode_grid(raw, image_size=config["image_size"],
                            score_threshold=config["score_threshold"], nms_iou=config["nms_iou"])


def match_image(prediction, target, iou_threshold):
    """evaluate_ap's matching inside one image: by descending score, each prediction takes its best
    still-unmatched same-class GT and is a TP when that IoU >= iou_threshold; each GT matches once."""
    labels, scores, gt_labels = prediction["labels"].tolist(), prediction["scores"].tolist(), target["labels"].tolist()
    ious = box_iou(prediction["boxes"].double(), target["boxes"].double())
    matched, rows = [False] * len(gt_labels), [None] * len(labels)
    for i in sorted(range(len(labels)), key=lambda i: -scores[i]):
        free = [j for j, label in enumerate(gt_labels) if label == labels[i] and not matched[j]]
        best = max(free, key=lambda j: float(ious[i, j]), default=None)
        iou = None if best is None else float(ious[i, best])
        hit = iou is not None and iou >= iou_threshold
        if hit:
            matched[best] = True
        rows[i] = {"label": labels[i], "score": scores[i], "match": "TP" if hit else "FP",
                   "iou_same_class_gt": iou, "iou_any_gt": float(ious[i].max()) if gt_labels else None}
    return {"predictions": rows, "missed_gt_labels": [label for label, hit in zip(gt_labels, matched) if not hit]}


def evaluate(predictions, targets, classes, config):
    result = evaluate_ap(predictions, targets, num_classes=len(classes), iou_threshold=config["eval_iou"])
    result["ap50_per_class_name"] = {name: result["ap_per_class"][i] for i, name in enumerate(classes)}
    result["images"] = len(targets)
    result["ground_truth_objects"] = sum(len(target["labels"]) for target in targets)
    result["decoded_predictions"] = sum(len(prediction["labels"]) for prediction in predictions)
    true_positives = sum(row["match"] == "TP" for prediction, target in zip(predictions, targets)
                         for row in match_image(prediction, target, config["eval_iou"])["predictions"])
    # The per-image TP/FP shown in the figure must reproduce evaluate_ap's own micro counts.
    if not (round(result["precision"] * result["decoded_predictions"]) == true_positives
            == round(result["recall"] * result["ground_truth_objects"])):
        raise RuntimeError("Per-image TP/FP matching disagrees with evaluate_ap")
    result["true_positives"] = true_positives
    result["false_positives"] = result["decoded_predictions"] - true_positives
    result["false_negatives"] = result["ground_truth_objects"] - true_positives
    return result


def run_outcome(steps, initial_loss, final_loss, metrics):
    """One sentence from the recorded numbers; 'fitted' means train mAP50, precision and recall are all 1."""
    train, validation = metrics["train"], metrics["validation"]
    shown = lambda value: "n/a" if value is None else f"{value:.5f}"
    fitted = train["map"] == train["precision"] == train["recall"] == 1
    return (f"{steps} fixed steps: full-train loss {initial_loss['total']:.5f} -> {final_loss['total']:.5f}; "
            f"train mAP50 {shown(train['map'])}, precision {train['precision']:.5f}, recall {train['recall']:.5f} "
            f"(training set {'fitted' if fitted else 'not yet fitted'}); validation mAP50 {shown(validation['map'])}. "
            "No seed, architecture, loss, optimizer or threshold tuning was applied.")


def training_box_diagnosis(model, images, targets, encoded, image_size, foreground_matches):
    """Box-target contract on the full training set after the last update (model in eval mode).

    grid_loss fits box outputs only on positive cells, so a correct pipeline has one positive cell
    per object, nonzero w/h targets and exactly zero box-loss gradient on every negative cell.
    """
    model.eval()
    with torch.enable_grad():
        raw = model(images).detach().requires_grad_(True)
        grid_loss(raw, encoded)["box"].backward()
    positive = encoded["positive"]
    target, logits = encoded["box"][positive], raw.detach()[..., :4][positive]
    predicted, gradient = logits.sigmoid(), raw.grad[..., :4]
    objects = sum(len(entry["labels"]) for entry in targets)
    diagnosis = {"scope": "Full training set after the last update, model in eval mode. xywh are grid_loss box "
                          "targets and sigmoid outputs (cell offsets, size / image); gradients are of the box term only",
                 "objects": objects, "positive_cells": int(positive.sum()),
                 "train_png_foreground_boxes_matching_json": foreground_matches}
    reductions = {"min": lambda values: values.amin(0), "max": lambda values: values.amax(0),
                  "mean": lambda values: values.mean(0)}
    for name, values in (("positive_target", target), ("positive_prediction", predicted), ("positive_raw_logits", logits)):
        for kind, reduce in reductions.items():
            diagnosis[f"{name}_{kind}_xywh"] = reduce(values).tolist() if len(values) else None
    diagnosis.update({
        "positive_prediction_min_wh_pixels": (predicted[:, 2:].amin(0) * image_size).tolist() if len(target) else None,
        "coordinate_mse_xywh": (predicted - target).square().mean(0).tolist() if len(target) else None,
        "box_loss_logit_abs_grad_mean_xywh": gradient[positive].abs().mean(0).tolist() if len(target) else None,
        "negative_cell_box_logit_abs_gradient_sum": float(gradient[~positive].abs().sum())})
    checks = {"one_positive_cell_per_object": diagnosis["positive_cells"] == objects,
              "positive_target_wh_nonzero": bool((target[:, 2:] > 0).all()),
              "negative_cells_zero_box_gradient": diagnosis["negative_cell_box_logit_abs_gradient_sum"] == 0}
    if foreground_matches is not None:
        checks["train_png_foreground_boxes_match_json"] = foreground_matches == len(targets)
    if not all(checks.values()):
        raise RuntimeError(f"Training box-target contract failed: {checks}")
    diagnosis["checks"] = checks
    return diagnosis


def exact_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            exact_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            exact_tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def tensor_rows(entries):
    return [{key: value.tolist() for key, value in entry.items()} for entry in entries]


def check_original_image(checkpoint, dataset, index, prediction, output, config):
    record = dataset.records[index]
    source = dataset.root / record["path"]
    destination = output / f"validation-{index:03d}-original-inference.png"
    # Explicit evaluation thresholds: detect_image's default is the higher display threshold.
    cli = detect_image(source, checkpoint, destination, score_threshold=config["score_threshold"],
                       nms_iou=config["nms_iou"])
    width, height = record["width"], record["height"]
    _, _, metadata = letterbox(torch.zeros(3, height, width), torch.empty(0, 4), 64)
    expected = undo_letterbox(prediction["boxes"], metadata)
    keep = (expected[:, 2:] > expected[:, :2]).all(dim=1)
    actual = torch.tensor(cli["boxes"]).reshape(-1, 4)
    with Image.open(destination) as opened:
        output_size_correct = opened.size == (width, height)
    checks = {"original_size_correct": cli["original_size"] == {"width": width, "height": height},
              "class_names_correct": cli["class_names"] == dataset.classes,
              "class_ids_in_range": all(type(i) is int and 0 <= i < len(dataset.classes) for i in cli["labels"]),
              "same_labels_as_letterbox": cli["labels"] == prediction["labels"][keep].tolist(),
              "same_scores_as_letterbox_within_1e_minus_6": torch.allclose(
                  torch.tensor(cli["scores"]), prediction["scores"][keep], atol=1e-6, rtol=0),
              "inverse_letterbox_boxes_within_1e_minus_4": torch.allclose(actual, expected[keep], atol=1e-4, rtol=0),
              "output_png_original_size": output_size_correct}
    if not all(checks.values()):
        raise RuntimeError(f"Original-image checkpoint inference failed: {checks}")
    # The same image through the command-line program, in its own process, as a reader would run it
    # from the repository root.
    standalone = output / f"validation-{index:03d}-cli-inference.png"
    command = [sys.executable, "scripts/detect_image.py", "--image", repo_path(source),
               "--checkpoint", repo_path(checkpoint), "--output", repo_path(standalone),
               "--score-threshold", str(config["score_threshold"]), "--nms-iou", str(config["nms_iou"])]
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"scripts/detect_image.py exited with {completed.returncode}: {completed.stderr[-2000:]}")
    in_process = json.loads(destination.with_suffix(".json").read_text())
    written = json.loads(standalone.with_suffix(".json").read_text())
    path_fields = ("image_path", "checkpoint_path")
    standalone_checks = {
        "path_fields_equivalent": all((REPO / written[key]).resolve() == Path(in_process[key]).resolve()
                                      for key in path_fields),
        "prediction_fields_exact": ({k: v for k, v in written.items() if k not in path_fields}
                                    == {k: v for k, v in in_process.items() if k not in path_fields})}
    if not all(standalone_checks.values()):
        raise RuntimeError(f"Standalone detect_image.py output differs from the in-process call: {standalone_checks}")
    # detect_image writes the paths it was given, absolute in this call; hash the JSON with them relative to
    # the repository, as the standalone command writes them, so the value does not depend on the checkout location.
    portable_json = json_text({**in_process, **{key: repo_path(in_process[key]) for key in path_fields}})
    return {"callable": "scripts.detect_image.detect_image", "validation_index": index,
            "detect_image_script_sha256": sha256(REPO / "scripts/detect_image.py"),
            "original_size": cli["original_size"], "output_png": repo_path(destination),
            "output_json": repo_path(destination.with_suffix(".json")),
            "output_json_sha256": hashlib.sha256(portable_json.encode()).hexdigest(),
            "output_json_sha256_scope": "Computed after rewriting output_json's image_path and checkpoint_path, which "
                                        "this call writes as absolute paths, relative to the repository (paths outside "
                                        "it stay absolute), so the value does not depend on where the checkout is",
            "checks": checks,
            "standalone_cli": {"command": shlex.join(["python", *command[1:]]), "working_directory": "repository root",
                               "output_json": repo_path(standalone.with_suffix(".json")),
                               "output_json_sha256": sha256(standalone.with_suffix(".json")),
                               "scope": "Separate process run from the repository root; its JSON must equal the in-process "
                                        "call's JSON exactly, except image_path and checkpoint_path, which may be spelled "
                                        "differently but must name the same files",
                               **standalone_checks},
            "coordinate_note": "Inference boxes use original-image pixel coordinates after undo_letterbox; AP50 evaluation and SVG use 64x64 letterboxed coordinates. This is validation-image inference, not another test evaluation.",
            "prediction": {key: cli[key] for key in ("class_names", "boxes", "scores", "labels")}}


def nice_step(raw):
    """Smallest 1, 2, 2.5 or 5 times a power of ten that is at least raw (> 0)."""
    magnitude = 10 ** math.floor(math.log10(raw))
    return next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)


def caption_lines(example, names, iou_threshold):
    """One line per prediction (TP/FP and the IoU that decided it), then one per missed GT. Past four
    lines the fourth only counts the rest; render_svg then names the record file that lists them all."""
    gt_labels = example["target"]["labels"]
    lines = []
    for row in example["matches"]["predictions"]:
        head = f'{names[row["label"]]} {row["score"]:.2f}：'
        if row["match"] == "TP":
            lines.append((head + f'TP，IoU {row["iou_same_class_gt"]:.3f}', "#15803d"))
        elif row["iou_same_class_gt"] is not None:
            lines.append((head + f'FP，IoU {row["iou_same_class_gt"]:.3f} < {iou_threshold:g}', "#c2410c"))
        elif not gt_labels:
            lines.append((head + "FP（圖中沒有 GT）", "#c2410c"))
        elif row["label"] in gt_labels:
            lines.append((head + "FP（同類 GT 已被較高分的框配對）", "#c2410c"))
        elif row["iou_any_gt"] >= iou_threshold:
            # Every GT here is another class, so this box covers one of them under the wrong class.
            lines.append((head + f'FP（類別錯，IoU {row["iou_any_gt"]:.3f}）', "#c2410c"))
        else:
            lines.append((head + f'FP（沒有同類 GT，最大 IoU {row["iou_any_gt"]:.3f}）', "#c2410c"))
    lines += [(f"GT {names[label]}：漏檢（FN）", "#b91c1c") for label in example["matches"]["missed_gt_labels"]]
    if len(lines) > 4:
        lines = lines[:3] + [(f"……另有 {len(lines) - 3} 項，見下方紀錄檔", "#0f172a")]
    return lines or [("沒有 GT，也沒有預測框", "#0f172a")]


def render_svg(path, result, images, classes, record):
    """Small SVG with embedded actual images; no plotting dependency or mock boxes.

    record is the path of the evidence JSON, named below the panels when a caption is cut short.
    """
    width, height = 1120, 840
    config, counts, steps = result["config"], result["data"]["split_counts"], result["steps_completed"]
    names = [FIGURE_NAMES.get(name, name) for name in classes]
    followup = result["predeclared_protocol"].get("followup")
    shown = lambda value: "—（沒有 GT）" if value is None else f"{value:.3f}"
    data_source = (f"{len(classes)} 類合成矩形" if result["fixture"] else f"{len(classes)} 類，使用者提供的 JSON 與圖片")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             f'<title id="title">JSON＋PNG 自訂資料：{steps} 步訓練的 loss 與 validation 偵測結果</title>',
             f'<desc id="desc">{"非正方形的合成圖" if result["fixture"] else "圖片"}先 letterbox 成 64×64。上方是每步更新前的 '
             'minibatch loss；下方綠色虛線是標註，橘框與標籤是模型實際解碼的預測，圖下寫出每個預測是 TP 還是 FP 及其 IoU。</desc>',
             f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
             f'<g font-family="{FONT_STACK}" fill="#0f172a">',
             '<text x="24" y="34" font-size="23">JSON＋PNG 自訂資料：訓練、偵測、存檔、重新載入</text>',
             f'<text x="24" y="60" font-size="14">固定設定的 CPU 實跑｜{escape(data_source)}｜train {counts["train"]["images"]} 張／'
             f'validation {counts["validation"]["images"]} 張／test {counts["test"]["images"]} 張｜模型 seed {config["seed"]}</text>',
             f'<text x="24" y="85" font-size="14">Adam {steps} 步、batch {config["batch_size"]}、learning rate {config["learning_rate"]:g}｜'
             f'候選 score ≥ {config["score_threshold"]:.2f}、同類 NMS IoU {config["nms_iou"]:.2f}、配對 IoU {config["eval_iou"]:.2f}</text>',
             f'<text x="24" y="110" font-size="14">mAP50（有 GT 的各類 AP50 平均）：validation {shown(result["final_validation"]["map"])}｜'
             f'test {shown(result["final_test"]["map"])}｜train {shown(result["final_train"]["map"])}｜'
             f'存檔再載入後模型輸出逐值相同：{"是" if result["reload_checks"]["raw_predictions_exact"] else "否"}</text>']
    values = [row["total"] for row in result["loss_history"]]
    tick = nice_step(max(values) / 5)
    top_value = math.ceil(max(values) / tick) * tick
    left, top, plot_w, plot_h = 80, 160, 980, 190
    x_of = lambda step: left + (step - 1) * plot_w / max(1, len(values) - 1)
    y_of = lambda value: top + plot_h * (1 - value / top_value)
    parts.append(f'<text x="24" y="{top - 14}" font-size="13">縱軸：每步更新前，那一批 {config["batch_size"]} 張 minibatch 的 total loss</text>')
    for k in range(round(top_value / tick) + 1):
        y = y_of(k * tick)
        parts += [f'<path d="M {left} {y:.2f} H {left + plot_w}" stroke="#e2e8f0"/>',
                  f'<text x="{left - 8}" y="{y + 4:.2f}" font-size="12" text-anchor="end">{k * tick:g}</text>']
    ticks = sorted({1, *(max(1, round(len(values) * k / 4)) for k in range(1, 5))})
    for step in ticks:
        parts += [f'<path d="M {x_of(step):.2f} {top + plot_h} v 5" stroke="#64748b"/>',
                  f'<text x="{x_of(step):.2f}" y="{top + plot_h + 19}" font-size="12" text-anchor="middle">{step}</text>']
    parts.append(f'<text x="{left}" y="{top + plot_h + 42}" font-size="13">橫軸：訓練步數（第幾次參數更新）</text>')
    points = " ".join(f"{x_of(i + 1):.2f},{y_of(value):.2f}" for i, value in enumerate(values))
    parts += [f'<path d="M {left} {top} V {top + plot_h} H {left + plot_w}" fill="none" stroke="#64748b"/>',
              f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>']
    if followup and followup["prior_steps"] <= len(values):
        # Drawn over the curve: where the diagnostic run stopped and was evaluated; its steps are
        # this run's first steps.
        prior = followup["prior_steps"]
        same = result["prior_diagnostic_comparison"]["loss_history_identical"]
        parts += [f'<path d="M {x_of(prior):.2f} {top} V {top + plot_h}" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="5 4"/>',
                  f'<path d="M 330 {top + plot_h + 37} h 26" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="5 4"/>',
                  f'<text x="364" y="{top + plot_h + 42}" font-size="13" fill="#dc2626">第 {prior} 步：{prior} 步診斷紀錄在此評估；'
                  f'前 {prior} 步的 loss 與該紀錄{"逐值相同" if same else "不同"}</text>']
        if prior not in ticks:
            parts.append(f'<text x="{x_of(prior):.2f}" y="{top + plot_h + 19}" font-size="12" text-anchor="middle" fill="#dc2626">{prior}</text>')
    parts += [f'<text x="24" y="426" font-size="15">validation 圖（沒參與訓練）｜綠虛線：GT｜橘框：模型預測與 score｜'
              f'同類且 IoU ≥ {config["eval_iou"]:g} 才算 TP，沒配到的 GT 算漏檢（FN）</text>']
    truncated = False
    for panel, example in enumerate(result["validation_examples"][:4]):
        index = example["validation_index"]
        x, y, side = 24 + panel * 274, 462, 224
        prepared = (images[index].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        encoded = io.BytesIO()
        Image.fromarray(prepared).save(encoded, format="PNG")
        source = base64.b64encode(encoded.getvalue()).decode("ascii")
        parts += [f'<text x="{x}" y="{y - 10}" font-size="14">validation 圖 #{index}</text>',
                  f'<image x="{x}" y="{y}" width="{side}" height="{side}" href="data:image/png;base64,{source}"/>']
        for entry, color, dashed in ((example["target"], "#22c55e", True), (example["prediction"], "#f97316", False)):
            for row, (label, box) in enumerate(zip(entry["labels"], entry["boxes"])):
                x1, y1, x2, y2 = box
                bx, by = x + x1 * side / 64, y + y1 * side / 64
                dash = ' stroke-dasharray="6 3"' if dashed else ""
                parts.append(f'<rect x="{bx:.2f}" y="{by:.2f}" width="{(x2-x1)*side/64:.2f}" height="{(y2-y1)*side/64:.2f}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
                label_text = f'GT {names[label]}' if dashed else f'{names[label]} {entry["scores"][row]:.2f}'
                # Use separate GT/prediction label rows so close boxes cannot
                # collapse two captions onto the same baseline.
                caption_y = y + 14 + row * 30 + (0 if dashed else 15)
                parts.append(f'<text x="{x+4}" y="{caption_y}" font-size="12" fill="{color}" stroke="#020617" stroke-width="0.4" paint-order="stroke">{escape(label_text)}</text>')
        lines = caption_lines(example, names, config["eval_iou"])
        truncated |= len(lines) < len(example["matches"]["predictions"]) + len(example["matches"]["missed_gt_labels"])
        for line, (text, color) in enumerate(lines):
            parts.append(f'<text x="{x}" y="{y + side + 20 + line * 16}" font-size="12" fill="{color}">{escape(text)}</text>')
    if truncated:
        # A repository path does not fit in one panel's column, so it gets its own line.
        parts.append(f'<text x="24" y="774" font-size="12">上面各圖完整的 TP／FP 與漏檢，列在紀錄檔 '
                     f'{escape(record)} 的 validation_examples。</text>')
    parts += ['<text x="24" y="795" font-size="14">存檔（checkpoint format v2）再載入後，逐值核對模型原始輸出、optimizer 狀態與亂數狀態。</text>',
              '<text x="24" y="820" font-size="14">' + ("合成資料的小實驗：不代表真實影片" if result["fixture"] else
                                                       "一次固定設定的短訓練：不代表其他資料") +
              '的準確度，也不能用來排名 YOLO 版本。</text>',
              '</g>', '</svg>']
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(line.rstrip() for line in parts) + "\n", encoding="utf-8")


def run(args):
    started_at = datetime.now(timezone.utc).isoformat()
    overall_start = time.perf_counter()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    annotations = create_fixture(output / "fixture", args.fixture_test_seed) if args.fixture else Path(args.annotations).resolve()
    data_root = annotations.parent if args.fixture or args.root is None else Path(args.root).resolve()
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.set_num_threads(2)
    datasets = {split: JsonDetectionDataset(annotations, root=data_root, split=split, image_size=64)
                for split in SPLITS}
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("Annotations must include nonempty train, validation, and test splits")
    counts, inventory, audit = audit_data(datasets)
    write_json(output / "image-inventory.json", inventory)
    classes = datasets["train"].classes
    # All settings are fixed before training or any held-out evaluation.
    config = {"seed": 7, "device": "cpu", "threads": 2, "batch_size": 8,
              "optimizer": "Adam", "learning_rate": .01, "width": 8, "grid_size": 4,
              "image_size": 64, "num_classes": len(classes), "steps": args.steps,
              "score_threshold": .1, "nms_iou": .5, "eval_iou": .5,
              "annotations": repo_path(annotations), "root": repo_path(data_root)}
    protocol = {"declared_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "fixture_test_seed": args.fixture_test_seed if args.fixture else None,
                "test_evaluations_at_declaration": 0, "selection": "Final fixed step; no parameter search"}
    prior = None
    if args.prior_diagnostic:
        prior = json.loads(Path(args.prior_diagnostic).read_text())
        fixed_keys = ("seed", "device", "threads", "batch_size", "optimizer", "learning_rate", "width",
                      "grid_size", "image_size", "num_classes", "score_threshold", "nms_iou", "eval_iou")
        if any(config[key] != prior["config"][key] for key in fixed_keys) or classes != prior["class_names"]:
            raise ValueError("Follow-up must retain the diagnostic run's fixed model/optimizer/evaluation settings")
        old_test_seed = prior.get("predeclared_protocol", {}).get("fixture_test_seed")
        if not prior.get("fixture") or old_test_seed is None or "decoded_rgb_inventory_sha256" not in prior["data"]:
            raise ValueError("--prior-diagnostic must be a --fixture report written by this version of the script; "
                             "rerun that diagnostic first")
        if old_test_seed == args.fixture_test_seed:
            raise ValueError("Follow-up needs a predeclared fresh synthetic test seed")
        # Recreate the diagnostic's data from its seeds rather than opening files it recorded by path.
        # Compare decoded pixels: PNG file bytes for identical pixels can differ between machines.
        with tempfile.TemporaryDirectory(prefix="prior-fixture-") as folder:
            previous_annotations = create_fixture(Path(folder), old_test_seed)
            previous_manifest = json.loads(previous_annotations.read_text())
            previous_inventory = image_inventory(previous_manifest["images"], folder)
            recreated = {"manifest_sha256": sha256(previous_annotations),
                         "decoded_rgb_inventory_sha256": rgb_inventory_sha256(previous_inventory)}
        if any(recreated[key] != prior["data"][key] for key in recreated):
            raise ValueError("Recreating the diagnostic fixture from its seeds does not reproduce the report's "
                             "manifest and decoded pixels; the fixture code or image decoding changed")
        retained = lambda rows: [row for row in rows if row["split"] in ("train", "validation")]
        if retained(inventory) != retained(previous_inventory):
            raise ValueError("Follow-up train/validation pixels must be identical to the diagnostic run")
        current_manifest = json.loads(annotations.read_text())
        if retained(current_manifest["images"]) != retained(previous_manifest["images"]):
            raise ValueError("Follow-up train/validation annotations and sources must be identical to the diagnostic run")
        old_test_pixels = {row["rgb_sha256"] for row in previous_inventory if row["split"] == "test"}
        if any(row["rgb_sha256"] in old_test_pixels for row in inventory if row["split"] == "test"):
            raise ValueError("Follow-up test must contain fresh images")
        protocol["followup"] = {"diagnostic_report": repo_path(args.prior_diagnostic),
                                "diagnostic_report_sha256": sha256(args.prior_diagnostic),
                                "prior_steps": prior["steps_completed"], "planned_steps": args.steps,
                                "recreated_diagnostic_fixture": {
                                    "split_seeds": previous_manifest["split_seeds"],
                                    "manifest_sha256_matches_report": True,
                                    "decoded_rgb_inventory_sha256_matches_report": True,
                                    "png_inventory_sha256_matches_report":
                                        hashlib.sha256(json_text(previous_inventory).encode()).hexdigest()
                                        == prior["data"]["image_inventory_sha256"]},
                                "same_train_validation_pngs": True, "same_train_validation_annotations": True,
                                "fixed_config_keys": list(fixed_keys),
                                "prior_test_seed": old_test_seed, "fresh_test_seed": args.fixture_test_seed,
                                "diagnosis": prior["short_run_outcome"],
                                "basis": "Training PNG/target/mask/coordinate diagnostics; no test-driven parameter selection",
                                "allowed_changes": ["Training step budget", "Fresh independent test only"],
                                "prior_train_box_diagnosis": prior["training_box_diagnosis"]}
    write_json(output / "protocol.json", protocol)
    batches = {split: collate([dataset[i] for i in range(len(dataset))]) for split, dataset in datasets.items()}
    train_images, train_targets = batches["train"]
    encoded_train = build_targets(train_targets, grid_size=4, image_size=64, num_classes=len(classes))
    model = GridDetector(num_classes=len(classes), grid_size=4, width=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    original = [parameter.detach().clone() for parameter in model.parameters()]
    initial_loss = full_loss(model, train_images, encoded_train)
    _, initial_predictions = predict(model, batches["validation"][0], config)
    initial_validation = evaluate(initial_predictions, batches["validation"][1], classes, config)
    history, gradients = [], []
    training_start = time.perf_counter()
    for step in range(args.steps):
        indices = [(step * 8 + offset) % len(train_images) for offset in range(8)]
        encoded = build_targets([train_targets[i] for i in indices], grid_size=4,
                                image_size=64, num_classes=len(classes))
        values, gradient = optimizer_step(model, optimizer, train_images[indices], encoded,
                                          validate_gradients=True)
        history.append({"step": step + 1, **values})
        gradients.append(gradient)
    training_seconds = time.perf_counter() - training_start
    final_loss = full_loss(model, train_images, encoded_train)
    weight_delta = math.sqrt(sum(float((parameter.detach() - before).square().sum())
                                 for parameter, before in zip(model.parameters(), original)))
    if not math.isfinite(weight_delta) or weight_delta <= 0:
        raise RuntimeError("Training did not produce a finite nonzero weight change")
    metrics, final_predictions, raw_validation = {}, {}, None
    # No test evaluation before this point, no selection after this point.
    for split, (images, targets) in batches.items():
        raw, predictions = predict(model, images, config)
        metrics[split] = evaluate(predictions, targets, classes, config)
        final_predictions[split] = predictions
        if split == "validation":
            raw_validation = raw.clone()
    foreground = (sum(fixture_foreground_boxes(data_root / r["path"]) == r["boxes"] for r in datasets["train"].records)
                  if args.fixture else None)
    box_diagnosis = training_box_diagnosis(model, train_images, train_targets, encoded_train, config["image_size"],
                                           foreground)
    checkpoint = output / "checkpoint.pt"
    saved = save_checkpoint(checkpoint, model, optimizer, config, classes, args.steps)
    restored = GridDetector(num_classes=len(classes), grid_size=4, width=8)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.01)
    loaded = load_checkpoint(checkpoint, restored, restored_optimizer)
    rng_exact = exact_tree_equal(saved["rng_state"], capture_rng())
    reload_raw, reload_predictions = predict(restored, batches["validation"][0], config)
    checks = {"format_version": loaded["format_version"],
              "steps_completed": loaded["steps_completed"],
              "raw_predictions_exact": torch.equal(raw_validation, reload_raw),
              "raw_max_absolute_difference": float((raw_validation - reload_raw).abs().max()),
              "decoded_predictions_exact": exact_tree_equal(final_predictions["validation"], reload_predictions),
              "optimizer_state_exact": exact_tree_equal(optimizer.state_dict(), restored_optimizer.state_dict()),
              "rng_state_exact": rng_exact,
              "optimizer_state_entries": len(loaded["optimizer_state_dict"]["state"]),
              "rng_keys": sorted(loaded["rng_state"])}
    if not all(checks[key] for key in ("raw_predictions_exact", "decoded_predictions_exact", "optimizer_state_exact", "rng_state_exact")):
        raise RuntimeError(f"Checkpoint reload parity failed: {checks}")
    selected = []
    for class_id in range(len(classes)):
        index = next((i for i, target in enumerate(batches["validation"][1]) if class_id in target["labels"].tolist()), None)
        if index is not None and index not in selected:
            selected.append(index)
    empty_index = next((i for i, target in enumerate(batches["validation"][1]) if len(target["labels"]) == 0), None)
    if empty_index is not None:
        selected.append(empty_index)
    selected = selected[:4] or [0]
    image_check = check_original_image(checkpoint, datasets["validation"], selected[0],
                                       final_predictions["validation"][selected[0]], output, config)
    examples = []
    for index in selected:
        image_path = output / f"validation-{index:03d}-letterbox.png"
        pixels = (batches["validation"][0][index].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(pixels).save(image_path)
        examples.append({"validation_index": index, "original_record": datasets["validation"].records[index],
                         "letterbox_png": repo_path(image_path), "letterbox_png_sha256": sha256(image_path),
                         "target": tensor_rows([batches["validation"][1][index]])[0],
                         "prediction": tensor_rows([final_predictions["validation"][index]])[0],
                         "matches": match_image(final_predictions["validation"][index], batches["validation"][1][index],
                                                config["eval_iou"])})
    write_json(output / "predictions.json", {split: tensor_rows(rows) for split, rows in final_predictions.items()})
    result = {"created_at_utc": started_at, "fixture": args.fixture, "config": config,
              "class_names": classes, "steps_completed": args.steps,
              "predeclared_protocol": protocol, "image_cli_compatibility": image_check,
              "dependencies": {"python": platform.python_version(), "torch": str(torch.__version__),
                               "numpy": str(np.__version__), "pillow": str(pillow_version)},
              "data": {"manifest_sha256": sha256(annotations), "image_inventory_sha256": sha256(output / "image-inventory.json"),
                       "decoded_rgb_inventory_sha256": rgb_inventory_sha256(inventory),
                       "split_counts": counts, "audit": audit, "inventory": repo_path(output / "image-inventory.json"),
                       "provenance": "Seeded synthetic rectangles, not real video frames" if args.fixture else "User-supplied JSON and image files; no real-world accuracy claim"},
              "code": {"script_sha256": sha256(__file__), "repository_commit": repository_commit(),
                       "commit_scope": "Checkout base commit; dependencies_sha256 identifies the executed working-tree files"},
              "dependencies_sha256": repo_dependencies(__file__),
              "initial_train_loss": initial_loss, "final_train_loss": final_loss,
              "loss_decreased_on_same_full_train_set": final_loss["total"] < initial_loss["total"],
              "loss_history": history, "gradient_l2_range": {"min": min(gradients), "max": max(gradients)},
              "all_step_gradients_finite_nonzero": all(math.isfinite(value) and value > 0 for value in gradients),
              "weight_delta_l2": weight_delta, "parameters": sum(p.numel() for p in model.parameters()),
              "initial_validation": initial_validation, "final_train": metrics["train"],
              "final_validation": metrics["validation"], "final_test": metrics["test"],
              "short_run_outcome": run_outcome(args.steps, initial_loss, final_loss, metrics),
              "training_box_diagnosis": box_diagnosis,
              "evaluation_policy": {"test_evaluations": 1, "parameter_search_runs": 0,
                                    "score_threshold": .1, "nms_iou": .5, "matching_iou": .5,
                                    "ap_definition": "All-points interpolated AP at IoU .50 per class; 'map' is mAP50, their mean over classes with GT; not COCO AP .50:.95",
                                    "score_definition": "sigmoid(objectness) * max softmax(class logits)",
                                    "precision_recall_definition": "Micro counts after score filtering and per-class NMS; each same-image same-class GT matches once",
                                    "same_evaluation_for_all_splits": True, "test_used_for_tuning": False,
                                    "selection": "Last fixed step; no AP acceptance threshold or best-validation checkpoint selection"},
              "checkpoint": {"path": repo_path(checkpoint), "sha256": sha256(checkpoint),
                             "format_version": 2, "contains_model_optimizer_rng": True},
              "reload_checks": checks, "validation_examples": examples,
              "runtime": {"training_seconds": training_seconds,
                          "training_seconds_scope": "Optimizer loop only: for every step, minibatch indexing, target building, "
                                                    "forward/backward, finite-gradient check, Adam step and loss logging",
                          "end_to_end_seconds": None,  # set just before the report is written
                          "end_to_end_seconds_scope": "run() up to the report write: fixture creation or annotation loading and data "
                                                      "checks (for a follow-up, also recreating the diagnostic's fixture), training, "
                                                      "evaluation and the training-box diagnosis, checkpoint save and reload, original-image "
                                                      "inference in this process and in a separate scripts/detect_image.py process "
                                                      "(including that process's start and imports), the other files under the output "
                                                      "folder, and the SVG; excludes package installation, this process's start and "
                                                      "imports, and writing the report",
                          "torch_threads": torch.get_num_threads(),
                          "device": "cpu", "platform": platform.platform(), "machine": machine_info(),
                          "gpu_runs": 0, "network_requests": 0},
              "limitations": ["Short overfit exercise on 24 training images for the fixture; no proof of broad generalization.",
                              "Independent synthetic sources do not reproduce real camera, lighting, occlusion or motion variation.",
                              "Exact duplicate/source checks cannot establish that near-duplicate real frames are independent.",
                              "One box per grid cell; same-cell annotations fail explicitly rather than being dropped.",
                              "Checkpoint parity verifies this CPU reload, not cross-device bitwise equality or resumed optimization parity.",
                              "AP and runtime describe one fixed run only; no YOLO-version ranking or benchmark claim."]}
    if prior is not None:
        compared = min(prior["steps_completed"], args.steps)
        differing = next((row["step"] for row, old in zip(history[:compared], prior["loss_history"][:compared])
                          if row != old), None)
        result["prior_diagnostic_comparison"] = {
            "scope": "Minibatch loss components of the first steps, compared exactly with the diagnostic report",
            "steps_compared": compared, "loss_history_identical": differing is None, "first_differing_step": differing}
    report = Path(args.report) if args.report else output / "report.json"
    diagram = Path(args.diagram) if args.diagram else output / "learning.svg"
    render_svg(diagram, result, batches["validation"][0], classes, repo_path(report))
    result["diagram"] = repo_path(diagram)
    result["runtime"]["end_to_end_seconds"] = time.perf_counter() - overall_start
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(report, result)
    print(json.dumps({"report": repo_path(report), "diagram": repo_path(diagram), "steps": args.steps,
                      "train_loss_before": initial_loss["total"], "train_loss_after": final_loss["total"],
                      "train_map50": metrics["train"]["map"], "validation_map50": metrics["validation"]["map"],
                      "test_map50": metrics["test"]["map"], "reload_raw_exact": checks["raw_predictions_exact"],
                      "training_seconds": training_seconds}, indent=2))
    if not result["loss_decreased_on_same_full_train_set"]:
        raise RuntimeError("This fixed run did not reduce full-training loss; retained the actual evidence without tuning")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", action="store_true", help="Generate independent synthetic PNG/JSON under output/fixture")
    source.add_argument("--annotations", help="Existing classes/images JSON, with train/validation/test records")
    parser.add_argument("--root", help="Root for relative image paths; defaults to annotation directory")
    parser.add_argument("--output", default=str(REPO / "artifacts/runs/custom-data-learning"))
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--fixture-test-seed", type=int, default=7000,
                        help="Synthetic test seed fixed before training; a follow-up uses a fresh explicitly declared seed")
    parser.add_argument("--prior-diagnostic", help="Prior --fixture report; regenerate its fixture and verify unchanged train/validation and fixed settings before a follow-up")
    parser.add_argument("--report", help="Evidence JSON path; default output/report.json")
    parser.add_argument("--diagram", help="Actual loss/validation SVG path; default output/learning.svg")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be a positive integer")
    if args.fixture and args.root is not None:
        parser.error("--root is for user annotations; fixture images are written under --output")
    if args.prior_diagnostic and not args.fixture:
        parser.error("--prior-diagnostic regenerates a synthetic fixture; use it with --fixture")
    run(args)


if __name__ == "__main__":
    main()
