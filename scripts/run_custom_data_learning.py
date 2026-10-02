"""CPU-only JSON + PNG -> learning -> held-out detection -> checkpoint reload.

Example fixture (synthetic rectangles, not frames from a real video)::

    .venv-model/bin/python scripts/run_custom_data_learning.py --fixture

Own annotations use the existing JsonDetectionDataset pixel-xyxy schema::

    .venv-model/bin/python scripts/run_custom_data_learning.py \
        --annotations /path/annotations.json --root /path/images --steps 160

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
import sys
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
from miniyolo.inference import decode_grid
from miniyolo.geometry import letterbox, undo_letterbox
from miniyolo.losses import grid_loss
from miniyolo.metrics import evaluate_ap
from miniyolo.models import GridDetector
from miniyolo.targets import build_targets
from miniyolo.train import optimizer_step
from scripts.detect_image import detect_image


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
            pixels = rng.integers(0, 13, (height, width, 3), dtype=np.uint8)
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


def audit_data(datasets):
    file_hashes, pixel_hashes, sources, counts = {}, {}, {}, {}
    for split, dataset in datasets.items():
        empty = sum(not record["boxes"] for record in dataset.records)
        class_counts = {name: 0 for name in dataset.classes}
        for record in dataset.records:
            sources.setdefault(record["source_id"], set()).add(split)
            path = dataset.root / record["path"]
            digest = sha256(path)
            file_hashes.setdefault(digest, set()).add(split)
            with Image.open(path) as image:
                # Include dimensions: identical bytes interpreted with a different
                # width are not identical images. Check decoded RGB as well as PNG.
                rgb = image.convert("RGB")
                pixel_digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            pixel_hashes.setdefault(pixel_digest, set()).add(split)
            for label in record["labels"]:
                class_counts[dataset.classes[label]] += 1
        counts[split] = {"images": len(dataset), "empty_images": empty,
                         "objects": sum(class_counts.values()), "objects_per_class": class_counts,
                         "source_ids": sorted({r["source_id"] for r in dataset.records}),
                         "non_square_images": sum(r["width"] != r["height"] for r in dataset.records)}
    leaks = sum(len(splits) > 1 for splits in sources.values())
    file_duplicates = sum(len(splits) > 1 for splits in file_hashes.values())
    pixel_duplicates = sum(len(splits) > 1 for splits in pixel_hashes.values())
    if leaks or file_duplicates or pixel_duplicates:
        raise ValueError("Cross-split source or exact image duplication; split original sources before training")
    # Fingerprint both the annotation manifest and every actual PNG, so editing
    # pixels without changing boxes changes the dataset identity.
    inventory = [{"path": r["path"], "split": split, "sha256": sha256(ds.root / r["path"])}
                 for split, ds in datasets.items() for r in ds.records]
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


def evaluate(predictions, targets, classes, config):
    result = evaluate_ap(predictions, targets, num_classes=len(classes), iou_threshold=config["eval_iou"])
    result["ap50_per_class_name"] = {name: result["ap_per_class"][i] for i, name in enumerate(classes)}
    result["images"] = len(targets)
    result["ground_truth_objects"] = sum(len(target["labels"]) for target in targets)
    result["decoded_predictions"] = sum(len(prediction["labels"]) for prediction in predictions)
    return result


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


def check_original_image(checkpoint, dataset, index, prediction, output):
    record = dataset.records[index]
    source = dataset.root / record["path"]
    destination = output / f"validation-{index:03d}-original-inference.png"
    cli = detect_image(source, checkpoint, destination)
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
    return {"callable": "scripts.detect_image.detect_image", "validation_index": index,
            "detect_image_script_sha256": sha256(REPO / "scripts/detect_image.py"),
            "original_size": cli["original_size"], "output_png": str(destination),
            "output_json": str(destination.with_suffix(".json")),
            "output_json_sha256": sha256(destination.with_suffix(".json")), "checks": checks,
            "coordinate_note": "Inference boxes use original-image pixel coordinates after undo_letterbox; AP50 evaluation and SVG use 64x64 letterboxed coordinates. This is validation-image inference, not another test evaluation.",
            "prediction": {key: cli[key] for key in ("class_names", "boxes", "scores", "labels")}}


def render_svg(path, result, images, targets, predictions, classes, indices):
    """Small SVG with embedded actual images; no plotting dependency or mock boxes."""
    width, height = 1120, 790
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             '<title id="title">Actual JSON PNG learning curve and independent validation detections</title>',
             '<desc id="desc">Synthetic non-square images are letterboxed to 64 pixels. Green dashed boxes are annotations. Orange boxes and labels are actual decoded model predictions.</desc>',
             f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
             '<g font-family="sans-serif" fill="#0f172a">',
             '<text x="24" y="34" font-size="23">JSON + PNG: learn, detect, save, reload</text>',
             '<text x="24" y="60" font-size="14">Fixed CPU run | 3 synthetic classes | 24 train / 12 validation / 12 test | seed 7</text>' if result["fixture"] else
             f'<text x="24" y="60" font-size="14">Fixed CPU run | {len(classes)} classes | user supplied JSON and images | seed 7</text>',
             f'<text x="24" y="85" font-size="14">{result["steps_completed"]} Adam steps, batch 8, lr 0.01 | score &gt;= 0.10, NMS 0.50, AP at IoU 0.50</text>',
             f'<text x="24" y="110" font-size="14">Validation AP50 {result["final_validation"]["map"]!s}; test AP50 {result["final_test"]["map"]!s}; raw reload exact: {result["reload_checks"]["raw_predictions_exact"]}</text>']
    values = [row["total"] for row in result["loss_history"]]
    maximum = max(values) * 1.08
    left, top, plot_w, plot_h = 70, 145, 980, 215
    points = " ".join(f'{left + i * plot_w / max(1, len(values) - 1):.2f},{top + plot_h * (1 - value / maximum):.2f}'
                      for i, value in enumerate(values))
    parts += [f'<path d="M {left} {top} V {top + plot_h} H {left + plot_w}" fill="none" stroke="#64748b"/>',
              f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>',
              f'<text x="24" y="{top + 12}" font-size="13">{maximum:.2f}</text>',
              f'<text x="45" y="{top + plot_h}" font-size="13">0</text>',
              '<text x="70" y="384" font-size="14">Actual minibatch total loss before each optimizer update</text>',
              f'<text x="985" y="384" font-size="14">step {len(values)}</text>',
              '<text x="24" y="417" font-size="15">Validation images (unseen during training) | green dashed: GT | orange: prediction + score</text>']
    for panel, index in enumerate(indices[:4]):
        x, y, side = 24 + panel * 274, 446, 224
        prepared = (images[index].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        encoded = io.BytesIO()
        Image.fromarray(prepared).save(encoded, format="PNG")
        source = base64.b64encode(encoded.getvalue()).decode("ascii")
        parts += [f'<text x="{x}" y="{y - 10}" font-size="14">validation #{index}</text>',
                  f'<image x="{x}" y="{y}" width="{side}" height="{side}" href="data:image/png;base64,{source}"/>']
        for entry, color, dashed in ((targets[index], "#22c55e", True), (predictions[index], "#f97316", False)):
            for row, label, box in zip(range(len(entry["labels"])), entry["labels"].tolist(), entry["boxes"].tolist()):
                x1, y1, x2, y2 = box
                bx, by = x + x1 * side / 64, y + y1 * side / 64
                dash = ' stroke-dasharray="6 3"' if dashed else ""
                parts.append(f'<rect x="{bx:.2f}" y="{by:.2f}" width="{(x2-x1)*side/64:.2f}" height="{(y2-y1)*side/64:.2f}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
                name = classes[label].removesuffix("_rectangle")
                label_text = f'GT {name}' if dashed else f'{name} {float(entry["scores"][row]):.2f}'
                # Use separate GT/prediction label rows so close boxes cannot
                # collapse two captions onto the same baseline.
                caption_y = y + 14 + row * 30 + (0 if dashed else 15)
                parts.append(f'<text x="{x+4}" y="{caption_y}" font-size="12" fill="{color}" stroke="#020617" stroke-width="0.4" paint-order="stroke">{escape(label_text)}</text>')
        parts.append(f'<text x="{x}" y="{y + side + 20}" font-size="12">GT: {len(targets[index]["labels"])} | decoded: {len(predictions[index]["labels"])}</text>')
    parts += ['<text x="24" y="730" font-size="14">Raw model output, optimizer state, and RNG are checked after format-v2 checkpoint reload.</text>',
              '<text x="24" y="757" font-size="14">Synthetic exercise only: a short run does not establish real-video accuracy or YOLO-version rankings.</text>',
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
                for split in ("train", "validation", "test")}
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
              "annotations": str(annotations), "root": str(data_root)}
    protocol = {"declared_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "fixture_test_seed": args.fixture_test_seed if args.fixture else None,
                "test_evaluations_at_declaration": 0, "selection": "Final fixed step; no parameter search"}
    if args.prior_diagnostic:
        prior = json.loads(Path(args.prior_diagnostic).read_text())
        fixed_keys = ("seed", "device", "threads", "batch_size", "optimizer", "learning_rate", "width",
                      "grid_size", "image_size", "num_classes", "score_threshold", "nms_iou", "eval_iou")
        if any(config[key] != prior["config"][key] for key in fixed_keys) or classes != prior["class_names"]:
            raise ValueError("Follow-up must retain the diagnostic run's fixed model/optimizer/evaluation settings")
        previous_inventory = json.loads(Path(prior["data"]["inventory"]).read_text())
        retained = lambda rows: [row for row in rows if row["split"] in ("train", "validation")]
        if retained(inventory) != retained(previous_inventory):
            raise ValueError("Follow-up train/validation pixels must be identical to the diagnostic run")
        previous_manifest = json.loads(Path(prior["config"]["annotations"]).read_text())
        current_manifest = json.loads(annotations.read_text())
        if retained(current_manifest["images"]) != retained(previous_manifest["images"]):
            raise ValueError("Follow-up train/validation annotations and sources must be identical to the diagnostic run")
        old_test_seed = previous_manifest.get("split_seeds", {}).get("test")
        if not args.fixture or old_test_seed == args.fixture_test_seed:
            raise ValueError("Follow-up needs a predeclared fresh synthetic test seed")
        old_test_hashes = {row["sha256"] for row in previous_inventory if row["split"] == "test"}
        if any(row["sha256"] in old_test_hashes for row in inventory if row["split"] == "test"):
            raise ValueError("Follow-up test must contain fresh images")
        protocol["followup"] = {"diagnostic_report": str(Path(args.prior_diagnostic)),
                                "diagnostic_report_sha256": sha256(args.prior_diagnostic),
                                "prior_steps": prior["steps_completed"], "planned_steps": args.steps,
                                "same_train_validation_pngs": True, "same_train_validation_annotations": True,
                                "fixed_config_keys": list(fixed_keys),
                                "prior_test_seed": old_test_seed, "fresh_test_seed": args.fixture_test_seed,
                                "diagnosis": prior.get("short_run_outcome"),
                                "basis": "Training PNG/target/mask/coordinate diagnostics; no test-driven parameter selection",
                                "allowed_changes": ["Training step budget", "Fresh independent test only"],
                                "prior_train_box_diagnosis": prior.get("training_box_diagnosis")}
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
                                       final_predictions["validation"][selected[0]], output)
    examples = []
    for index in selected:
        image_path = output / f"validation-{index:03d}-letterbox.png"
        pixels = (batches["validation"][0][index].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(pixels).save(image_path)
        examples.append({"validation_index": index, "original_record": datasets["validation"].records[index],
                         "letterbox_png": str(image_path), "letterbox_png_sha256": sha256(image_path),
                         "target": tensor_rows([batches["validation"][1][index]])[0],
                         "prediction": tensor_rows([final_predictions["validation"][index]])[0]})
    write_json(output / "predictions.json", {split: tensor_rows(rows) for split, rows in final_predictions.items()})
    result = {"created_at_utc": started_at, "fixture": args.fixture, "config": config,
              "class_names": classes, "steps_completed": args.steps,
              "predeclared_protocol": protocol, "image_cli_compatibility": image_check,
              "dependencies": {"python": platform.python_version(), "torch": str(torch.__version__),
                               "numpy": str(np.__version__), "pillow": str(pillow_version)},
              "data": {"manifest_sha256": sha256(annotations), "image_inventory_sha256": sha256(output / "image-inventory.json"),
                       "split_counts": counts, "audit": audit, "inventory": str(output / "image-inventory.json"),
                       "provenance": "Seeded synthetic rectangles, not real video frames" if args.fixture else "User-supplied JSON and image files; no real-world accuracy claim"},
              "code": {"script_sha256": sha256(__file__), "repository_commit": repository_commit(),
                       "commit_scope": "Checkout base commit; script SHA identifies the executed working-tree script",
                       "core_sha256": {name: sha256(REPO / "miniyolo" / name) for name in
                                       ("custom_data.py", "models.py", "targets.py", "train.py", "losses.py", "inference.py", "metrics.py", "checkpoint.py")}},
              "initial_train_loss": initial_loss, "final_train_loss": final_loss,
              "loss_decreased_on_same_full_train_set": final_loss["total"] < initial_loss["total"],
              "loss_history": history, "gradient_l2_range": {"min": min(gradients), "max": max(gradients)},
              "all_step_gradients_finite_nonzero": all(math.isfinite(value) and value > 0 for value in gradients),
              "weight_delta_l2": weight_delta, "parameters": sum(p.numel() for p in model.parameters()),
              "initial_validation": initial_validation, "final_train": metrics["train"],
              "final_validation": metrics["validation"], "final_test": metrics["test"],
              "evaluation_policy": {"test_evaluations": 1, "parameter_search_runs": 0,
                                    "score_threshold": .1, "nms_iou": .5, "matching_iou": .5,
                                    "ap_definition": "All-points interpolated AP at IoU .50, mean over classes with GT; not COCO AP .50:.95",
                                    "score_definition": "sigmoid(objectness) * max softmax(class logits)",
                                    "precision_recall_definition": "Micro counts after score filtering and per-class NMS; each same-image same-class GT matches once",
                                    "same_evaluation_for_all_splits": True, "test_used_for_tuning": False,
                                    "selection": "Last fixed step; no AP acceptance threshold or best-validation checkpoint selection"},
              "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint),
                             "format_version": 2, "contains_model_optimizer_rng": True},
              "reload_checks": checks, "validation_examples": examples,
              "runtime": {"training_seconds": training_seconds, "torch_threads": torch.get_num_threads(),
                          "device": "cpu", "platform": platform.platform(),
                          "scope": "Observed wall time for this CPU run, including dataset preparation, training, evaluation and reload; excludes environment installation",
                          "gpu_runs": 0, "network_requests": 0},
              "limitations": ["Short overfit exercise on 24 training images for the fixture; no proof of broad generalization.",
                              "Independent synthetic sources do not reproduce real camera, lighting, occlusion or motion variation.",
                              "Exact duplicate/source checks cannot establish that near-duplicate real frames are independent.",
                              "One box per grid cell; same-cell annotations fail explicitly rather than being dropped.",
                              "Checkpoint parity verifies this CPU reload, not cross-device bitwise equality or resumed optimization parity.",
                              "AP and runtime describe one fixed run only; no YOLO-version ranking or benchmark claim."]}
    report = Path(args.report) if args.report else output / "report.json"
    result["code_commit_scope"] = result["code"]["commit_scope"]
    diagram = Path(args.diagram) if args.diagram else output / "learning.svg"
    render_svg(diagram, result, batches["validation"][0], batches["validation"][1],
               final_predictions["validation"], classes, selected)
    result["diagram"] = str(diagram)
    result["runtime"]["end_to_end_seconds"] = time.perf_counter() - overall_start
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(report, result)
    print(json.dumps({"report": str(report), "diagram": str(diagram), "steps": args.steps,
                      "train_loss_before": initial_loss["total"], "train_loss_after": final_loss["total"],
                      "train_ap50": metrics["train"]["map"], "validation_ap50": metrics["validation"]["map"],
                      "test_ap50": metrics["test"]["map"], "reload_raw_exact": checks["raw_predictions_exact"],
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
    parser.add_argument("--prior-diagnostic", help="Prior short-run evidence; verify unchanged train/validation and fixed settings before a follow-up")
    parser.add_argument("--report", help="Evidence JSON path; default output/report.json")
    parser.add_argument("--diagram", help="Actual loss/validation SVG path; default output/learning.svg")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be a positive integer")
    if args.fixture and args.root is not None:
        parser.error("--root is for user annotations; fixture images are written under --output")
    run(args)


if __name__ == "__main__":
    main()
