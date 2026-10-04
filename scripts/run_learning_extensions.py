"""Short learning extensions using the exact lesson models, no pretrained weights.

Results go to artifacts/runs/learning/<section>/ (report.json, learning.svg). The recorded run that
the website shows adds --record, which also writes artifacts/checks/curriculum/<section>-learning.json
and docs/assets/diagrams/<section>-learning.svg.
"""
import argparse
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/miniyolo-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/miniyolo-cache")
import importlib.util
import json
import math
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(lesson):
    spec = importlib.util.spec_from_file_location("lesson", ROOT / f"lesson_cases/{lesson}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LIMITS = {
    "01-small-cnn": "One fixed tiny synthetic batch, trained and scored on the same images; no held-out evaluation; not a version ranking",
    "03-comparison": "One fixed tiny synthetic batch; held-out accuracy only on the 4 shifted examples; not a version ranking",
    "04-localization": "Two fixed synthetic images, trained and scored on the same images; no held-out evaluation; not a version ranking",
}
# 04-localization trains the lesson's loss: classification cross entropy + BOX_WEIGHT * box MSE.
BOX_WEIGHT = 5
# Line colours as on the course's other loss charts (blue total, purple classification, red box). This
# blue is lighter than theirs so it stays distinct from the purple for colour-blind readers; orange is
# the second model of 03-comparison.
BLUE, ORANGE, PURPLE, RED = "#0284c7", "#ea580c", "#7c3aed", "#dc2626"
# Legend labels. Legends get no frame edge: browsers draw Chinese text wider than matplotlib
# measures it (miniyolo/figures.py), so a drawn frame would cut through the labels.
LEGEND = {"plain": "plain（無 shortcut）", "residual": "residual（有 shortcut）"}
STEP_AXIS = "第幾次更新（每點在該次更新之前量）"


def run(lesson, output, record=False):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    module = load(lesson)
    if lesson == "01-small-cnn":
        images, labels = module.make_batch()
        models = {"cnn": module.SmallCNN()}
        targets = None
    elif lesson == "03-comparison":
        images, labels = module.data(8, 0)
        models = {"plain": module.Classifier(False), "residual": module.Classifier(True)}
        models["residual"].load_state_dict(models["plain"].state_dict())
        targets = None
    else:
        images = torch.zeros(2, 3, 32, 32)
        boxes = torch.tensor([[4., 6., 16., 18.], [16., 10., 28., 22.]])
        labels = torch.tensor([0, 1])
        for i, (x1, y1, x2, y2) in enumerate(boxes.long().tolist()):
            images[i, 0 if i == 0 else 2, y1:y2, x1:x2] = 1
        targets = torch.cat(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]), 1) / 32
        models = {"localizer": module.Localizer()}
    results = {}
    for name, model in models.items():
        optimizer = (torch.optim.SGD(model.parameters(), lr=.1) if lesson == "03-comparison"
                     else torch.optim.Adam(model.parameters(), lr=.01))
        before = [p.detach().clone() for p in model.parameters()]
        history, class_history, box_history, gradients = [], [], [], []
        for step in range(40):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(images)
            if targets is None:
                loss = nn.functional.cross_entropy(prediction, labels)
            else:
                class_loss = nn.functional.cross_entropy(prediction[0], labels)
                box_loss = nn.functional.mse_loss(prediction[1], targets)
                loss = class_loss + BOX_WEIGHT * box_loss
                class_history.append(float(class_loss.detach()))
                box_history.append(float(box_loss.detach()))
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            norm = float(torch.stack([p.grad.square().sum() for p in model.parameters()]).sum().sqrt())
            assert norm > 0
            optimizer.step()
            history.append(float(loss.detach()))
            gradients.append(norm)
        changed = sum(float((p.detach() - old).square().sum()) for p, old in zip(model.parameters(), before)) ** .5
        assert changed > 0
        model.eval()
        with torch.no_grad():
            prediction = model(images)
            logits = prediction if targets is None else prediction[0]
            train_accuracy = float((logits.argmax(1) == labels).float().mean())
            val_accuracy = None
            if lesson == "03-comparison":
                vx, vy = module.data(4, 2)
                val_accuracy = float((model(vx).argmax(1) == vy).float().mean())
        result = {"steps": 40, "samples": len(images), "optimizer": type(optimizer).__name__,
                  "learning_rate": optimizer.param_groups[0]["lr"], "parameters": sum(p.numel() for p in model.parameters()),
                  "initial_loss": history[0], "last_pre_update_loss": history[-1], "loss_history": history,
                  "train_accuracy": train_accuracy, "validation_accuracy": val_accuracy,
                  "predicted_classes": logits.argmax(1).tolist(), "labels": labels.tolist(),
                  "weight_delta_l2": changed, "min_gradient_l2": min(gradients), "max_gradient_l2": max(gradients)}
        if targets is not None:
            from miniyolo.geometry import box_iou
            pixel_pred = module.to_xyxy(prediction[1]) * 32
            result["predicted_pixel_boxes"] = pixel_pred.tolist()
            result["training_box_ious"] = box_iou(pixel_pred, boxes).diag().tolist()
            # Record both terms of every step's total (classification CE, unweighted box MSE); they must add up.
            weighted = [BOX_WEIGHT * value for value in box_history]
            assert all(math.isclose(t, c + w, rel_tol=1e-6)
                       for t, c, w in zip(history, class_history, weighted, strict=True))
            result.update({"box_loss_weight": BOX_WEIGHT, "classification_loss_history": class_history,
                           "box_loss_history": box_history, "weighted_box_loss_history": weighted})
        results[name] = result
    from miniyolo.provenance import file_sha256, machine_info, repo_dependencies
    report = {"lesson": lesson, "seed": 7, "torch": str(torch.__version__), "device": "cpu",
              "machine": machine_info(2),
              "code": {"script_sha256": file_sha256(__file__),
                       "lesson_case_sha256": file_sha256(ROOT / f"lesson_cases/{lesson}.py")},
              # The script, the lesson case and every repository module they import (scripts/evidence_records.py).
              "dependencies_sha256": repo_dependencies(__file__, ROOT / f"lesson_cases/{lesson}.py"),
              "models": results, "limits": LIMITS[lesson]}
    from miniyolo.figures import save_svg, use_svg_text
    use_svg_text()
    import matplotlib.pyplot as plt
    steps = range(1, 41)
    setup = f"{len(images)} 張固定的人工訓練圖、seed 7、更新 40 次"
    if targets is None:
        fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
        for (name, result), color in zip(results.items(), (BLUE, ORANGE)):
            ax.plot(steps, result["loss_history"], color=color, label=LEGEND.get(name, name))
        ax.set(xlabel=STEP_AXIS, ylabel="訓練 loss（交叉熵）", title=setup)
        if len(results) > 1:  # one line needs no legend
            ax.legend(loc="lower left", edgecolor="none")
        axes = [ax]
        title = "40 次更新的實測訓練 loss"
        label = f"{' 與 '.join(results)} 的訓練 loss：{setup}，每點在該次更新之前量"
    else:
        result = results["localizer"]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True)
        # Dashed and on top: where the box term is tiny the total lies on the classification line.
        axes[0].plot(steps, result["loss_history"], color=BLUE, linestyle="--", zorder=3, label="總 loss")
        axes[0].plot(steps, result["classification_loss_history"], color=PURPLE, label="分類 loss（交叉熵）")
        axes[0].plot(steps, result["weighted_box_loss_history"], color=RED, label=f"{BOX_WEIGHT}×框 loss")
        axes[0].set(xlabel=STEP_AXIS, ylabel="訓練 loss", title=f"總 loss＝分類 loss＋{BOX_WEIGHT}×框 loss")
        axes[0].legend(loc="center left", edgecolor="none")
        # The box MSE alone, on its own scale (a second chart, not a second y-axis).
        axes[1].plot(steps, result["box_loss_history"], color=RED)
        axes[1].set(xlabel=STEP_AXIS, ylabel="框 loss（MSE）", title=f"框 loss 本身（未乘 {BOX_WEIGHT}）")
        fig.suptitle(setup)
        title = "40 次更新的實測 loss：總 loss、分類 loss 與框 loss"
        label = (f"左圖：總 loss 與它的兩項，分類 loss 和 {BOX_WEIGHT} 倍框 loss；右圖：框 loss 本身。"
                 f"{setup}，每點在該次更新之前量")
    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb")
        ax.set_axisbelow(True)
    # Files are written only now, after every check, and the record last: a run that fails before it
    # leaves the record out of date, so scripts/record_evidence.py runs it again.
    output.mkdir(parents=True, exist_ok=True)
    svg = save_svg(fig, output / "learning.svg", title, label)
    plt.close(fig)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if record:
        (ROOT / f"docs/assets/diagrams/{lesson}-learning.svg").write_text(svg)
        folder = ROOT / "artifacts/checks/curriculum"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{lesson}-learning.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=sorted(LIMITS), required=True)
    parser.add_argument("--output", type=Path, help="default: artifacts/runs/learning/<section>")
    parser.add_argument("--record", action="store_true",
                        help="also write the website record and figure (artifacts/checks/curriculum, docs/assets/diagrams)")
    args = parser.parse_args()
    result = run(args.section, args.output or ROOT / "artifacts/runs/learning" / args.section, args.record)
    print(json.dumps({**result, "models": {n: {k: v for k, v in r.items() if not k.endswith("_history")}
                                          for n, r in result["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
