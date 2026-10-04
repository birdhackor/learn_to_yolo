"""Fit the lesson's two-scale network to one image, then decode actual outputs.

Results go to artifacts/runs/multiscale-learning/ (report.json, learning.svg). The recorded run that
the website shows adds --record, which also writes artifacts/checks/curriculum/10-multiscale-learning.json
and docs/assets/diagrams/10-multiscale-learning.svg.
"""
import argparse
import io
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_learning_extensions import BLUE, STEP_AXIS, load
from miniyolo.targets import build_targets
from miniyolo.losses import grid_loss
from miniyolo.inference import decode_grid
from miniyolo.geometry import nms
from miniyolo.metrics import evaluate_ap


def overlaps(a, b, margin=0):
    """Whether rectangles a and b, each (x1, y1, x2, y2), are less than margin apart (overlap, for 0)."""
    return a[0] < b[2] + margin and b[0] - margin < a[2] and a[1] < b[3] + margin and b[1] - margin < a[3]


def label_size(text, fontsize):
    """Generous size in points of one line of text as a browser draws it with its own fonts: CJK
    characters 1 em wide, others .6 em (DejaVu Sans averages about .5), 1.6 em tall."""
    return sum(1 if ord(ch) >= 0x2E80 else .6 for ch in text) * fontsize, 1.6 * fontsize


def label_spot(box, size, boxes, taken, margin, side=64):
    """Rectangle and text alignment for a w x h label of box: outside every box by at least margin, so no
    box line crosses it, inside the side x side image and off the rectangles taken. Above or below box
    and the boxes it overlaps, else beside box at its bottom, whichever fits first."""
    (x1, _, x2, y2), (w, h) = box, size
    cluster = [other for other in boxes if overlaps(other, box)]  # box and the boxes it overlaps
    top, bottom = min(b[1] for b in cluster) - margin, max(b[3] for b in cluster) + margin
    spots = [((x1, top - h, x1 + w, top), "left"), ((x1, bottom, x1 + w, bottom + h), "left"),
             ((x1 - margin - w, y2 - margin - h, x1 - margin, y2 - margin), "right"),
             ((x2 + margin, y2 - margin - h, x2 + margin + w, y2 - margin), "left")]
    free = [(rect, ha) for rect, ha in spots if 0 <= rect[0] and 0 <= rect[1] and rect[2] <= side and rect[3] <= side
            and not any(overlaps(rect, other, margin) for other in boxes)
            and not any(overlaps(rect, other) for other in taken)]
    assert free, f"no room outside every box for the label of box {box}"
    return free[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/runs/multiscale-learning")
    parser.add_argument("--record", action="store_true",
                        help="also write the website record and figure (artifacts/checks/curriculum, docs/assets/diagrams)")
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(2)
    module = load("10-multiscale")
    small = {"boxes": torch.tensor([[5., 5., 13., 13.]]), "labels": torch.tensor([0])}
    large = {"boxes": torch.tensor([[32., 32., 56., 56.]]), "labels": torch.tensor([1])}
    fine_target, coarse_target = build_targets([small], 8), build_targets([large], 4)
    image = torch.zeros(1, 3, 64, 64)
    image[0, 0, 5:13, 5:13] = 1
    image[0, 2, 32:56, 32:56] = 1
    model = module.TwoScale()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    history = []
    for step in range(40):
        optimizer.zero_grad(set_to_none=True)
        fine, coarse = model(image)
        loss = grid_loss(fine, fine_target)["total"] + grid_loss(coarse, coarse_target)["total"]
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.fine_head.weight.grad.abs().sum() > 0 and model.coarse_head.weight.grad.abs().sum() > 0
        optimizer.step()
        history.append(float(loss.detach()))
    assert history[-1] < history[0]
    delta = sum(float((p.detach() - old).square().sum()) for p, old in zip(model.parameters(), before)) ** .5
    assert delta > 0
    model.eval()
    with torch.inference_mode():
        heads = model(image)
        decoded = [decode_grid(raw, score_threshold=.05)[0] for raw in heads]
    boxes = torch.cat([d["boxes"] for d in decoded])
    scores = torch.cat([d["scores"] for d in decoded])
    labels = torch.cat([d["labels"] for d in decoded])
    keep = []
    for cls in labels.unique():
        ids = torch.where(labels == cls)[0]
        keep.append(ids[nms(boxes[ids], scores[ids], .5)])
    ids = torch.cat(keep) if keep else torch.empty(0, dtype=torch.long)
    prediction = {"boxes": boxes[ids], "scores": scores[ids], "labels": labels[ids]}
    gt = {"boxes": torch.cat([small["boxes"], large["boxes"]]), "labels": torch.tensor([0, 1])}
    metrics = evaluate_ap([prediction], [gt], num_classes=2)
    from miniyolo.provenance import file_sha256, machine_info, repo_dependencies
    report = {"torch": str(torch.__version__), "device": "cpu", "machine": machine_info(2),
              "code": {"script_sha256": file_sha256(__file__),
                       "lesson_case_sha256": file_sha256(ROOT / "lesson_cases/10-multiscale.py")},
              # The script, the lesson case and every repository module they import (scripts/evidence_records.py).
              "dependencies_sha256": repo_dependencies(__file__, ROOT / "lesson_cases/10-multiscale.py"),
              "seed": 7, "steps": 40, "samples": 1,
              "optimizer": "Adam", "learning_rate": .01, "initial_loss": history[0], "last_pre_update_loss": history[-1],
              "loss_history": history, "parameters": sum(p.numel() for p in model.parameters()), "weight_delta_l2": delta,
              "training_image_metrics": metrics, "prediction": {k: v.tolist() for k, v in prediction.items()},
              "limits": "One training image, hand-chosen size responsibility; not held-out small-object AP and not a single-scale comparison"}
    from miniyolo.figures import save_svg, use_svg_text
    use_svg_text()
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True)
    axes[0].plot(range(1, 41), history, color=BLUE)
    axes[0].grid(axis="y", color="#e5e7eb")
    axes[0].set_axisbelow(True)
    axes[0].set(xlabel=STEP_AXIS, ylabel="訓練 loss（fine＋coarse）", title="1 張固定的人工訓練圖、seed 7、更新 40 次")
    axes[1].imshow(image[0].permute(1, 2, 0).numpy(), extent=(0, 64, 64, 0))
    # As on the chapter 7 and 8 prediction figures: GT green dashed, predictions orange.
    gt_color, prediction_color = "#22c55e", "#fb923c"
    for group, color, style in ((gt, gt_color, "--"), (prediction, prediction_color, "-")):
        for x1, y1, x2, y2 in group["boxes"].tolist():
            axes[1].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linestyle=style, linewidth=1.5))
    # Black, edgeless legend on the image's black top: Chinese labels drawn wider than matplotlib
    # measured them still sit on black (miniyolo/figures.py).
    legend = axes[1].legend(handles=[Line2D([], [], color=gt_color, linestyle="--", label="GT"),
                                     Line2D([], [], color=prediction_color, label="預測框")], loc="upper center",
                            facecolor="black", edgecolor="none", labelcolor="white")
    axes[1].set(xlim=(0, 64), ylim=(64, 0), xlabel="x（pixel）", ylabel="y（pixel）",
                title="40 次更新後，同一張訓練圖上的預測")
    # Lay the figure out with the SVG renderer, so that points convert to image pixels as in the saved file.
    fig.savefig(io.BytesIO(), format="svg")
    unit = axes[1].get_window_extent().width / 64 * 72 / fig.dpi  # points per image pixel
    # Labels keep 3 pt from the centre of every box line (the lines are 1.5 pt wide).
    drawn, margin = gt["boxes"].tolist() + prediction["boxes"].tolist(), 3 / unit
    # The legend's place is taken, widened by the generous width of its widest text: browsers draw its
    # Chinese wider than matplotlib measured it.
    frame = legend.get_window_extent().transformed(axes[1].transData.inverted())
    wider = max(label_size(entry.get_text(), entry.get_fontsize())[0] for entry in legend.get_texts())
    taken = [(frame.xmin, frame.ymin, frame.xmax + wider / unit, frame.ymax)]
    assert not any(overlaps(taken[0], box, margin) for box in drawn), "the legend covers or touches a box"
    for box, cls, score in zip(prediction["boxes"].tolist(), prediction["labels"].tolist(),
                               prediction["scores"].tolist()):
        text = f"類別 {cls} score {score:.2f}"
        rect, ha = label_spot(box, [size / unit for size in label_size(text, 8)], drawn, taken, margin)
        taken.append(rect)
        axes[1].text(rect[0] if ha == "left" else rect[2], (rect[1] + rect[3]) / 2, text, ha=ha, va="center",
                     color="white", fontsize=8)
    # Files are written only now, after every check, and the record last: a run that fails before it
    # leaves the record out of date, so scripts/record_evidence.py runs it again.
    args.output.mkdir(parents=True, exist_ok=True)
    s = save_svg(fig, args.output / "learning.svg", "兩尺度模型 40 次更新的 loss 與實際預測框",
                 "左圖：fine 與 coarse 兩項相加的訓練 loss，每點在該次更新之前量；"
                 "右圖：40 次更新後同一張訓練圖上的 GT（綠虛線）與模型預測框（橙線），框旁標類別與 score")
    plt.close(fig)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.record:
        (ROOT / "docs/assets/diagrams/10-multiscale-learning.svg").write_text(s)
        (ROOT / "artifacts/checks/curriculum/10-multiscale-learning.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "loss_history"}, indent=2))


if __name__ == "__main__":
    main()
