"""Fit the lesson's two-scale network to one image, then decode actual outputs."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_learning_extensions import load
from miniyolo.targets import build_targets
from miniyolo.losses import grid_loss
from miniyolo.inference import decode_grid
from miniyolo.geometry import nms
from miniyolo.metrics import evaluate_ap


def main():
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
    report = {"torch": str(torch.__version__), "device": "cpu", "seed": 7, "steps": 40, "samples": 1,
              "optimizer": "Adam", "learning_rate": .01, "initial_loss": history[0], "last_pre_update_loss": history[-1],
              "loss_history": history, "parameters": sum(p.numel() for p in model.parameters()), "weight_delta_l2": delta,
              "training_image_metrics": metrics, "prediction": {k: v.tolist() for k, v in prediction.items()},
              "limits": "One training image, hand-chosen size responsibility; not held-out small-object AP and not a single-scale comparison"}
    destination = ROOT / "artifacts/checks/curriculum/10-multiscale-learning.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    axes[0].plot(range(1, 41), history)
    axes[0].set(xlabel="Training step (pre-update loss)", ylabel="Training loss")
    axes[1].imshow(image[0].permute(1, 2, 0).numpy(), extent=(0, 64, 64, 0))
    for group, color in ((gt, "lime"), (prediction, "orange")):
        for x1, y1, x2, y2 in group["boxes"].tolist():
            axes[1].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color))
    axes[1].set(xlim=(0, 64), ylim=(64, 0), title="Training image: green GT, orange predictions")
    path = ROOT / "docs/assets/diagrams/10-multiscale-learning.svg"
    fig.savefig(path)
    plt.close(fig)
    s = path.read_text().replace('<svg ', '<svg role="img" aria-label="Actual two-scale training curve and decoded predictions on the training image" ', 1)
    s = s.replace('version="1.1">', 'version="1.1"><title>Two-scale 40-step training and actual decoded boxes</title>', 1)
    path.write_text("\n".join(line.rstrip() for line in s.splitlines()) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "loss_history"}, indent=2))


if __name__ == "__main__":
    main()
