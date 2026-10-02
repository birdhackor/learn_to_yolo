"""Short learning extensions using the exact lesson models, no pretrained weights."""
import argparse
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/miniyolo-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/miniyolo-cache")
import importlib.util
import json
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


def run(lesson):
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
        history, gradients = [], []
        for step in range(40):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(images)
            if targets is None:
                loss = nn.functional.cross_entropy(prediction, labels)
            else:
                loss = nn.functional.cross_entropy(prediction[0], labels) + 5 * nn.functional.mse_loss(prediction[1], targets)
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
        results[name] = result
    report = {"lesson": lesson, "seed": 7, "torch": str(torch.__version__), "device": "cpu", "models": results,
              "limits": "One fixed tiny synthetic batch, no held-out quality except the 4 shifted ResNet examples; not a version ranking"}
    folder = ROOT / "artifacts/checks/curriculum"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{lesson}-learning.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    for name, result in results.items():
        ax.plot(range(1, 41), result["loss_history"], label=name)
    ax.set(xlabel="Training step (pre-update loss)", ylabel="Training loss", title="Fixed synthetic batch, seed 7; 40 updates")
    ax.legend()
    target = ROOT / f"docs/assets/diagrams/{lesson}-learning.svg"
    fig.savefig(target)
    plt.close(fig)
    svg = target.read_text().replace('<svg ', '<svg role="img" aria-label="Actual fixed-batch training losses across 40 optimizer updates" ', 1)
    svg = svg.replace('version="1.1">', 'version="1.1"><title>40-step fixed-batch learning evidence</title>', 1)
    target.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=["01-small-cnn", "03-comparison", "04-localization"], required=True)
    args = parser.parse_args()
    result = run(args.section)
    print(json.dumps({**result, "models": {n: {k: v for k, v in r.items() if k != "loss_history"}
                                          for n, r in result["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
