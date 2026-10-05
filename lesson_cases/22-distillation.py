"""真正更新 tiny ViT，並從中間 checkpoint 恢復到與連續訓練相同的狀態。"""
import json
from pathlib import Path
import time

import torch

from miniyolo.figures import save_svg, use_svg_text
from miniyolo.self_distillation import DINOTrainer, extract_features, feature_diagnostics
from miniyolo.vision_data import ColorRectangles


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def main(steps=160):
    if not 80 <= steps <= 200:
        raise ValueError("課堂 CPU 示範 steps 必須介於80與200")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    output = Path("artifacts/runs/dino")
    output.mkdir(parents=True, exist_ok=True)
    images = ColorRectangles(samples=128, seed=101).images  # labels/bboxes 不進 SSL
    trainer = DINOTrainer(seed=7)
    before = {name: tensor.clone() for name, tensor in trainer.student.backbone.state_dict().items()}
    start = time.perf_counter()
    midpoint = steps // 2
    history = trainer.train_steps(images, midpoint)
    checkpoint = output / "22-distillation-midpoint.pt"
    trainer.save_checkpoint(checkpoint)
    next_views = trainer.peek_next_views(images)
    continuation = trainer.train_steps(images, steps - midpoint)
    continuous_next_views = trainer.peek_next_views(images)
    resumed = DINOTrainer.load_checkpoint(checkpoint)
    assert all(torch.equal(a, b) for a, b in zip(next_views, resumed.peek_next_views(images)))
    resume_history = resumed.train_steps(images, steps - midpoint)
    assert continuation == resume_history
    for component in ("student", "teacher", "optimizer", "loss"):
        assert_same(getattr(trainer, component).state_dict(), getattr(resumed, component).state_dict())
    assert trainer.step == resumed.step == steps
    assert torch.equal(trainer.generator.get_state(), resumed.generator.get_state())
    assert all(torch.equal(a, b) for a, b in zip(continuous_next_views, resumed.peek_next_views(images)))
    elapsed = time.perf_counter() - start
    history += continuation
    backbone_changed = any(not torch.equal(before[name], tensor)
                           for name, tensor in trainer.student.backbone.state_dict().items() if not name.startswith("head."))
    assert backbone_changed and all(parameter.grad is None for parameter in trainer.teacher.parameters())
    features = extract_features(trainer.teacher.backbone, images)
    with torch.no_grad():
        probabilities = trainer.loss.teacher_probabilities(trainer.teacher(images))
    result = {"seed": 7, "device": "cpu", "steps": steps, "checkpoint_step": midpoint,
              "train_samples": len(images), "batch_size": trainer.config.batch_size,
              "backbone": trainer.backbone_config, "training_config": vars(trainer.config),
              "first_step": history[0], "last_step": history[-1], "history": history,
              "backbone_weights_changed": backbone_changed, "teacher_has_no_grad": True,
              "resume": {"next_random_views_identical": True, "student_teacher_optimizer_center_identical": True,
                         "loss_history_identical": True, "final_step": resumed.step},
              "diagnostics_train": feature_diagnostics(features, probabilities),
              "elapsed_training_and_resume_seconds": elapsed,
              "checkpoint_path": str(checkpoint),
              "simplifications": "2 global views; ordinary MLP and no weight normalization; fixed temperatures/LR/EMA; no multi-crop or official schedules",
              "limitation": "DINO2021 mechanism demonstration from scratch; loss/diagnostics are not downstream or natural-image capability proof"}
    trainer.save_checkpoint(output / "22-distillation-final.pt")
    (output / "22-distillation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    use_svg_text()
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot([row["step"] for row in history], [row["loss"] for row in history])
    axes[0].set(xlabel="student update step", ylabel="cross-view CE", title="實際160步：loss 不代表特徵用途")
    axes[1].plot([row["step"] for row in history], [row["teacher_entropy"] for row in history])
    axes[1].set(xlabel="student update step", ylabel="teacher output entropy", title="centered teacher 分佈的平均熵")
    figure.tight_layout()
    save_svg(figure, output / "22-distillation.svg", "DINO 短訓練實測", "跨 view CE 與 teacher entropy 隨更新步數變化")
    plt.close(figure)
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, ensure_ascii=False, indent=2))
    print(f"actual history: {output / '22-distillation.json'}")


if __name__ == "__main__":
    main()
