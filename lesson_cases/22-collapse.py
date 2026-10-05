"""手工固定輸出的反例：跨 view CE 很低，仍可能完全無法區分圖片。"""
import json
from pathlib import Path

import torch

from miniyolo.self_distillation import DINOLoss, feature_diagnostics


def main():
    torch.set_num_threads(2)
    count, output_dim = 8, 16
    constant_features = torch.ones(count, 32)
    loss = DINOLoss(output_dim, student_temperature=1., teacher_temperature=1.)
    uniform = torch.zeros(count, output_dim)
    peaked = torch.full((count, output_dim), -10.)
    peaked[:, 0] = 10.
    results = {}
    for name, logits in (("constant_uniform", uniform), ("constant_peaked", peaked)):
        value = loss([logits, logits], [logits, logits])
        probabilities = loss.teacher_probabilities(logits)
        results[name] = {"cross_view_ce": float(value),
                         **feature_diagnostics(constant_features, probabilities)}
    # 相同 features 的 deterministic 分類器只會給所有圖同一答案；平衡兩類只對4/8。
    balanced_labels = torch.arange(count) % 2
    constant_predictions = torch.zeros(count, dtype=torch.long)
    results["constant_feature_balanced_color_accuracy"] = float((constant_predictions == balanced_labels).float().mean())
    results["constant_feature_balanced_color_correct"] = int((constant_predictions == balanced_labels).sum())
    results["constant_feature_balanced_color_count"] = count
    results["construction"] = "hand-written outputs, no learned model and no optimizer"
    results["interpretation"] = "low cross-view CE alone does not establish useful features; entropy/std/cosine describe different failures"
    results["centering_limit"] = "not an ablation: this does not show that disabling centering immediately collapses every training run"
    assert results["constant_peaked"]["cross_view_ce"] < 1e-5
    assert results["constant_peaked"]["feature_std"] == 0
    output = Path("artifacts/runs/dino")
    output.mkdir(parents=True, exist_ok=True)
    (output / "22-collapse.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
