"""獨立從零 SSL，再凍結 teacher：1-NN／linear probe 與同起點 random baseline。"""
import json
from pathlib import Path
import time

import torch

from miniyolo.figures import save_svg, use_svg_text
from miniyolo.self_distillation import (DINOTrainer, extract_features, feature_diagnostics,
                                       linear_probe, nearest_neighbor_accuracy)
from miniyolo.vision_data import make_color_splits


def main(steps=160, probe_steps=120):
    torch.set_num_threads(2)
    data = make_color_splits()  # train101/val202/test303；每張只有一個紅或藍矩形
    start = time.perf_counter()
    trainer = DINOTrainer(seed=7)
    history = trainer.train_steps(data["train"].images, steps)
    result = {"ssl_steps": steps, "probe_steps": probe_steps, "ssl_seed": 7,
              "split": {name: {"samples": len(split), "seed": split.seed} for name, split in data.items()},
              "task": "red vs blue rectangle color, both classes share position/size generation",
              "feature_source": "frozen teacher CLS [N,32]; no projection head in downstream",
              "ssl_first_loss": history[0]["loss"], "ssl_last_loss": history[-1]["loss"],
              "controls": "same data, TinyViT initial backbone, probe head seed900, optimizer/lr/120steps; train-only feature standardization",
              "validation_use": "report only; no hyperparameter selection", "test_use": "evaluation only",
              "models": {}}
    backbones = {"ssl_teacher": trainer.teacher.backbone, "random_baseline": trainer.random_baseline()}
    for name, backbone in backbones.items():
        backbone.requires_grad_(False).eval()
        before = {key: value.clone() for key, value in backbone.state_dict().items()}
        features = {split: extract_features(backbone, dataset.images) for split, dataset in data.items()}
        nn_scores = {}
        for split in ("val", "test"):
            score, indices = nearest_neighbor_accuracy(features["train"], data["train"].labels,
                                                       features[split], data[split].labels)
            nn_scores[split] = {"accuracy": score, "correct": round(score * len(data[split])),
                                "count": len(data[split]), "first_query_neighbor_train_index": int(indices[0]),
                                "first_query_class": data[split].class_names[int(data[split].labels[0])],
                                "first_neighbor_class": data["train"].class_names[int(data["train"].labels[indices[0]])]}
        probe = linear_probe(features["train"], data["train"].labels,
                             features["val"], data["val"].labels,
                             features["test"], data["test"].labels, steps=probe_steps, seed=900)
        diagnostics = feature_diagnostics(features["test"])
        if name == "ssl_teacher":
            with torch.no_grad():
                probabilities = trainer.loss.teacher_probabilities(trainer.teacher(data["test"].images))
            diagnostics = feature_diagnostics(features["test"], probabilities)
        frozen = all(torch.equal(before[key], value) for key, value in backbone.state_dict().items())
        assert frozen and all(parameter.grad is None for parameter in backbone.parameters())
        result["models"][name] = {"nearest_neighbor": nn_scores, "linear_probe": probe,
                                 "diagnostics_test": diagnostics, "backbone_unchanged": frozen,
                                 "backbone_has_no_grad": True}
    result["elapsed_seconds"] = time.perf_counter() - start
    result["limitation"] = "a simple color task may already be solved by random features; equal accuracy is not evidence of SSL improvement or natural-image transfer"
    output = Path("artifacts/runs/dino")
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(output / "22-features-ssl.pt")
    (output / "22-features.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    use_svg_text()
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(8.5, 3.4))
    images = [data["test"].images[0]]
    titles = [f"test 0: {data['test'].class_names[int(data['test'].labels[0])]}（query）"]
    for name in ("ssl_teacher", "random_baseline"):
        nearest = result["models"][name]["nearest_neighbor"]["test"]
        index = nearest["first_query_neighbor_train_index"]
        images.append(data["train"].images[index])
        titles.append(f"{name}\ntrain {index}: {nearest['first_neighbor_class']}")
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image.permute(1, 2, 0).numpy())
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("實際 cosine 1-NN：reference bank 只有 train 圖，test query 不在 bank 裡")
    figure.tight_layout()
    save_svg(figure, output / "22-features.svg", "CLS 特徵的實際近鄰", "測試 query 與 SSL 及 random backbone 找到的訓練圖近鄰")
    plt.close(figure)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
