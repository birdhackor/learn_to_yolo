"""真實更新完整小ViT，評估獨立seed資料，並核對包含RNG的checkpoint續訓。"""

import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from miniyolo.checkpoint import capture_rng, load_checkpoint, save_checkpoint
from miniyolo.provenance import machine_info, repo_dependencies
from miniyolo.vision_data import make_color_splits
from miniyolo.vision_transformer import TinyViT


MODEL_CONFIG = {"image_size": 32, "patch_size": 8, "embed_dim": 32, "depth": 2,
                "num_heads": 4, "mlp_ratio": 2, "num_classes": 2, "dropout": .1}


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    logits = model(dataset.images)
    correct = int((logits.argmax(-1) == dataset.labels).sum())
    return {"correct": correct, "count": len(dataset), "accuracy": correct / len(dataset),
            "loss": float(F.cross_entropy(logits, dataset.labels))}


def tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def tree_max_error(left, right):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape:
            raise ValueError("比較的tensor結構不同")
        return float((left.double() - right.double()).abs().max()) if left.numel() else 0.
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError("比較的dict結構不同")
        return max((tree_max_error(left[key], right[key]) for key in left), default=0.)
    if isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError("比較的sequence結構不同")
        return max((tree_max_error(a, b) for a, b in zip(left, right)), default=0.)
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return abs(left - right)
    if left != right:
        raise ValueError("比較的非數值欄位不同")
    return 0.


def optimizer_steps(optimizer):
    counts = [int(state["step"]) for state in optimizer.state.values()]
    return [min(counts), max(counts)]


def train_step(model, optimizer, dataset, batch_size):
    model.train()
    # batch抽樣與dropout都消耗torch RNG；只恢復權重不能重現下一步。
    indices = torch.randint(len(dataset), (batch_size,))
    loss = F.cross_entropy(model(dataset.images[indices]), dataset.labels[indices])
    if not torch.isfinite(loss):
        raise RuntimeError("training loss 非有限值")
    optimizer.zero_grad()
    loss.backward()
    parameters = dict(model.named_parameters())
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters.values()):
        raise RuntimeError("有缺失或非有限的gradient")
    grad_norm = float(sum(parameter.grad.square().sum() for parameter in parameters.values()).sqrt())
    if not grad_norm > 0:
        raise RuntimeError("gradient全部為0")
    groups = {
        "patch_projection": float(model.patch_embed.projection.weight.grad.norm()),
        "first_qkv": float(model.blocks[0].attention.qkv.weight.grad.norm()),
        "first_mlp": float(model.blocks[0].mlp[0].weight.grad.norm()),
        "cls": float(model.cls_token.grad.norm()), "position": float(model.pos_embed.grad.norm()),
        "head": float(model.head.weight.grad.norm()),
    }
    optimizer.step()
    return float(loss.detach()), grad_norm, groups


def run_experiment(steps=60, checkpoint_step=30, batch_size=32,
                   output="artifacts/runs/vit-training"):
    if steps < 2 or not 1 <= checkpoint_step < steps or batch_size < 1:
        raise ValueError("需要1<=checkpoint_step<steps及正batch_size")
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    random.seed(7)
    np.random.seed(7)
    splits = make_color_splits()
    model = TinyViT(**MODEL_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    initial = copy.deepcopy(model.state_dict())
    train_before = evaluate(model, splits["train"])
    validation_before = evaluate(model, splits["val"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    history, norms, first_groups = [], [], None
    saved_state = None
    for step in range(1, steps + 1):
        loss, norm, groups = train_step(model, optimizer, splits["train"], batch_size)
        history.append({"step": step, "loss": loss})
        norms.append(norm)
        if first_groups is None:
            first_groups = groups
        if step == checkpoint_step:
            saved_state = save_checkpoint(checkpoint_path, model, optimizer, MODEL_CONFIG,
                                          splits["train"].class_names, step)
    train_after = evaluate(model, splits["train"])
    heldout = {name: evaluate(model, splits[name]) for name in ("val", "test")}
    continuous_model = copy.deepcopy(model.state_dict())
    continuous_optimizer = copy.deepcopy(optimizer.state_dict())
    continuous_rng = capture_rng()
    # 重新建立模型會消耗RNG；load_checkpoint最後恢復RNG，故仍能接回中點。
    resumed_model = TinyViT(**MODEL_CONFIG)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=.9, weight_decay=.5)
    restored = load_checkpoint(checkpoint_path, resumed_model, resumed_optimizer)
    restored_steps = optimizer_steps(resumed_optimizer)
    restored_learning_rate = resumed_optimizer.param_groups[0]["lr"]
    resumed_history = []
    for step in range(restored["steps_completed"] + 1, steps + 1):
        loss, _, _ = train_step(resumed_model, resumed_optimizer, splits["train"], batch_size)
        resumed_history.append({"step": step, "loss": loss})
    resumed_rng = capture_rng()
    model_error = tree_max_error(continuous_model, resumed_model.state_dict())
    optimizer_error = tree_max_error(continuous_optimizer, resumed_optimizer.state_dict())
    losses_match = tree_equal(history[checkpoint_step:], resumed_history)
    rng_match = tree_equal(continuous_rng, resumed_rng)
    if model_error != 0 or optimizer_error != 0 or not losses_match or not rng_match:
        raise RuntimeError("continuation與resume不一致")
    changed = {"patch_projection": not torch.equal(initial["patch_embed.projection.weight"], continuous_model["patch_embed.projection.weight"]),
               "first_qkv": not torch.equal(initial["blocks.0.attention.qkv.weight"], continuous_model["blocks.0.attention.qkv.weight"]),
               "first_mlp": not torch.equal(initial["blocks.0.mlp.0.weight"], continuous_model["blocks.0.mlp.0.weight"]),
               "cls": not torch.equal(initial["cls_token"], continuous_model["cls_token"]),
               "position": not torch.equal(initial["pos_embed"], continuous_model["pos_embed"]),
               "head": not torch.equal(initial["head.weight"], continuous_model["head.weight"])}
    if not all(changed.values()):
        raise RuntimeError("ViT有一組主要權重未更新")
    result = {
        "event": "vit_training", "device": "cpu", "seed": 7, "model_config": MODEL_CONFIG,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer": "AdamW", "learning_rate": .003, "weight_decay": .01,
        "steps": steps, "batch_size": batch_size,
        "split": {name: {"samples": len(dataset), "seed": dataset.seed} for name, dataset in splits.items()},
        "class_names": list(splits["train"].class_names), "answer_rule": "0=red rectangle; 1=blue rectangle; both classes share shape/position rules",
        "fixed_train_loss_before": train_before["loss"], "fixed_train_loss_after": train_after["loss"],
        "train_before": train_before, "train_after": train_after,
        "validation_before": validation_before, "heldout": heldout,
        "loss_history": history, "all_gradients_finite": True,
        "gradient_norm_min_max": [min(norms), max(norms)], "first_step_gradient_norms": first_groups,
        "weights_changed": changed, "optimizer_step_min_max": optimizer_steps(optimizer),
        "checkpoint": {"path": checkpoint_path.as_posix(), "keys": sorted(saved_state),
                       "steps_completed": restored["steps_completed"],
                       "restored_optimizer_step_min_max": restored_steps,
                       "restored_learning_rate": restored_learning_rate,
                       "rng_sources": sorted(restored["rng_state"])},
        "resume": {"additional_steps": steps - checkpoint_step,
                   "model_max_abs_error": model_error, "optimizer_max_abs_error": optimizer_error,
                   "loss_history_matches": losses_match, "rng_states_match": rng_match,
                   "optimizer_step_min_max": optimizer_steps(resumed_optimizer)},
        "accuracy_definition": "argmax class correct / count, evaluated in eval() without dropout",
        "elapsed_seconds": time.perf_counter() - started,
        "timing_scope": "data/model construction, continuous training, evaluation, checkpoint IO and resumed suffix; excludes Python import/startup",
        "machine": machine_info(2), "source_hashes": repo_dependencies(__file__),
        "limitation": "Only this controlled color classification distribution; no natural-image, localization, or ViT-vs-CNN performance claim.",
    }
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/runs/vit-training")
    args = parser.parse_args()
    result = run_experiment(output=args.output)
    assert result["fixed_train_loss_after"] < result["fixed_train_loss_before"]
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
