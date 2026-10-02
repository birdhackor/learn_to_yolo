"""Resume-safe checkpoints, retaining the original GridDetector field names."""

from pathlib import Path
import random

import numpy as np
import torch


def capture_rng():
    name, keys, position, has_gaussian, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"name": name, "keys": torch.from_numpy(keys.copy()),
                  "position": position, "has_gaussian": has_gaussian,
                  "cached_gaussian": cached_gaussian},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy["name"], numpy["keys"].cpu().numpy(),
                         numpy["position"], numpy["has_gaussian"], numpy["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda = state["torch_cuda"]
    if cuda:
        if not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count():
            raise ValueError("Exact RNG resume needs the same number of CUDA devices")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda])


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def save_checkpoint(path, model, optimizer, config, class_names, steps_completed,
                    scheduler=None):
    """No Python class instances: compatible with torch.load(weights_only=True)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 2,
        "model_state_dict": _cpu_tree(model.state_dict()),
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
        "config": dict(config), "class_names": list(class_names),
        "steps_completed": int(steps_completed),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": capture_rng(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return state


def load_checkpoint(path, model, optimizer, scheduler=None):
    """Restore optimizer/scheduler before RNG; caller supplies the same architecture."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 2 or "rng_state" not in state:
        raise ValueError("This legacy checkpoint supports inference, not exact RNG resume")
    if (state["scheduler_state_dict"] is None) != (scheduler is None):
        raise ValueError("Checkpoint and caller must agree about the learning-rate scheduler")
    model.load_state_dict(state["model_state_dict"], strict=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    restore_rng(state["rng_state"])
    return state
