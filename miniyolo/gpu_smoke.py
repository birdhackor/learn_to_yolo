"""Small real GridDetector training/resume experiment; no alternate probe model."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .checkpoint import capture_rng, load_checkpoint, save_checkpoint
from .data import ShapeDataset, collate
from .losses import grid_loss
from .models import GridDetector
from .targets import build_targets
from .train import CLASS_NAMES, optimizer_step


CONFIG = {"seed": 7, "train_seed": 7, "samples": 8, "batch_size": 8,
          "image_size": 64, "grid_size": 4, "width": 8, "num_classes": 2,
          "learning_rate": .01, "total_steps": 40, "midpoint_step": 20,
          "scheduler": {"name": "StepLR", "step_size": 20, "gamma": .8},
          "allow_empty": False}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def setup(device):
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.set_num_threads(2)
    # AdaptiveAvgPool2d CUDA backward can lack a deterministic implementation.
    # Keep the actual project architecture and measure agreement within tolerance.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("This smoke test requires exactly one visible CUDA GPU")
        if "L4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("The requested test is restricted to an NVIDIA L4")
        torch.cuda.manual_seed_all(CONFIG["seed"])
    dataset = ShapeDataset(n=CONFIG["samples"], size=64, seed=CONFIG["train_seed"], allow_empty=False)
    images, annotations = collate([dataset[index] for index in range(len(dataset))])
    digest = hashlib.sha256(images.numpy().tobytes())
    for annotation in annotations:
        for name in ("boxes", "labels"):
            digest.update(annotation[name].numpy().tobytes())
    images = images.to(device)
    annotations = [{key: value.to(device) for key, value in item.items()} for item in annotations]
    targets = build_targets(annotations)
    model = GridDetector(width=CONFIG["width"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **{key: CONFIG["scheduler"][key] for key in ("step_size", "gamma")})
    if any(p.device != device for p in model.parameters()) or images.device != device:
        raise RuntimeError("Model and images must be on the requested device")
    if any(value.device != device for value in targets.values()):
        raise RuntimeError("All encoded targets must be on the requested device")
    return {"model": model, "optimizer": optimizer, "scheduler": scheduler,
            "images": images, "targets": targets, "data_sha256": digest.hexdigest(), "device": device}


def fixed_loss(state):
    with torch.no_grad():
        return float(grid_loss(state["model"](state["images"]), state["targets"])["total"])


def random_probe(device):
    # These probes verify restored RNG streams; they do not modify images or loss.
    return [random.random(), float(np.random.random()), float(torch.rand(())),
            float(torch.rand((), device=device)) if device.type == "cuda" else None]


def run_steps(state, start, end):
    history = []
    synchronize(state["device"])
    begun = time.perf_counter()
    for index in range(start, end):
        lr = float(state["optimizer"].param_groups[0]["lr"])
        values, norm = optimizer_step(state["model"], state["optimizer"], state["images"],
                                      state["targets"], validate_gradients=True)
        state["scheduler"].step()
        history.append({"step": index + 1, **values, "gradient_l2": norm,
                        "learning_rate": lr, "rng_probe": random_probe(state["device"])})
        if (index + 1) % 10 == 0:
            print(json.dumps({"event": "grid_detector_optimizer_step", **history[-1]}), flush=True)
    synchronize(state["device"])
    elapsed = time.perf_counter() - begun
    if any(not math.isfinite(row["gradient_l2"]) or row["gradient_l2"] <= 0 for row in history):
        raise RuntimeError("Every optimizer step must have finite nonzero gradients")
    return history, elapsed


def compare(left, right, exact=False):
    maximum = 0.
    if isinstance(left, torch.Tensor):
        if exact or not left.is_floating_point():
            if not torch.equal(left.cpu(), right.cpu()):
                raise AssertionError("Checkpoint tensor/RNG state differs")
        else:
            torch.testing.assert_close(left.cpu(), right.cpu(), rtol=1e-5, atol=1e-6)
            maximum = float((left.cpu().double() - right.cpu().double()).abs().max()) if left.numel() else 0.
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError("Checkpoint keys differ")
        maximum = max([compare(left[key], right[key], exact) for key in left] or [0.])
    elif isinstance(left, (tuple, list)):
        if type(left) != type(right) or len(left) != len(right):
            raise AssertionError("Checkpoint sequence differs")
        maximum = max([compare(a, b, exact) for a, b in zip(left, right)] or [0.])
    elif isinstance(left, float) and not exact:
        if not math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-6):
            raise AssertionError("Checkpoint floating-point value differs beyond tolerance")
        maximum = abs(left - right)
    elif left != right:
        raise AssertionError("Checkpoint scalar/scheduler value differs")
    return maximum


def produce(directory, code_commit, device="cuda:0"):
    begun = time.perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    baseline = setup(device)
    initial_loss = fixed_loss(baseline)
    initial_weights = {name: value.detach().cpu().clone() for name, value in baseline["model"].state_dict().items()}
    baseline_history, baseline_seconds = run_steps(baseline, 0, CONFIG["total_steps"])
    final_loss = fixed_loss(baseline)
    if not final_loss < initial_loss * .8:
        raise RuntimeError(f"Fixed-data loss did not decrease reasonably: {initial_loss} -> {final_loss}")
    delta = math.sqrt(sum(float((value.detach().cpu() - initial_weights[name]).square().sum())
                          for name, value in baseline["model"].state_dict().items()))
    if not delta > 0:
        raise RuntimeError("Optimizer ran but detector weights did not change")
    saving = time.perf_counter()
    save_checkpoint(directory / "baseline-final.pt", baseline["model"], baseline["optimizer"],
                    CONFIG, CLASS_NAMES, CONFIG["total_steps"], baseline["scheduler"])
    baseline_save_seconds = time.perf_counter() - saving
    interrupted = setup(device)
    interrupted_history, interrupted_seconds = run_steps(interrupted, 0, CONFIG["midpoint_step"])
    compare(baseline_history[:CONFIG["midpoint_step"]], interrupted_history)
    compare([row["rng_probe"] for row in baseline_history[:20]],
            [row["rng_probe"] for row in interrupted_history], exact=True)
    saving = time.perf_counter()
    save_checkpoint(directory / "midpoint.pt", interrupted["model"], interrupted["optimizer"],
                    CONFIG, CLASS_NAMES, CONFIG["midpoint_step"], interrupted["scheduler"])
    midpoint_save_seconds = time.perf_counter() - saving
    write_json(directory / "model-config.json", {**CONFIG, "architecture": "miniyolo.models.GridDetector", "class_names": CLASS_NAMES})
    write_json(directory / "preprocessing.json", {"color": "RGB", "layout": "NCHW", "dtype": "float32", "range": [0, 1],
                                                "input_size": [64, 64], "boxes": "half-open pixel xyxy",
                                                "tokenizer": None, "tokenizer_note": "Vision model; no text tokenizer"})
    write_json(directory / "data-source.json", {"source": "miniyolo.data.ShapeDataset", "synthetic": True,
                                              "download_required": False, "seed": 7, "samples": 8, "allow_empty": False,
                                              "data_sha256": baseline["data_sha256"]})
    write_json(directory / "provenance.json", {"repository": "birdhackor/learn_to_yolo", "code_commit": code_commit,
                                             "created_at": datetime.now(timezone.utc).isoformat()})
    result = {"stage": "produce", "code_commit": code_commit, "config": CONFIG, "device": str(device),
              "gpu_verified": device.type == "cuda", "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "torch_version": str(torch.__version__), "cuda_build": torch.version.cuda,
              "parameter_count": sum(p.numel() for p in baseline["model"].parameters()),
              "model_images_targets_on_device": True, "initial_fixed_loss": initial_loss, "final_fixed_loss": final_loss,
              "weight_delta_l2": delta, "finite_nonzero_gradients_every_step": True,
              "baseline_history": baseline_history, "interrupted_history": interrupted_history,
              "training_seconds": {"baseline_40_steps": baseline_seconds, "interrupted_20_steps": interrupted_seconds},
              "checkpoint_save_seconds": {"baseline": baseline_save_seconds, "midpoint": midpoint_save_seconds},
              "timing_definition": "Training timers synchronize CUDA at both ends; include forward/backward/optimizer/scheduler, gradient validation, RNG probes, scalar collection and console logging. Exclude setup, checkpoint save/Volume commit, container startup/image build and network transfers.",
              "files": {name: sha256(directory / name) for name in ["midpoint.pt", "baseline-final.pt", "model-config.json", "preprocessing.json", "data-source.json", "provenance.json"]}}
    result["function_body_seconds_before_volume_commit"] = time.perf_counter() - begun
    write_json(directory / "producer.json", result)
    return result


def resume(directory, checkpoint_name="midpoint.pt", device="cuda:0"):
    begun = time.perf_counter()
    directory = Path(directory)
    produced = json.loads((directory / "producer.json").read_text())
    device = torch.device(device)
    state = setup(device)
    restored = load_checkpoint(directory / checkpoint_name, state["model"], state["optimizer"], state["scheduler"])
    if restored["config"] != CONFIG or state["data_sha256"] != json.loads((directory / "data-source.json").read_text())["data_sha256"]:
        raise RuntimeError("Model/data configuration changed across resume")
    if restored["steps_completed"] != 20:
        raise RuntimeError("Expected the actual step-20 intermediate checkpoint")
    learning_rate_restored = float(state["optimizer"].param_groups[0]["lr"])
    history, elapsed = run_steps(state, restored["steps_completed"], CONFIG["total_steps"])
    expected = torch.load(directory / "baseline-final.pt", map_location="cpu", weights_only=True)
    model_error = compare(state["model"].state_dict(), expected["model_state_dict"])
    optimizer_error = compare(state["optimizer"].state_dict(), expected["optimizer_state_dict"])
    compare(state["scheduler"].state_dict(), expected["scheduler_state_dict"], exact=True)
    compare(capture_rng(), expected["rng_state"], exact=True)
    history_error = compare(history, produced["baseline_history"][20:])
    compare([row["rng_probe"] for row in history],
            [row["rng_probe"] for row in produced["baseline_history"][20:]], exact=True)
    optimizer_steps = [int(item["step"]) for item in state["optimizer"].state.values()]
    if not optimizer_steps or set(optimizer_steps) != {40}:
        raise RuntimeError("Adam optimizer state did not restore/continue to step 40")
    final_loss = fixed_loss(state)
    saving = time.perf_counter()
    save_checkpoint(directory / "resumed-final.pt", state["model"], state["optimizer"], CONFIG, CLASS_NAMES, 40, state["scheduler"])
    result = {"stage": "resume", "checkpoint_used": checkpoint_name, "checkpoint_sha256": sha256(directory / checkpoint_name),
              "restored_step": 20, "steps_completed": 40, "additional_optimizer_steps": 20,
              "optimizer_step_min_max": [min(optimizer_steps), max(optimizer_steps)],
              "learning_rate_restored": learning_rate_restored, "learning_rate_final": float(state["optimizer"].param_groups[0]["lr"]),
              "final_fixed_loss": final_loss, "finite_nonzero_gradients_every_step": True,
              "model_max_abs_error": model_error, "optimizer_max_abs_error": optimizer_error,
              "scheduler_matches": True, "rng_states_match": True, "loss_and_rng_trace_matches": True,
              "history_max_abs_error": history_error,
              "comparison_tolerance": {"rtol": 1e-5, "atol": 1e-6}, "training_seconds": elapsed,
              "checkpoint_save_seconds": time.perf_counter() - saving, "history": history,
              "resumed_final_sha256": sha256(directory / "resumed-final.pt"),
              "function_body_seconds_before_volume_commit": time.perf_counter() - begun}
    write_json(directory / "resume.json", result)
    return result
