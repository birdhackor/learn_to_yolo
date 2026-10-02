"""Actual uninterrupted-vs-resumed training, including scheduler and four RNG streams."""

import json
from pathlib import Path

import pytest
import torch

from miniyolo.checkpoint import load_checkpoint
from miniyolo.gpu_smoke import produce, resume, setup
from scripts.detect_image import load_grid_checkpoint


def test_project_training_checkpoint_and_resume(tmp_path):
    directory = tmp_path / "run"
    produced = produce(directory, "test-commit", device="cpu")
    restored = resume(directory, device="cpu")
    assert produced["gpu_verified"] is False  # A CPU precursor is not GPU evidence.
    assert produced["final_fixed_loss"] < produced["initial_fixed_loss"] * .8
    assert restored["optimizer_step_min_max"] == [40, 40]
    assert restored["learning_rate_restored"] == pytest.approx(.008)
    assert restored["learning_rate_final"] == pytest.approx(.0064)
    assert restored["model_max_abs_error"] == restored["optimizer_max_abs_error"] == 0
    assert restored["scheduler_matches"] and restored["rng_states_match"]
    # The existing inference consumer must still accept the extended checkpoint.
    model, config, names = load_grid_checkpoint(directory / "midpoint.pt")
    assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 4, 4, 7)
    assert names == ["red rectangle", "blue rectangle"]
    assert config["width"] == 8
    json.dumps(produced, allow_nan=False)
    json.dumps(restored, allow_nan=False)


def test_legacy_checkpoint_is_not_claimed_to_restore_rng(tmp_path):
    state = setup(torch.device("cpu"))
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": state["model"].state_dict(), "config": {"image_size": 64, "grid_size": 4, "width": 8},
                "class_names": ["red rectangle", "blue rectangle"]}, path)
    load_grid_checkpoint(path)  # Legacy inference remains supported.
    with pytest.raises(ValueError, match="legacy checkpoint"):
        load_checkpoint(path, state["model"], state["optimizer"], state["scheduler"])
