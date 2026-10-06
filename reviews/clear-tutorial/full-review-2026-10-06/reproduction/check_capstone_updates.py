"""Add read-only gradient/update probes around the unchanged lesson-17 Adam steps.

Run from the repository root with .venv-model/bin/python. This supplementary CPU
check does not modify the lesson case or its frozen notebook; timing includes probes.
"""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys

import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from miniyolo.provenance import machine_info, repo_dependencies

case_path = ROOT / "lesson_cases/17-capstone.py"
case = runpy.run_path(str(case_path))
original_step = torch.optim.Adam.step
runs = {}


def checked_step(optimizer, *args, **kwargs):
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    key = optimizer  # Keep each optimizer alive: Python may reuse id() after fit returns.
    if key not in runs:
        runs[key] = {"before": [p.detach().clone() for p in parameters],
                     "gradient_norms": [], "learning_rate": optimizer.param_groups[0]["lr"]}
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    norm = float(sum(p.grad.square().sum() for p in parameters).sqrt())
    assert norm > 0
    runs[key]["gradient_norms"].append(norm)
    result = original_step(optimizer, *args, **kwargs)
    assert all(torch.isfinite(p).all() for p in parameters)
    runs[key]["after"] = [p.detach().clone() for p in parameters]
    return result


stdout = io.StringIO()
torch.optim.Adam.step = checked_step
try:
    with redirect_stdout(stdout):
        case["main"]()
finally:
    torch.optim.Adam.step = original_step

assert [len(run["gradient_norms"]) for run in runs.values()] == [10, 160, 160]
checked = []
for role, run in zip(["warmup", "baseline", "changed"], runs.values()):
    delta = max(float((a-b).abs().max()) for a, b in zip(run["after"], run["before"]))
    assert delta > 0
    checked.append({"role": role, "steps": len(run["gradient_norms"]), "finite_gradients_every_step": True,
                    "finite_weights_every_step": True,
                    "gradient_l2_min": min(run["gradient_norms"]),
                    "gradient_l2_max": max(run["gradient_norms"]),
                    "weight_max_abs_change": delta, "learning_rate": run["learning_rate"],
                    "gradient_l2_per_step": run["gradient_norms"]})
source = Path(__file__).resolve()
hashes = {**repo_dependencies(case_path),
          source.relative_to(ROOT).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest()}
report = {"executed_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
          "device": "cpu", "torch": str(torch.__version__), "machine": machine_info("2"),
          "source_hashes": hashes, "runs": checked, "instrumented_stdout": stdout.getvalue(),
          "scope": "Unchanged full lesson-17 main: ten-step warmup, then box weights 5 and 10 with the same actual data and 160-step fits. Probes observe gradients before Adam.step and parameter values after it. They do not alter model inputs, optimizer state or updates. Timing includes probes and is not used as a benchmark."}
target = ROOT / "reviews/clear-tutorial/full-review-2026-10-06/rechecks/capstone-updates.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in {"instrumented_stdout", "source_hashes", "runs"}}, ensure_ascii=False))
print(json.dumps([{k: v for k, v in run.items() if k != "gradient_l2_per_step"} for run in checked], ensure_ascii=False))
