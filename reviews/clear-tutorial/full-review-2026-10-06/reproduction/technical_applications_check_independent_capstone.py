"""Independent read-only probes; never write curriculum pages or existing records."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from miniyolo.provenance import repo_dependencies, machine_info

torch.set_num_threads(2)
case_path = ROOT / "lesson_cases/17-capstone.py"
case = runpy.run_path(str(case_path))
images, targets = case["batch"](1100, 32)
val_images, val_targets = case["batch"](2200, 16)
test_images, test_targets = case["batch"](3300, 16)
original_step = torch.optim.Adam.step
result = []
models = []
for role, box_weight, steps in [("warmup", 5, 10), ("baseline", 5, 160), ("changed", 10, 160)]:
    plain, _ = case["fit"](images, targets, box_weight, steps=steps)
    before, after, norms, parameter_gradient_norms = [], [], [], []
    def checked_step(optimizer, *args, **kwargs):
        parameters = [p for g in optimizer.param_groups for p in g["params"]]
        if not before:
            before.extend(p.detach().clone() for p in parameters)
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
        norm = float(sum(p.grad.square().sum() for p in parameters).sqrt())
        assert torch.isfinite(torch.tensor(norm)) and norm > 0
        norms.append(norm)
        parameter_gradient_norms.append([float(p.grad.norm()) for p in parameters])
        returned = original_step(optimizer, *args, **kwargs)
        assert all(bool(torch.isfinite(p).all()) for p in parameters)
        after[:] = [p.detach().clone() for p in parameters]
        return returned
    torch.optim.Adam.step = checked_step
    try:
        observed, _ = case["fit"](images, targets, box_weight, steps=steps)
    finally:
        torch.optim.Adam.step = original_step
    assert len(norms) == steps
    assert all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), observed.state_dict().values()))
    delta = max(float((a-b).abs().max()) for a, b in zip(after, before))
    assert delta > 0
    result.append({"role": role, "steps": steps, "box_weight": box_weight,
                   "all_parameter_gradients_present_and_finite_every_step": True,
                   "combined_gradient_norm_finite_nonzero_every_step": True,
                   "all_weights_finite_after_every_step": True,
                   "gradient_l2_min": min(norms), "gradient_l2_max": max(norms),
                   "weight_max_abs_change": delta,
                   "observed_matches_plain_final_state_bitwise": True,
                   "gradient_l2_per_step": norms,
                   "zero_parameter_gradient_norm_occurrences": sum(v == 0 for row in parameter_gradient_norms for v in row),
                   "parameters_per_step": len(parameter_gradient_norms[0])})
    models.append(observed)
base_metrics, base_predictions = case["evaluate"](models[1], val_images, val_targets)
change_metrics, change_predictions = case["evaluate"](models[2], val_images, val_targets)
keep = change_metrics["map"] >= base_metrics["map"] + .01
test_metrics, _ = case["evaluate"](models[2] if keep else models[1], test_images, test_targets)
record = json.loads((ROOT / "artifacts/checks/curriculum/17-capstone.json").read_text())
normal = json.JSONDecoder().raw_decode(record["stdout"])[0]
assert normal["baseline_validation"] == json.loads(json.dumps(base_metrics))
assert normal["changed_validation"] == json.loads(json.dumps(change_metrics))
assert normal["chosen_test"] == json.loads(json.dumps(test_metrics))
assert normal["baseline_errors"] == case["error_diagnostics"](base_predictions, val_targets)
assert normal["changed_errors"] == case["error_diagnostics"](change_predictions, val_targets)
supplement = json.loads((ROOT / "reviews/clear-tutorial/full-review-2026-10-06/rechecks/capstone-updates.json").read_text())
for independent, saved in zip(result, supplement["runs"]):
    for field in ["role", "steps", "gradient_l2_min", "gradient_l2_max", "weight_max_abs_change", "gradient_l2_per_step"]:
        assert independent[field] == saved[field], field
report = {"executed_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
          "machine": machine_info(2), "device": "cpu", "source_hashes": repo_dependencies(case_path),
          "review_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "runs": result, "normal_curriculum_quality_and_error_metrics_match": True,
          "saved_supplement_gradient_arrays_and_weight_changes_match_exactly": True,
          "scope": "三次 fit 分別 10、160、160 步；每次另跑無 probe 對照並逐一比較最終 state_dict 位元相同。觀察只讀梯度與權重；未呼叫 main、未寫教材或既有實測紀錄；耗時不作效能比較。有限非零是每步全網路合併梯度範數，並非每個梯度元素非零。"}
target = ROOT / "reviews/clear-tutorial/full-review-2026-10-06/rechecks/technical-applications-capstone-independent.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({"passed": True, "runs": [{k:v for k,v in r.items() if k != "gradient_l2_per_step"} for r in result]}, ensure_ascii=False))
