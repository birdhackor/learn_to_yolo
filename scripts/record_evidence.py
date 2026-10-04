"""List, and on request regenerate, exactly the execution records whose code has changed.

Every record names, by SHA-256, the code that produced it: the lesson case or script plus every
repository module it imports. A record stays valid until any of that code changes, so a release only
re-runs what actually changed; unchanged records keep their date and machine.

    python scripts/record_evidence.py                       # list out-of-date records, run nothing
    python scripts/record_evidence.py --run                 # regenerate them on this machine
    python scripts/record_evidence.py --run --push BRANCH   # ... then commit the new records and push
    python scripts/record_evidence.py --run --bundle FILE   # ... or pack them into FILE (.tar.gz)
    python scripts/record_evidence.py --gpu BRANCH          # rerun out-of-date GPU records on Modal

Run it on the machine that made the existing records when possible (see docs/status.md), so that
computations that did not change reproduce their numbers. Each new record stores its machine.
GPU records come from manually started GitHub Actions workflows that run on one Modal L4; --gpu starts
them for BRANCH (pushed, at this checkout's commit) with the GitHub CLI and saves each run's result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from evidence_records import CPU_RECORDS, GPU_RECORDS, is_current, provenance

ROOT = Path(__file__).resolve().parents[1]

PY = sys.executable
C = "artifacts/checks/curriculum/"
# Commands that regenerate each supplementary record, in the order they must run.
COMMANDS = {
    "artifacts/checks/grid-learning.json": [
        [PY, "-m", "miniyolo.train", "--steps", "160", "--samples", "32", "--device", "cpu",
         "--report", "artifacts/checks/grid-learning.json"],
        [PY, "scripts/render_learning_evidence.py"]],
    C + "custom-data-160-step.json": [
        [PY, "scripts/run_custom_data_learning.py", "--fixture", "--steps", "160",
         "--output", "artifacts/runs/custom-data-learning",
         "--report", C + "custom-data-160-step.json", "--diagram", "docs/assets/diagrams/08-custom-160-step.svg"]],
    C + "custom-data-learning.json": [
        [PY, "scripts/run_custom_data_learning.py", "--fixture", "--steps", "1600", "--fixture-test-seed", "7001",
         "--output", "artifacts/runs/custom-data-learning-1600", "--prior-diagnostic", C + "custom-data-160-step.json",
         "--report", C + "custom-data-learning.json", "--diagram", "docs/assets/diagrams/08-custom-learning.svg"]],
    C + "01-small-cnn-learning.json": [[PY, "scripts/run_learning_extensions.py", "--section", "01-small-cnn", "--record"]],
    C + "03-comparison-learning.json": [[PY, "scripts/run_learning_extensions.py", "--section", "03-comparison", "--record"]],
    C + "04-localization-learning.json": [[PY, "scripts/run_learning_extensions.py", "--section", "04-localization", "--record"]],
    C + "10-multiscale-learning.json": [[PY, "scripts/run_multiscale_learning.py", "--record"]],
    C + "video-file.json": [[PY, "scripts/verify_video_file.py", "--record", C + "video-file.json"]],
    # Downloads Fashion-MNIST (30.88 MB, checked against data/manifest.json) unless data/downloads/ has it.
    C + "fashion-mnist-learning.json": [
        [PY, "scripts/download_data.py", "fetch", "fashion-mnist"],
        [PY, "scripts/run_fashion_cnn.py", "--train-steps", "40", "--subset", "64", "--eval-samples", "128", "--record"]],
}
assert set(COMMANDS) == set(CPU_RECORDS)
PINS = ["requirements-model.txt", "requirements-video.txt"]
OUTPUTS = ["artifacts/checks", "docs/assets/diagrams", "docs/lessons", "notebooks"]


def out_of_date() -> list[tuple[str, list]]:
    """(name, commands) to run, in order."""
    stale = subprocess.run([PY, "scripts/verify_curriculum.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    jobs = []
    sections = [line.split(":")[0] for line in stale.stdout.splitlines() if line.endswith("no current record")]
    if sections:
        jobs.append(("lesson records: " + ", ".join(sections), [[PY, "scripts/verify_curriculum.py"]]))
    names = [name for name in COMMANDS if not is_current(name)]
    # The 1600-step follow-up is declared against the 160-step diagnostic: regenerate both together.
    if any("custom-data" in name for name in names):
        names = [n for n in names if "custom-data" not in n] + [C + "custom-data-160-step.json", C + "custom-data-learning.json"]
    jobs += [(name, COMMANDS[name]) for name in COMMANDS if name in names]
    grid = "artifacts/checks/grid-learning.json"
    if grid not in names and grid_figures_differ():
        jobs.append(("grid-learning figures (renderer changed)", COMMANDS[grid][1:]))
    return jobs


def grid_figures_differ() -> bool:
    """The renderer is not part of the grid record's code, so compare its current output with the site figures."""
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run([PY, "scripts/render_learning_evidence.py", "--output", folder], cwd=ROOT, check=True,
                       capture_output=True)
        return any((Path(folder) / name).read_bytes() != (ROOT / "docs/assets/diagrams" / name).read_bytes()
                   for name in ("grid-learning-curve.svg", "grid-learning-predictions.svg"))


def check_pins() -> list[str]:
    problems = []
    if sys.version_info[:2] != (3, 12):
        problems.append(f"Python {sys.version.split()[0]} (expected 3.12)")
    for name in PINS:
        for line in (ROOT / name).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if "==" not in line:
                continue
            package, wanted = line.split("==")
            try:
                found = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                problems.append(f"{package} missing (expected {wanted})")
                continue
            if found.split("+")[0] != wanted:
                problems.append(f"{package} {found} (expected {wanted})")
    return problems


def run_gpu(branch: str) -> None:
    """Start each out-of-date GPU workflow on BRANCH, wait for it, and save its result as the record."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    remote = subprocess.run(["git", "ls-remote", "origin", f"refs/heads/{branch}"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.split()
    if not remote or remote[0] != head:
        raise SystemExit(f"STOP: push this commit to origin/{branch} first; the workflow runs the pushed code")
    for name, (_, workflow) in GPU_RECORDS.items():
        if is_current(name):
            continue
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        subprocess.run(["gh", "workflow", "run", workflow, "--ref", branch], cwd=ROOT, check=True)
        run_id = None
        while run_id is None:
            time.sleep(10)
            runs = json.loads(subprocess.run(
                ["gh", "run", "list", "--workflow", workflow, "--branch", branch, "--event", "workflow_dispatch",
                 "--limit", "5", "--json", "databaseId,headSha,createdAt"],
                cwd=ROOT, capture_output=True, text=True, check=True).stdout)
            run_id = next((r["databaseId"] for r in runs if r["headSha"] == head and r["createdAt"] >= started[:19]), None)
        print(f"\n== {name}: {workflow} run {run_id}", flush=True)
        watched = subprocess.run(["gh", "run", "watch", str(run_id), "--exit-status", "--interval", "30"], cwd=ROOT)
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run(["gh", "run", "download", str(run_id), "--dir", folder], cwd=ROOT, check=True)
            results = list(Path(folder).rglob("result.json"))
            assert len(results) == 1, results
            shutil.copyfile(results[0], ROOT / name)
        if watched.returncode:
            raise SystemExit(f"STOP: {workflow} run {run_id} failed; its result was saved to {name} for inspection")
    subprocess.run([sys.executable, "scripts/validate_curriculum_evidence.py", "--scope", "gpu"], cwd=ROOT, check=True)


def changed_files() -> list[str]:
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *OUTPUTS],
                            cwd=ROOT, capture_output=True, text=True)
    return [line[3:] for line in status.stdout.splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="regenerate the out-of-date records")
    parser.add_argument("--push", metavar="BRANCH", help="after --run, commit the new records and push to origin/BRANCH")
    parser.add_argument("--bundle", metavar="FILE", type=Path, help="after --run, pack the changed files into FILE (.tar.gz)")
    parser.add_argument("--gpu", metavar="BRANCH", help="rerun the out-of-date GPU records through GitHub Actions on BRANCH")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    args = parser.parse_args()
    if args.gpu:
        return run_gpu(args.gpu)
    gpu = [name for name in GPU_RECORDS if not is_current(name)]
    if gpu:
        print("GPU records out of date (rerun with --gpu BRANCH): " + ", ".join(gpu))
    jobs = out_of_date()
    print("Machine:", json.dumps(provenance.machine_info(2), ensure_ascii=False))
    if not jobs:
        print("Every CPU record matches the current code; nothing to run on this machine.")
        return
    print("Out of date:\n" + "\n".join(f"  - {name}" for name, _ in jobs))
    if not args.run:
        return
    problems = check_pins()
    if problems and not args.allow_version_mismatch:
        raise SystemExit("STOP: installed versions differ from the pins: " + "; ".join(problems))
    for name, commands in jobs:
        for command in commands:
            print(f"\n== {name}: {' '.join(command[1:])}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([PY, "scripts/validate_curriculum_evidence.py", "--scope", "cpu"], cwd=ROOT, check=True)
    left = out_of_date()
    if left:
        raise SystemExit("STOP: still out of date: " + ", ".join(name for name, _ in left))
    files = changed_files()
    print("\nChanged files:\n" + "\n".join(f"  {f}" for f in files))
    if args.bundle:
        import tarfile
        with tarfile.open(args.bundle, "w:gz") as archive:
            for name in files:
                archive.add(ROOT / name, arcname=name)
        print(f"Packed {len(files)} files into {args.bundle}")
    if args.push:
        subprocess.run(["git", "add", "--", *OUTPUTS], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-m", "Record execution evidence for the changed code"], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", f"HEAD:refs/heads/{args.push}"], cwd=ROOT, check=True)
        print(f"Pushed to origin/{args.push}")


if __name__ == "__main__":
    main()
