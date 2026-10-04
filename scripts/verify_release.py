"""Verify a published lesson tag from outside the repository: the public website and a fresh clone.

    python scripts/verify_release.py site --tag lessons-v0.4.0 --output site.json
    python scripts/verify_release.py bootstrap --tag lessons-v0.4.0 --output bootstrap.json
    python scripts/verify_release.py save --run RUN_ID

site       builds the tag's website (strict) and compares every navigation page's article and every
           figure with the public GitHub Pages site; checks that each lesson page links to the tag's
           notebook on Colab, that the notebook's last cell is the lesson program, and that the earlier
           lesson tags still point to the commits the previous verification recorded.
bootstrap  gives each case an empty folder and a new virtual environment, then runs a notebook's
           environment cell exactly as written and its last cell in the same namespace, as one Colab
           session does: without PyTorch, with another PyTorch version, with the pinned version (kept),
           and with another version already imported (the cell asks for a restart; a new process, like
           the restarted session, then runs both cells again); plus chapter 20, whose cell also installs
           the ONNX packages. Then it runs every command in README.md's bash blocks, in order, in another
           fresh clone (all but `zensical serve`, which does not return).
save       downloads both results of a verify-release.yml run into artifacts/checks/.

The manually started workflow .github/workflows/verify-release.yml runs site and bootstrap on a
GitHub-hosted Linux runner with Python 3.12. Exit status is 1 when a check fails; the output is written
either way.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "birdhackor/learn_to_yolo"
CLONE_URL = f"https://github.com/{REPOSITORY}.git"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
OTHER_TORCH = "2.8.0"
RESTART_REQUEST = "重新啟動工作階段"  # what the environment cell asks for after replacing an imported PyTorch
CELL_BREAK = "<<next cell>>"
RECORDS = {"site": ROOT / "artifacts/checks/curriculum-publication.json",
           "bootstrap": ROOT / "artifacts/checks/curriculum-release-bootstrap.json"}
# One Colab session: the cells share one namespace; optionally PyTorch was imported before them.
SESSION = f'''import json, sys
cells = json.loads(open(sys.argv[1], encoding="utf-8").read())
if sys.argv[2] == "import-torch-first":
    import torch
namespace = {{"__name__": "__main__"}}
for index, source in enumerate(cells):
    if index:
        print("{CELL_BREAK}", flush=True)
    exec(compile(source, f"<cell {{index + 1}}>", "exec"), namespace)
'''
TORCH_STATE = '''import importlib.metadata as metadata, json, os
try:
    dist = metadata.distribution("torch")
except metadata.PackageNotFoundError:
    print("null")
else:
    record = next(f for f in dist.files if f.name == "RECORD")
    print(json.dumps({"version": dist.version, "record_mtime_ns": os.stat(dist.locate_file(record)).st_mtime_ns}))
'''


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tail(text: str | bytes | None, limit: int = 100_000) -> str:
    text = text.decode(errors="replace") if isinstance(text, bytes) else (text or "")
    return text[-limit:]


def run(command: list | str, cwd: Path, environment: dict | None = None, timeout: int = 3600) -> dict:
    """Run a command (a string runs in bash); stdout and stderr are kept."""
    started = time.perf_counter()
    try:
        done = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, timeout=timeout,
                              shell=isinstance(command, str), executable="/bin/bash" if isinstance(command, str) else None)
        code, stdout, stderr = done.returncode, done.stdout, done.stderr
    except subprocess.TimeoutExpired as error:
        code, stdout, stderr = None, error.stdout, f"timed out after {timeout} s"
    return {"command": command if isinstance(command, str) else " ".join(str(part) for part in command),
            "exit_code": code, "seconds": round(time.perf_counter() - started, 1),
            "stdout": tail(stdout), "stderr": tail(stderr)}


def fetch(url: str) -> tuple[int | None, bytes]:
    headers = {"User-Agent": "learn-to-yolo-release-check"}
    if url.startswith("https://api.github.com/") and os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    try:
        with urlopen(Request(url, headers=headers), timeout=60) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, b""
    except URLError:
        return None, b""


def clone(tag: str, folder: Path) -> str:
    """Shallow public clone of the tag without LFS content; returns its commit."""
    environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    subprocess.run(["git", "-c", "advice.detachedHead=false", "clone", "--quiet", "--depth", "1", "--branch", tag,
                    CLONE_URL, str(folder)],
                   env=environment, check=True)
    return subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"], text=True).strip()


def new_environment(folder: Path) -> Path:
    subprocess.run([sys.executable, "-m", "venv", str(folder)], check=True)
    return folder / "bin/python"


def runner() -> dict:
    cpuinfo = Path("/proc/cpuinfo")
    models = [line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
              if line.startswith("model name")] if cpuinfo.is_file() else []
    details = {"os": platform.platform(), "architecture": platform.machine(),
               "cpu": models[0] if models else platform.processor(), "logical_cpus": os.cpu_count(),
               "python": platform.python_version()}
    if os.environ.get("GITHUB_RUN_ID"):
        details["workflow_run"] = (f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
                                   f"{os.environ.get('GITHUB_REPOSITORY', REPOSITORY)}/actions/runs/{os.environ['GITHUB_RUN_ID']}")
    return details


def article(html: str) -> str | None:
    found = re.search(r"<article\b.*?</article>", html, re.S)
    return found.group(0) if found else None


def site_path(page: str) -> str:
    """URL path of a navigation entry: index.md -> '', lessons/00-warmup.md -> 'lessons/00-warmup/'."""
    stem = page.removesuffix(".md")
    if stem == "index":
        return ""
    return stem.removesuffix("/index") + "/"


def navigation(items: list) -> list[str]:
    pages = []
    for item in items:
        for value in (item.values() if isinstance(item, dict) else [item]):
            pages += navigation(value) if isinstance(value, list) else [value] if value.endswith(".md") else []
    return pages


def check_site(tag: str, work: Path) -> dict:
    source = work / "source"
    commit = clone(tag, source)
    config = tomllib.loads((source / "zensical.toml").read_text())["project"]
    base = config["site_url"]
    python = new_environment(work / "docs-venv")
    steps = [run([python, "-m", "pip", "install", "-q", "-r", "requirements-docs.txt"], source),
             run([work / "docs-venv/bin/zensical", "build", "--clean", "--strict"], source)]
    result = {"verified_at_utc": now(), "source_ref": tag, "release_commit": commit, "website": base,
              "runner": runner(), "build": steps}
    if any(step["exit_code"] != 0 for step in steps):
        result["passed"] = False
        return result
    site = source / config["site_dir"]
    pages, public = [], {}
    for page in navigation(config["nav"]):
        path = site_path(page)
        built = article((site / path / "index.html").read_text())
        status, body = fetch(base + path)
        public[path] = article(body.decode()) if status == 200 else None
        pages.append({"path": path, "http_status": status, "article_matches_build": public[path] == built,
                      "article_sha256": hashlib.sha256((built or "").encode()).hexdigest()})
    figures = []
    for figure in sorted((site / "assets/diagrams").iterdir()):
        status, body = fetch(base + "assets/diagrams/" + figure.name)
        figures.append({"name": figure.name, "http_status": status, "bytes_match_build": body == figure.read_bytes()})
    lessons = []
    for lesson in json.loads((source / "section-map.json").read_text())["sections"]:
        if lesson["kind"] != "lesson":
            continue
        link = f"https://colab.research.google.com/github/{REPOSITORY}/blob/{tag}/{lesson['notebook']}"
        page = site_path(lesson["page"].removeprefix(config["docs_dir"].rstrip("/") + "/"))
        status, body = fetch(f"https://raw.githubusercontent.com/{REPOSITORY}/{tag}/{lesson['notebook']}")
        cells = json.loads(body)["cells"] if status == 200 else None
        program = (source / "lesson_cases" / f"{lesson['id']}.py").read_text()
        lessons.append({"id": lesson["id"], "colab_link_on_public_page": link in (public.get(page) or ""),
                        "notebook_http_status": status,
                        "environment_cell_uses_tag": bool(cells) and f"REF = '{tag}'" in "".join(cells[1]["source"]),
                        "last_cell_is_lesson_program": bool(cells) and "".join(cells[-1]["source"]) == program})
    peeled = {}
    for line in subprocess.check_output(["git", "ls-remote", "--tags", CLONE_URL], text=True).splitlines():
        sha, ref = line.split("\t")
        name = ref.removeprefix("refs/tags/")
        if name.endswith("^{}"):
            peeled[name[:-3]] = sha
        else:
            peeled.setdefault(name, sha)
    previous = json.loads((source / "artifacts/checks/curriculum-publication.json").read_text())
    expected = {old["source_ref"]: old["code_commit"] for old in previous.get("old_releases_retained", [])}
    if "release_commit" in previous:
        expected[previous["source_ref"]] = previous["release_commit"]
    old = [{"source_ref": ref, "code_commit": peeled.get(ref), "matches_previous_verification": peeled.get(ref) == sha}
           for ref, sha in sorted(expected.items())]
    status, body = fetch(f"https://api.github.com/repos/{REPOSITORY}/actions/workflows/pages.yml/runs?status=success&per_page=1")
    runs = json.loads(body).get("workflow_runs", []) if status == 200 else []
    deployment = {"workflow_run_id": runs[0]["id"], "workflow_url": runs[0]["html_url"],
                  "deployed_commit": runs[0]["head_sha"], "deployed_commit_is_release_commit": runs[0]["head_sha"] == commit,
                  } if runs else {"error": f"GitHub API status {status}"}
    result.update({
        "pages_workflow": deployment, "old_releases_retained": old,
        "navigation_pages_verified": len(pages), "pages": pages, "figures": figures, "lessons": lessons,
        "scope": "Public HTTP content of every navigation page's article and every figure equals a strict build of the "
                 "tag on this runner; every lesson page links to the tag's notebook on Colab, whose environment cell "
                 "uses the tag and whose last cell is the lesson program; earlier lesson tags point to the commits "
                 "the previous verification recorded. Colab itself was not opened.",
    })
    result["passed"] = (all(p["http_status"] == 200 and p["article_matches_build"] for p in pages)
                        and all(f["http_status"] == 200 and f["bytes_match_build"] for f in figures)
                        and all(l["colab_link_on_public_page"] and l["notebook_http_status"] == 200
                                and l["environment_cell_uses_tag"] and l["last_cell_is_lesson_program"] for l in lessons)
                        and old and all(o["matches_previous_verification"] for o in old) and "workflow_run_id" in deployment)
    return result


def torch_state(python: Path) -> dict | None:
    return json.loads(subprocess.check_output([python, "-c", TORCH_STATE], text=True))


def session(python: Path, work: Path, cells: list[str], mode: str) -> dict:
    (work / "cells.json").write_text(json.dumps(cells, ensure_ascii=False), encoding="utf-8")
    (work / "session.py").write_text(SESSION, encoding="utf-8")
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    return run([python, "session.py", "cells.json", mode], work, environment)


def check_bootstrap(tag: str, work: Path) -> dict:
    source = work / "source"
    commit = clone(tag, source)
    cases = []
    for name, lesson, installed, imported in (
            ("no-torch", "00-warmup", None, False),
            ("other-torch", "00-warmup", OTHER_TORCH, False),
            ("same-torch", "00-warmup", "pinned", False),
            ("imported-torch", "00-warmup", OTHER_TORCH, True),
            ("deployment", "20-deployment", None, False)):
        notebook = json.loads((source / "notebooks" / f"{lesson}.ipynb").read_text())
        cells = ["".join(notebook["cells"][1]["source"]), "".join(notebook["cells"][-1]["source"])]
        pinned = re.search(r'"torch==([0-9.]+)"', cells[0]).group(1)
        saved = "".join(output.get("text", "") for output in notebook["cells"][-1].get("outputs", [])
                        if output.get("output_type") == "stream" and output.get("name") == "stdout")
        folder = work / name
        folder.mkdir(parents=True)
        python = new_environment(folder / "venv")
        version = pinned if installed == "pinned" else installed
        case = {"name": name, "notebook": f"notebooks/{lesson}.ipynb", "pinned_torch": pinned,
                "setup": run([python, "-m", "pip", "install", "-q", f"torch=={version}", "--index-url", CPU_INDEX],
                             folder) if version else None,
                "torch_before": torch_state(python)}
        first = session(python, folder, cells, "import-torch-first" if imported else "plain")
        case["session"] = first
        final = first
        if imported:
            case["torch_after_first_session"] = torch_state(python)
            case["restart_requested"] = first["exit_code"] != 0 and RESTART_REQUEST in first["stderr"]
            final = case["restarted_session"] = session(python, folder, cells, "plain")
        case["torch_after"] = torch_state(python)
        output = final["stdout"].split(CELL_BREAK + "\n", 1)
        case["case_stdout_matches_notebook"] = len(output) == 2 and output[1] == saved
        after, before = case["torch_after"] or {}, case["torch_before"] or {}
        on_pinned = after.get("version", "").split("+")[0] == pinned
        case["passed"] = final["exit_code"] == 0 and on_pinned and {
            "no-torch": before == {},
            "other-torch": before.get("version", "").startswith(OTHER_TORCH),
            "same-torch": after == before,
            "imported-torch": case.get("restart_requested", False),
            "deployment": True,
        }[name] and (case["case_stdout_matches_notebook"] or name == "deployment")
        cases.append(case)
    readme_folder = work / "readme"
    clone(tag, readme_folder)
    blocks = re.findall(r"```bash\n(.*?)```", (readme_folder / "README.md").read_text(), re.S)
    commands = [line.strip() for block in blocks for line in block.splitlines()
                if line.strip() and not line.strip().startswith("#") and "zensical serve" not in line]
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    readme = [run(command, readme_folder, environment) for command in commands]
    runtime_report = readme_folder / "artifacts/runs/lesson-runtime.json"
    runtime = json.loads(runtime_report.read_text()) if runtime_report.is_file() else {}
    return {
        "verified_at_utc": now(), "source_ref": tag, "release_commit": commit, "runner": runner(),
        "scope": "Fresh public clones of the tag without LFS content, PYTHONPATH unset. Notebook environment cells run "
                 "verbatim with the last cell in one namespace in new virtual environments of the runner's Python; "
                 "README.md's bash commands run in order. A Linux runner, not a hosted Colab runtime; no GPU.",
        "notebook_sessions": cases,
        "readme_commands": readme,
        "lesson_runtime": {"passed": runtime.get("passed"), "total": runtime.get("total"),
                           "same_output_as_record": [r["id"] for r in runtime.get("results", []) if r.get("same_output_as_record")],
                           "different_output": [r["id"] for r in runtime.get("results", [])
                                                if r.get("passed") and not r.get("same_output_as_record")]},
        "passed": all(case["passed"] for case in cases) and all(step["exit_code"] == 0 for step in readme),
    }


def save(run_id: str) -> None:
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run(["gh", "run", "download", run_id, "--dir", folder], cwd=ROOT, check=True)
        for name, record in RECORDS.items():
            found = list(Path(folder).rglob(f"{name}.json"))
            assert len(found) == 1, (name, found)
            shutil.copyfile(found[0], record)
            print(f"{record.relative_to(ROOT)}: passed={json.loads(record.read_text())['passed']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=("site", "bootstrap", "save"))
    parser.add_argument("--tag", help="published lesson tag (site, bootstrap)")
    parser.add_argument("--output", type=Path, help="result JSON (site, bootstrap)")
    parser.add_argument("--run", help="verify-release.yml run ID (save)")
    args = parser.parse_args()
    if args.check == "save":
        if not args.run:
            parser.error("save needs --run")
        return save(args.run)
    if not args.tag or not args.output:
        parser.error(f"{args.check} needs --tag and --output")
    with tempfile.TemporaryDirectory() as folder:
        result = (check_site if args.check == "site" else check_bootstrap)(args.tag, Path(folder))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.check}: {'passed' if result['passed'] else 'FAILED'} ({args.output})")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
