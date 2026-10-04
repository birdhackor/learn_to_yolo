"""Submit one manual, bounded Modal experiment. The client never imports PyTorch."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import modal

from evidence_records import bound_code

ROOT = Path(__file__).resolve().parents[1]
HF_HUB_VERSION = "2.1.1"


def build_app(environment, volume_name):
    volume = modal.Volume.from_name(volume_name, environment_name=environment, create_if_missing=True)
    gpu_image = (modal.Image.debian_slim(python_version="3.12")
                 .pip_install_from_requirements(str(ROOT / "requirements-gpu.lock"),
                                                extra_options="--require-hashes --only-binary=:all:")
                 .env({"PYTHONPATH": "/root/project", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "7"})
                 .add_local_dir(ROOT / "miniyolo", "/root/project/miniyolo"))
    io_image = modal.Image.debian_slim(python_version="3.12").pip_install(f"huggingface_hub=={HF_HUB_VERSION}")
    app = modal.App("learn-to-yolo-gpu-smoke", include_source=False)

    @app.function(image=gpu_image, gpu="L4", cpu=2, memory=4096, timeout=600, startup_timeout=600,
                  retries=0, min_containers=0, buffer_containers=0, max_containers=1,
                  single_use_containers=True, volumes={"/checkpoints": volume}, serialized=True)
    def gpu(stage, run_key, code_commit, checkpoint_name="midpoint.pt"):
        import json
        import os
        from pathlib import Path
        import re
        import time
        import warnings
        from miniyolo.gpu_smoke import produce, resume, write_json
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,160}", run_key):
            raise ValueError("Invalid project run key")
        directory = Path("/checkpoints/projects/learn-to-yolo/runs") / run_key
        volume.reload()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            if stage == "produce":
                result = produce(directory, code_commit)
            elif stage == "resume":
                if checkpoint_name not in {"midpoint.pt", "hf-downloaded.pt"}:
                    raise ValueError("Unexpected checkpoint filename")
                result = resume(directory, checkpoint_name)
            else:
                raise ValueError("Unknown GPU stage")
        result["warnings"] = sorted({str(item.message) for item in captured})
        result["container_task_id"] = os.environ.get("MODAL_TASK_ID", "unknown")
        result["volume_directory"] = str(directory)
        begun = time.perf_counter()
        volume.commit()
        result["volume_commit_seconds"] = time.perf_counter() - begun
        result["volume_committed"] = True
        write_json(directory / ("producer-committed.json" if stage == "produce" else "resume-committed.json"), result)
        volume.commit()
        # Native JSON string: no tensors or TorchVersion reach the Actions client.
        return json.dumps(result, allow_nan=False)

    @app.function(image=io_image, cpu=1, memory=512, timeout=60, retries=0, max_containers=1,
                  single_use_containers=True, serialized=True)
    def probe_hf():
        import json
        import os
        return json.dumps({"hf_token_present": bool(os.environ.get("HF_TOKEN"))})

    @app.function(image=io_image, cpu=1, memory=1024, timeout=600, retries=0, max_containers=1,
                  single_use_containers=True, volumes={"/checkpoints": volume}, serialized=True)
    def transfer(run_key, checkpoint_repo, use_hf):
        import hashlib
        import json
        import os
        from pathlib import Path
        import shutil
        import time
        directory = Path("/checkpoints/projects/learn-to-yolo/runs") / run_key
        volume.reload()
        produced = json.loads((directory / "producer-committed.json").read_text())
        def checksum(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        observed = {name: checksum(directory / name) for name in produced["files"]}
        if observed != produced["files"]:
            raise RuntimeError("Committed Volume checkpoint/metadata failed SHA-256 verification")
        task_id = os.environ.get("MODAL_TASK_ID", "unknown")
        if task_id == "unknown" or task_id == produced["container_task_id"]:
            raise RuntimeError("Persistence must be verified by a different container")
        result = {"volume": {"read_in_different_container": True, "producer_container": produced["container_task_id"],
                              "verifier_container": task_id, "all_sha256_match": True, "files": observed},
                  "hf": {"status": "not_configured", "missing": ["Modal Secret containing HF_TOKEN"]},
                  "checkpoint_for_resume": "midpoint.pt"}
        token = os.environ.get("HF_TOKEN") if use_hf else None
        if token:
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi(token=token)
            stage = "resolve_private_checkpoint_repo"
            try:
                repo = checkpoint_repo
                if not repo:
                    owner = api.whoami()["name"]
                    repo = owner + "/learn-to-yolo-checkpoints"
                info = api.create_repo(repo_id=repo, private=True, exist_ok=True)
                repository = api.repo_info(repo_id=repo)
                if not repository.private:
                    result["hf"] = {"status": "blocked", "repo": repo, "reason": "Checkpoint repository is public; nothing uploaded"}
                else:
                    stage = "upload_private_checkpoint"
                    remote_path = "projects/learn-to-yolo/runs/" + run_key
                    begun = time.perf_counter()
                    committed = api.upload_folder(repo_id=repo, folder_path=str(directory), path_in_repo=remote_path,
                        allow_patterns=["midpoint.pt", "model-config.json", "preprocessing.json", "data-source.json", "provenance.json"],
                        commit_message="Learn to YOLO bounded GPU checkpoint " + run_key)
                    upload_seconds = time.perf_counter() - begun
                    revision = str(committed.oid)
                    stage = "download_exact_hf_commit"
                    begun = time.perf_counter()
                    downloaded = hf_hub_download(repo_id=repo, filename=remote_path + "/midpoint.pt", revision=revision,
                                                 token=token, cache_dir="/tmp/learn-to-yolo-hf")
                    download_seconds = time.perf_counter() - begun
                    digest = checksum(downloaded)
                    if digest != produced["files"]["midpoint.pt"]:
                        raise RuntimeError("Hugging Face immutable-commit download failed SHA-256 verification")
                    shutil.copyfile(downloaded, directory / "hf-downloaded.pt")
                    begun = time.perf_counter()
                    volume.commit()
                    commit_seconds = time.perf_counter() - begun
                    result["checkpoint_for_resume"] = "hf-downloaded.pt"
                    result["hf"] = {"status": "passed", "repo": repo, "private": True, "path": remote_path,
                        "upload_commit": revision, "sha256": digest, "matches_volume_checkpoint": True,
                        "upload_seconds": upload_seconds, "download_seconds": download_seconds,
                        "downloaded_file_volume_commit_seconds": commit_seconds,
                        "timing_definition": "Upload/download timers include API and network work, exclude container startup; Volume commit timed separately"}
            except Exception as error:
                response = getattr(error, "response", None)
                result["hf"] = {"status": "failed", "stage": stage, "error_type": type(error).__name__,
                                "http_status": getattr(response, "status_code", None)}
        (directory / "transfer.json").write_text(json.dumps(result, indent=2) + "\n")
        volume.commit()
        return json.dumps(result, allow_nan=False)

    return app, gpu, probe_hf, transfer


def save_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def confirm_stopped(app_id, environment):
    """Inspect only this run's app; never stop another project's app."""
    for attempt in range(8):
        process = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json", "--env", environment],
                                 capture_output=True, text=True, timeout=30)
        if process.returncode:
            return {"verified": False, "reason": "Modal app listing denied or unavailable"}
        apps = json.loads(process.stdout)
        own = next((item for item in apps if item["app_id"] == app_id), None)
        if own and own["state"] == "stopped" and int(own["tasks"]) == 0:
            return {"verified": True, "app_id": app_id, "state": "stopped", "running_tasks": 0,
                    "stopped_at": own["stopped_at"]}
        if attempt == 2:
            subprocess.run([sys.executable, "-m", "modal", "app", "stop", app_id],
                           capture_output=True, text=True, timeout=30)
        time.sleep(3)
    return {"verified": False, "reason": "This run's Modal app has not reported stopped/zero tasks"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/runs/gpu-smoke/result.json")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--confirm-stopped", action="store_true")
    args = parser.parse_args()
    if args.confirm_stopped:
        output = ROOT / args.output
        if not output.exists():
            print("No submitted app record to clean up")
            return 0
        result = json.loads(output.read_text())
        if not result.get("modal_app_id"):
            print("No Modal app was created")
            return 0
        stopped = confirm_stopped(result["modal_app_id"], result["modal_environment"])
        result["modal_stop_confirmation"] = stopped
        save_result(output, result)
        print(json.dumps(stopped))
        return 0 if stopped["verified"] else 1
    if args.validate_only:
        assert importlib.util.find_spec("torch") is None, "Modal client must not install the model environment"
        build_app("main", "learn-to-yolo-checkpoints")
        print("Modal SDK declarations valid; client has no PyTorch; no cloud job submitted")
        return 0
    output = ROOT / args.output
    environment = os.environ.get("MODAL_ENVIRONMENT") or "main"
    volume_name = os.environ.get("MODAL_CHECKPOINT_VOLUME") or "learn-to-yolo-checkpoints"
    commit = os.environ.get("GITHUB_SHA", "")
    run_key = "github-" + os.environ.get("GITHUB_RUN_ID", "local") + "-" + os.environ.get("GITHUB_RUN_ATTEMPT", "1") + "-" + commit[:12]
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "code_commit": commit, "run_key": run_key,
              "dependencies_sha256": bound_code("artifacts/checks/gpu-smoke.json"),
              "github_run_url": os.environ.get("GITHUB_SERVER_URL", "https://github.com") + "/" + os.environ.get("GITHUB_REPOSITORY", "birdhackor/learn_to_yolo") + "/actions/runs/" + os.environ.get("GITHUB_RUN_ID", ""),
              "modal_environment": environment, "modal_volume": volume_name,
              "hf_release_repo_configured": bool(os.environ.get("HF_RELEASE_REPO")),
              "limits": {"gpu": "L4:1", "max_containers": 1, "timeout_seconds_per_gpu_call": 600,
                         "automatic_retries": 0, "total_optimizer_updates": 80,
                         "billing_limits_changed": False, "public_model_published": False},
              "client_has_torch": importlib.util.find_spec("torch") is not None,
              "model_lock_sha256": hashlib.sha256((ROOT / "requirements-gpu.lock").read_bytes()).hexdigest(),
              "status": "running"}
    save_result(output, result)
    phase = "validate_actions_bindings"
    begun = time.perf_counter()
    try:
        if result["client_has_torch"]:
            raise RuntimeError("Actions client unexpectedly contains PyTorch")
        missing = [name for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET") if not os.environ.get(name)]
        if missing:
            result["missing_actions_bindings"] = missing
            raise RuntimeError("Missing Modal Actions bindings")
        app, gpu, probe, transfer = build_app(environment, volume_name)
        phase = "list_modal_secret_names"
        preferred = os.environ.get("MODAL_HF_SECRET_NAME")
        try:
            names = [secret.name for secret in modal.Secret.objects.list(environment_name=environment)]
            candidates = ([preferred] if preferred else []) + (["codex_cloud"] if "codex_cloud" in names else []) + names
        except Exception:
            # Listing permissions are optional; use the documented binding directly.
            candidates = [preferred or "codex_cloud"]
        candidates = list(dict.fromkeys(candidates))
        result["hf_secret_names_checked"] = []
        with modal.enable_output(), app.run(environment_name=environment):
            result["modal_app_id"] = app.app_id
            selected = None
            phase = "probe_hf_secret_presence"
            for name in candidates:
                try:
                    presence = json.loads(probe.with_options(secrets=[modal.Secret.from_name(name, environment_name=environment)]).remote())
                    result["hf_secret_names_checked"].append({"name": name, "hf_token_present": presence["hf_token_present"]})
                    if presence["hf_token_present"]:
                        selected = name
                        break
                except Exception as error:
                    result["hf_secret_names_checked"].append({"name": name, "probe_error_type": type(error).__name__})
            phase = "gpu_produce"
            call_started = time.perf_counter()
            result["producer"] = json.loads(gpu.remote("produce", run_key, commit))
            result["producer_remote_wall_seconds"] = time.perf_counter() - call_started
            save_result(output, result)
            phase = "verify_volume_and_hugging_face"
            io = transfer.with_options(secrets=[modal.Secret.from_name(selected, environment_name=environment)]) if selected else transfer
            result["storage"] = json.loads(io.remote(run_key, os.environ.get("HF_CHECKPOINT_REPO", ""), bool(selected)))
            result["hf_secret_used"] = selected
            save_result(output, result)
            phase = "gpu_resume"
            call_started = time.perf_counter()
            result["resume"] = json.loads(gpu.remote("resume", run_key, commit, result["storage"]["checkpoint_for_resume"]))
            result["resume_remote_wall_seconds"] = time.perf_counter() - call_started
            if result["resume"]["container_task_id"] == result["producer"]["container_task_id"]:
                raise RuntimeError("GPU restore should run in a fresh single-use container")
            if result["resume"]["checkpoint_sha256"] != result["producer"]["files"]["midpoint.pt"]:
                raise RuntimeError("GPU resume checkpoint hash changed")
            result["resumed_from_hf_download"] = result["storage"]["hf"]["status"] == "passed"
        # App.run is non-detached; single-use GPU containers shut down after inputs.
        result["modal_app_context_closed"] = True
        result["gpu_calls_finished"] = True
        result["status"] = "passed" if result["storage"]["hf"]["status"] in {"passed", "not_configured"} else "partial"
    except Exception as error:
        result["status"] = "failed"
        message = str(error)
        for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
            if os.environ.get(name):
                message = message.replace(os.environ[name], "[REDACTED]")
        result["failure"] = {"phase": phase, "error_type": type(error).__name__, "message": message}
    finally:
        result["client_total_wall_seconds"] = time.perf_counter() - begun
        result["wall_timing_definition"] = "Client total includes image setup/build and all cloud calls; per-call walls include container startup, network, execution and persistence; not a throughput benchmark"
        save_result(output, result)
    print(json.dumps({"status": result["status"], "phase": phase, "artifact": str(output), "run_key": run_key}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
