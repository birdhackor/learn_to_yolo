"""Manual single-L4 TensorRT check. Actions installs only the Modal client."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import modal
from modal_gpu_smoke import confirm_stopped, save_result

ROOT = Path(__file__).resolve().parents[1]


def build_app(environment, volume_name):
    volume = modal.Volume.from_name(volume_name, environment_name=environment, create_if_missing=True)
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install_from_requirements(str(ROOT / "requirements-gpu.lock"), extra_options="--require-hashes --only-binary=:all:")
             .pip_install("tensorrt-cu12==10.13.3.9")
             .env({"PYTHONPATH": "/root/project"})
             .add_local_dir(ROOT / "miniyolo", "/root/project/miniyolo"))
    app = modal.App("learn-to-yolo-deployment-verification", include_source=False)

    @app.function(image=image, gpu="L4", cpu=2, memory=4096, timeout=600, startup_timeout=600,
                  retries=0, min_containers=0, buffer_containers=0, max_containers=1,
                  serialized=True, single_use_containers=True, volumes={"/checkpoints": volume})
    def gpu(run_key, commit):
        from miniyolo.deployment_gpu import verify
        volume.reload()
        result = verify("/checkpoints/projects/learn-to-yolo/deployment/" + run_key, commit)
        volume.commit()
        result["volume_committed"] = True
        result["container_task_id"] = os.environ.get("MODAL_TASK_ID")
        return json.dumps(result, allow_nan=False)

    @app.function(image=modal.Image.debian_slim(python_version="3.12"), timeout=120, retries=0,
                  max_containers=1, serialized=True, single_use_containers=True, volumes={"/checkpoints": volume})
    def persisted(run_key):
        volume.reload()
        folder = Path("/checkpoints/projects/learn-to-yolo/deployment") / run_key
        result = {"files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()},
                  "container_task_id": os.environ.get("MODAL_TASK_ID")}
        return json.dumps(result)

    return app, gpu, persisted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--confirm-stopped", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "artifacts/runs/deployment/result.json"
    environment = os.environ.get("MODAL_ENVIRONMENT") or "main"
    if args.confirm_stopped:
        result = json.loads(destination.read_text())
        if not result.get("modal_app_id"):
            return
        result["modal_stop_confirmation"] = confirm_stopped(result["modal_app_id"], environment)
        save_result(destination, result)
        print(json.dumps(result["modal_stop_confirmation"]))
        if not result["modal_stop_confirmation"]["verified"]:
            raise SystemExit(1)
        return
    assert importlib.util.find_spec("torch") is None, "Actions must not install model dependencies"
    app, gpu, persisted = build_app(environment, os.environ.get("MODAL_CHECKPOINT_VOLUME") or "learn-to-yolo-checkpoints")
    if args.validate_only:
        print("Client isolation and Modal declarations validated; no GPU launched")
        return
    commit = os.environ.get("GITHUB_SHA", "")
    run_key = "github-" + os.environ.get("GITHUB_RUN_ID", "local") + "-" + os.environ.get("GITHUB_RUN_ATTEMPT", "1") + "-" + commit[:12]
    result = {"status": "running", "code_commit": commit, "run_key": run_key,
              "modal_environment": environment, "limits": {"gpu": "L4:1", "gpu_timeout_seconds": 600,
              "max_containers": 1, "retries": 0, "billing_limits_changed": False}}
    save_result(destination, result)
    start = time.perf_counter()
    try:
        with modal.enable_output(), app.run(environment_name=environment):
            result["modal_app_id"] = app.app_id
            result["gpu"] = json.loads(gpu.remote(run_key, commit))
            save_result(destination, result)
            result["storage"] = json.loads(persisted.remote(run_key))
            assert result["storage"]["container_task_id"] != result["gpu"]["container_task_id"]
            assert result["storage"]["files"]["grid.onnx"] == result["gpu"]["onnx_sha256"]
            for e in result["gpu"]["engines"]:
                assert result["storage"]["files"]["grid-" + e["precision_flag"] + ".engine"] == e["engine_sha256"]
            result["storage"]["sha256_verified"] = True
        result["status"] = "passed"
        result["gpu_calls_finished"] = True
    except Exception as error:
        # Do not expose SDK error strings that might contain credentials.
        result["status"] = "failed"
        result["failure_type"] = type(error).__name__
        raise
    finally:
        result["client_wall_seconds_including_build_startup_transfer"] = time.perf_counter() - start
        save_result(destination, result)


if __name__ == "__main__":
    main()
