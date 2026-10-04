"""The notebooks' environment cell, run against stubbed pip, git and installed-package metadata."""

import importlib.metadata
import subprocess
import sys

import pytest

from scripts.build_lesson_notebooks import bootstrap

REF = "lessons-v9.9.9"
CPU_TORCH = [sys.executable, "-m", "pip", "install", "-q", "torch==2.9.1",
             "--index-url", "https://download.pytorch.org/whl/cpu"]


def run_cell(monkeypatch, tmp_path, torch_version, torch_imported, deployment=False, tag=REF, commands=None):
    """Execute the cell in tmp_path and return the commands it ran; commands also fills if the cell raises."""
    commands = [] if commands is None else commands

    def version(package):
        if package == "torch" and torch_version is not None:
            return torch_version
        raise importlib.metadata.PackageNotFoundError(package)

    def run(command, **options):
        commands.append(command)
        if command[:2] == ["git", "clone"]:
            assert options["env"]["GIT_LFS_SKIP_SMUDGE"] == "1"
            (tmp_path / command[-1]).mkdir()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "check_output", lambda command, **options: tag + "\n")
    if torch_imported:
        monkeypatch.setitem(sys.modules, "torch", object())
    else:
        monkeypatch.delitem(sys.modules, "torch", raising=False)
    exec(bootstrap(REF, deployment), {})
    return commands


def pip_installs(commands):
    return [c for c in commands if c[1:4] == ["-m", "pip", "install"]]


def test_fresh_runtime_without_torch_installs_the_cpu_build(monkeypatch, tmp_path):
    commands = run_cell(monkeypatch, tmp_path, None, torch_imported=False)
    assert commands[0][:2] == ["git", "clone"] and REF in commands[0]
    assert pip_installs(commands)[0] == CPU_TORCH
    assert pip_installs(commands)[1][5:] == ["numpy==2.3.5", "pillow==12.0.0", "matplotlib==3.10.7"]


def test_matching_version_is_kept_even_as_a_cuda_build(monkeypatch, tmp_path):
    commands = run_cell(monkeypatch, tmp_path, "2.9.1+cu128", torch_imported=True)
    assert CPU_TORCH not in commands
    assert len(pip_installs(commands)) == 1


def test_other_version_is_replaced_by_the_cpu_build(monkeypatch, tmp_path):
    commands = run_cell(monkeypatch, tmp_path, "2.8.0+cu126", torch_imported=False)
    assert pip_installs(commands)[0] == CPU_TORCH


def test_replacing_an_imported_torch_asks_for_a_restart(monkeypatch, tmp_path):
    commands = []
    with pytest.raises(RuntimeError, match="重新啟動工作階段"):
        run_cell(monkeypatch, tmp_path, "2.8.0+cu126", torch_imported=True, commands=commands)
    assert commands[-1] == CPU_TORCH  # stops before installing anything else


def test_a_clone_of_another_release_is_refused(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="程式版本不同"):
        run_cell(monkeypatch, tmp_path, "2.9.1", torch_imported=False, tag="lessons-v0.0.1")


def test_deployment_lesson_also_installs_onnx(monkeypatch, tmp_path):
    commands = run_cell(monkeypatch, tmp_path, "2.9.1", torch_imported=False, deployment=True)
    assert pip_installs(commands)[0][5:] == ["numpy==2.3.5", "pillow==12.0.0", "matplotlib==3.10.7",
                                            "onnx==1.19.1", "onnxruntime==1.23.2"]
