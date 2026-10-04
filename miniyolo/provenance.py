"""Describe the machine and code that produced an evidence record."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys


def cpu_model() -> str:
    """CPU model name as the operating system reports it; never raises."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            brand = subprocess.run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            brand = ""
        if brand:
            return brand
    return platform.processor() or platform.machine()


def machine_info(threads: int | str | None = None) -> dict:
    """OS, CPU, Python and PyTorch of this process; threads is the intra-op thread count the run used."""
    import torch
    return {"os": platform.platform(), "architecture": platform.machine(), "cpu": cpu_model(),
            "logical_cpus": os.cpu_count(),
            "threads": int(threads) if threads is not None else torch.get_num_threads(),
            "python": sys.version.split()[0], "torch": str(torch.__version__),
            "torch_cpu_capability": torch.backends.cpu.get_cpu_capability()}


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_dependencies(*entries: str | Path) -> dict[str, str]:
    """SHA-256 of every repository file the entries import, directly or indirectly.

    Follows absolute imports of miniyolo / scripts, relative imports inside packages, and sibling
    modules of a script (scripts put their own folder on sys.path). Files loaded by path at run time
    (e.g. a lesson case loaded with importlib) must be passed as extra entries. Third-party packages
    are not listed; their versions are part of machine_info().
    """
    import ast
    root = Path(__file__).resolve().parents[1]

    def candidates(base: Path, parts: list[str]) -> list[Path]:
        found, here = [], base
        for part in parts:
            here = here / part
            if (here / "__init__.py").is_file():
                found.append(here / "__init__.py")
        if here.with_suffix(".py").is_file():
            found.append(here.with_suffix(".py"))
        return found

    seen: dict[str, str] = {}
    todo = [Path(entry).resolve() for entry in entries]
    while todo:
        path = todo.pop()
        relative = path.relative_to(root).as_posix()
        if relative in seen:
            continue
        seen[relative] = file_sha256(path)
        if path.name != "__init__.py" and (path.parent / "__init__.py").is_file():
            todo.append(path.parent / "__init__.py")  # importing a package module runs the package first
        for node in ast.walk(ast.parse(path.read_text(), filename=relative)):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                bases = [root, path.parent]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = path.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    prefix = node.module.split(".") if node.module else []
                    for alias in node.names:
                        todo += candidates(base, prefix + [alias.name])
                    todo += candidates(base, prefix) if prefix else []
                    continue
                modules = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
                bases = [root, path.parent]
            else:
                continue
            for module in modules:
                parts = module.split(".")
                if parts[0] not in {"miniyolo", "scripts"} and not (path.parent / f"{parts[0]}.py").is_file():
                    continue
                for base in bases:
                    todo += [c for c in candidates(base, parts) if c.resolve().is_relative_to(root)]
    return dict(sorted(seen.items()))

