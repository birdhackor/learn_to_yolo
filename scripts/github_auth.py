"""Run Git using an optional scoped HTTPS Authorization secret.

The secret is supplied to child processes through temporary Git configuration,
never saved in Git config, command arguments, the repository, or output.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit


def git_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    authorization = env.get("GIT_LFS_AUTHORIZATION")
    if authorization:
        count = int(env.get("GIT_CONFIG_COUNT", "0"))
        settings = [
            (
                "http.https://github.com/birdhackor/learn_to_yolo.git.extraheader",
                "Authorization: " + authorization,
            ),
            ("credential.helper", ""),
        ]
        for key, value in settings:
            env[f"GIT_CONFIG_KEY_{count}"] = key
            env[f"GIT_CONFIG_VALUE_{count}"] = value
            count += 1
        env["GIT_CONFIG_COUNT"] = str(count)
    return env


def safe_output(output: str) -> str:
    for name in ["GIT_LFS_AUTHORIZATION", "YOLO_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"]:
        secret = os.environ.get(name)
        if secret:
            output = output.replace(secret, "[REDACTED]")
    output = re.sub(r"(?i)Authorization:\s*[^\r\n]+", "Authorization: [REDACTED]", output)
    output = re.sub(r"(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)", "[REDACTED]", output)

    def sanitize_url(match: re.Match) -> str:
        url = urlsplit(match.group(0))
        return urlunsplit((url.scheme, url.hostname or "", url.path, "", ""))

    return re.sub(r"https?://[^\s]+", sanitize_url, output)


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] != "git":
        print("Usage: python scripts/github_auth.py git <arguments>", file=sys.stderr)
        return 2
    result = subprocess.run(
        ["git", *sys.argv[2:]], env=git_environment(), capture_output=True, text=True
    )
    if result.stdout:
        print(safe_output(result.stdout), end="")
    if result.stderr:
        print(safe_output(result.stderr), end="", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
