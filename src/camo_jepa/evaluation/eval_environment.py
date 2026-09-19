"""Run environment for CaMo-JEPA evaluations: device, code version, input file identity."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import torch


class IdentityError(RuntimeError):
    """An input file is not the one the run was pinned to."""

    def __init__(self, report: dict) -> None:
        super().__init__(f"input file identity check failed: {report}")
        self.report = report


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def verify_files(pinned: dict[str, tuple[str | Path, str | None]]) -> dict[str, dict]:
    """Hash every input before anything unpickles it; raise if a pinned digest differs.

    ``pinned`` maps a label to ``(path, expected_sha256 or None)``. Two checkpoints
    can store the same epoch with different weights, so the digest, not the
    epoch, is what identifies the file.
    """
    report = {}
    for label, (path, expected) in pinned.items():
        digest = sha256_file(path)
        report[label] = {"path": str(path), "sha256": digest, "expected_sha256": expected,
                         "match": None if expected is None else digest == expected.lower()}
    if any(entry["match"] is False for entry in report.values()):
        raise IdentityError(report)
    return report


def _arch_runs_on(arch: str, major: int, minor: int) -> bool:
    """SASS for sm_XY runs on the same major with an equal or higher minor; PTX JIT-compiles upward."""
    match = re.fullmatch(r"(sm|compute)_(\d)(\d+)[a-z]?", arch)
    if not match:
        return False
    kind, arch_major, arch_minor = match.group(1), int(match.group(2)), int(match.group(3))
    if kind == "sm":
        return arch_major == major and arch_minor <= minor
    return (arch_major, arch_minor) <= (major, minor)


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available; this evaluation needs a GPU")
        major, minor = torch.cuda.get_device_capability(device)
        if not any(_arch_runs_on(arch, major, minor) for arch in torch.cuda.get_arch_list()):
            raise SystemExit(f"torch {torch.__version__} has no kernels for sm_{major}{minor} "
                             f"({torch.cuda.get_device_name(device)}): {torch.cuda.get_arch_list()}")
    return device


def git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[3]
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
