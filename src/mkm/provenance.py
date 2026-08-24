"""Run provenance helpers for reproducible posterior fits."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess

import yaml


DEFAULT_VERSION_PACKAGES = (
    "pymc",
    "pytensor",
    "numpy",
    "scipy",
    "numba",
    "nutpie",
    "arviz",
    "arviz-base",
    "arviz-plots",
    "arviz-stats",
    "pandas",
    "xarray",
    "pyarrow",
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with open(path, "rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    commit = result.stdout.strip()
    return commit or None


def package_versions(packages=DEFAULT_VERSION_PACKAGES):
    versions = {}

    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None

    return versions


def build_fit_metadata(
    *,
    root: str | Path,
    material: str,
    model_name: str,
    likelihood_name: str,
    data_path: str | Path,
    model_config_path: str | Path,
    sampler: dict,
):
    root = Path(root)
    data_path = Path(data_path)
    model_config_path = Path(model_config_path)

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "material": material,
        "model": model_name,
        "likelihood": likelihood_name,
        "inputs": {
            "data_path": str(data_path.relative_to(root)),
            "data_sha256": sha256_file(data_path),
            "model_config_path": str(model_config_path.relative_to(root)),
            "model_config_sha256": sha256_file(model_config_path),
        },
        "sampler": dict(sampler),
        "package_versions": package_versions(),
    }


def write_run_metadata(metadata: dict, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)
