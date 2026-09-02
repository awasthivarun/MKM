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
    fit_scope: str,
    materials,
    model_name: str,
    error_structure: str,
    data_path: str | Path,
    model_config_path: str | Path,
    sampler: dict,
    parameterization: str | None = None,
    parameterization_specification: dict | None = None,
    prior_material: str | None = None,
    sampling_health: dict | None = None,
):
    root = Path(root)
    data_path = Path(data_path)
    model_config_path = Path(model_config_path)
    materials = tuple(materials)

    if fit_scope not in {"individual", "all_materials"}:
        raise ValueError("fit_scope must be 'individual' or 'all_materials'.")
    if not materials:
        raise ValueError("At least one material is required for fit metadata.")
    if fit_scope == "individual" and len(materials) != 1:
        raise ValueError("Individual fit metadata must contain exactly one material.")
    if fit_scope == "all_materials" and parameterization is None:
        raise ValueError("All-material fit metadata require a parameterization.")
    if fit_scope == "all_materials" and parameterization_specification is None:
        raise ValueError(
            "All-material fit metadata require the resolved parameterization specification."
        )
    if fit_scope == "individual" and parameterization_specification is not None:
        raise ValueError(
            "Individual fit metadata must not contain a composition parameterization specification."
        )

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "fit_scope": fit_scope,
        "materials": list(materials),
        "model": model_name,
        "likelihood": "rate_normal",
        "error_structure": error_structure,
        "parameterization": parameterization,
        "parameterization_specification": parameterization_specification,
        "prior_material": prior_material,
        "inputs": {
            "data_path": str(data_path.relative_to(root)),
            "data_sha256": sha256_file(data_path),
            "model_config_path": str(model_config_path.relative_to(root)),
            "model_config_sha256": sha256_file(model_config_path),
        },
        "sampler": dict(sampler),
        "sampling_health": dict(sampling_health or {}),
        "package_versions": package_versions(),
    }


def read_run_metadata(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Run metadata not found: {path}")

    with open(path, "r") as file:
        metadata = yaml.safe_load(file)

    if not isinstance(metadata, dict):
        raise ValueError(f"Run metadata must contain a mapping: {path}")
    return metadata


def write_run_metadata(metadata: dict, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)
