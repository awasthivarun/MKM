"""Run provenance helpers for reproducible posterior fits."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess
import os
import tempfile

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


def sha256_mapping(mapping: dict) -> str:
    """Stable SHA256 for a YAML-serializable resolved specification."""
    payload = yaml.safe_dump(mapping, sort_keys=True).encode("utf-8")
    return sha256(payload).hexdigest()


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
    model_config_paths=None,
    parameterization: str | None = None,
    parameterization_specification: dict | None = None,
    prior_material: str | None = None,
    sampling_health: dict | None = None,
    model_specification: dict | None = None,
    prior_specification: dict | None = None,
):
    root = Path(root)
    data_path = Path(data_path)
    model_config_path = Path(model_config_path)
    if model_config_paths is None:
        model_config_paths = (model_config_path,)
    else:
        model_config_paths = tuple(Path(path) for path in model_config_paths)
        if model_config_path not in model_config_paths:
            raise ValueError("model_config_path must be included in model_config_paths.")
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
        "model_specification": model_specification,
        "prior_specification": prior_specification,
        "resolved_fit_sha256": sha256_mapping({
            "model_specification": model_specification,
            "parameterization_specification": parameterization_specification,
            "prior_specification": prior_specification,
            "error_structure": error_structure,
            "likelihood": "rate_normal",
        }),
        "inputs": {
            "data_path": str(data_path.relative_to(root)),
            "data_sha256": sha256_file(data_path),
            "model_config_path": str(model_config_path.relative_to(root)),
            "model_config_paths": [str(path.relative_to(root)) for path in model_config_paths],
            "model_config_sha256": sha256_mapping({
                str(path.relative_to(root)): sha256_file(path) for path in model_config_paths
            }),
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

    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            yaml.safe_dump(metadata, file, sort_keys=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# Explicitly bound model-construction dependencies. Postprocessing-only changes need not
# invalidate a scientifically identical fitted posterior.
MODEL_SOURCE_COMMON = (
    "src/mkm/constants.py",
    "src/mkm/mechanisms/base.py",
    "src/mkm/mechanisms/co_ssa.py",
    "src/mkm/mechanisms/electrochem.py",
    "src/mkm/model_data.py",
    "src/mkm/model_inputs.py",
    "src/mkm/inference/model.py",
    "src/mkm/inference/likelihoods.py",
    "src/mkm/inference/priors.py",
    "src/mkm/workflows/configuration.py",
    "src/mkm/workflows/likelihood_configuration.py",
)
AGPD_MODEL_SOURCE_FILES = MODEL_SOURCE_COMMON + (
    "src/mkm/models/agpd_basic.py", "src/mkm/models/agpd_registry.py",
    "src/mkm/mechanisms/pd_basic.py",
    "src/mkm/mechanisms/agpd_basic/__init__.py",
    "src/mkm/mechanisms/agpd_basic/electrochem.py",
    "src/mkm/mechanisms/agpd_basic/finite_co.py",
    "src/mkm/mechanisms/agpd_basic/full_ssa.py",
    "src/mkm/mechanisms/agpd_basic/legacy_qea.py",
    "src/mkm/mechanisms/agpd_basic/state.py",
    "src/mkm/mechanisms/agpd_basic/surface.py",
    "src/mkm/workflows/agpd_basic.py",
    "src/mkm/workflows/agpd_fit.py",
)
PT100_MODEL_SOURCE_FILES = MODEL_SOURCE_COMMON + (
    "src/mkm/mechanisms/pt100.py",
    "src/mkm/models/pt100.py",
    "src/mkm/models/ptru_registry.py",
    "src/mkm/workflows/pt100_fit.py",
)


def source_file_hashes(root, filenames):
    root = Path(root)
    return {name: sha256_file(root / name) for name in sorted(set(filenames))}


def validate_fit_input_identity(metadata, root, data_path, model_config_path, model_config_paths):
    root = Path(root)
    config_paths = tuple(Path(path) for path in model_config_paths)
    expected = {
        "data_path": str(Path(data_path).relative_to(root)),
        "data_sha256": sha256_file(data_path),
        "model_config_path": str(Path(model_config_path).relative_to(root)),
        "model_config_paths": [str(path.relative_to(root)) for path in config_paths],
        "model_config_sha256": sha256_mapping({
            str(path.relative_to(root)): sha256_file(path) for path in config_paths
        }),
    }
    actual = metadata.get("inputs") or {}
    return [f"inputs.{key}" for key, value in expected.items() if actual.get(key) != value]


def write_preprocessing_manifest(output_path, *, root, raw_paths, config_paths, source_files, products):
    """Write lineage only for files actually created; products maps dataset identifier to path/count."""
    root = Path(root)
    def file_info(path):
        path = Path(path)
        return {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "raw_inputs": [file_info(path) for path in raw_paths],
        "config_inputs": [file_info(path) for path in config_paths],
        "source_sha256": source_file_hashes(root, source_files),
        "outputs": {
            name: {**file_info(path), "rows": int(rows)}
            for name, (path, rows) in products.items()
        },
    }
    write_run_metadata(manifest, output_path)
    return manifest


def model_data_observation_sha256(model_data):
    observations = model_data.observations
    return sha256_mapping({
        "columns": list(observations.columns),
        "rows": observations.to_dict(orient="records"),
    })
