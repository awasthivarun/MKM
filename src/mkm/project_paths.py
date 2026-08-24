"""Canonical repository paths for AgPd CO-oxidation workflows."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls, anchor: str | Path) -> "ProjectPaths":
        """Find the repository root by walking upward to pyproject.toml."""
        path = Path(anchor).resolve()
        start = path if path.is_dir() else path.parent

        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").exists():
                return cls(root=candidate)

        raise FileNotFoundError(f"Could not locate repository root from '{path}'.")

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def agpd_analysis_dir(self) -> Path:
        return self.processed_data_dir / "AgPd_COOx_basic" / "analysis"

    @property
    def agpd_selected_path(self) -> Path:
        return self.agpd_analysis_dir / "AgPd_COOx_basic_selected.parquet"

    @property
    def agpd_summary_path(self) -> Path:
        return self.agpd_analysis_dir / "AgPd_COOx_basic_summary.parquet"

    @property
    def agpd_delta_oh_path(self) -> Path:
        return self.agpd_analysis_dir / "AgPd_COOx_basic_delta_OH.parquet"

    @property
    def agpd_delta_co_path(self) -> Path:
        return self.agpd_analysis_dir / "AgPd_COOx_basic_delta_CO.parquet"

    @property
    def agpd_model_config_path(self) -> Path:
        return self.config_dir / "models" / "agpd_basic.yaml"

    @property
    def agpd_preprocessing_config_path(self) -> Path:
        return self.config_dir / "preprocessing" / "agpd_basic.yaml"

    @property
    def agpd_posterior_root(self) -> Path:
        return self.results_dir / "AgPd_COOx_basic" / "posterior"

    def agpd_posterior_output_dir(self, material: str, model_name: str, likelihood_name: str) -> Path:
        """Return the canonical write location for a posterior fit."""
        if likelihood_name not in {"iid", "setup_intercept"}:
            raise ValueError(f"Unsupported likelihood '{likelihood_name}'.")
        return self.agpd_posterior_root / material / likelihood_name / model_name

    def agpd_posterior_dir(
        self,
        material: str,
        model_name: str,
        likelihood_name: str,
        *,
        require_posterior: bool = True,
    ) -> Path:
        """Resolve a posterior directory, preserving the historical iid fallback."""
        canonical = self.agpd_posterior_output_dir(material, model_name, likelihood_name)

        if likelihood_name == "iid" and not (canonical / "posterior.nc").exists():
            legacy = self.agpd_posterior_root / material / model_name
            if (legacy / "posterior.nc").exists():
                canonical = legacy

        if require_posterior and not (canonical / "posterior.nc").exists():
            raise FileNotFoundError(f"Posterior not found: {canonical / 'posterior.nc'}")

        return canonical


    def agpd_composition_posterior_output_dir(
        self,
        composition_model: str,
        model_name: str,
        likelihood_name: str,
    ) -> Path:
        if likelihood_name not in {"iid", "setup_intercept"}:
            raise ValueError(f"Unsupported likelihood '{likelihood_name}'.")
        return self.agpd_posterior_root / "composition" / composition_model / likelihood_name / model_name

    def agpd_model_comparison_dir(self, material: str, likelihood_name: str) -> Path:
        if likelihood_name not in {"iid", "setup_intercept"}:
            raise ValueError(f"Unsupported likelihood '{likelihood_name}'.")
        return self.agpd_posterior_root / material / likelihood_name / "model_comparison"
