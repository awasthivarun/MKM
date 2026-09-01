"""Canonical repository paths for AgPd CO-oxidation workflows."""

from dataclasses import dataclass
from pathlib import Path


FIT_SCOPES = ("individual", "all_materials")
ERROR_STRUCTURES = ("shared", "material")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls, anchor: str | Path) -> "ProjectPaths":
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
    def agpd_results_root(self) -> Path:
        return self.results_dir / "AgPd_COOx_basic"

    @property
    def agpd_posterior_root(self) -> Path:
        return self.agpd_results_root / "posterior"

    @property
    def agpd_prior_predictive_root(self) -> Path:
        return self.agpd_results_root / "prior_predictive"

    @property
    def agpd_validation_root(self) -> Path:
        return self.agpd_results_root / "validation"

    @staticmethod
    def _validate_fit_scope(fit_scope: str):
        if fit_scope not in FIT_SCOPES:
            raise ValueError(
                f"Unknown fit scope '{fit_scope}'. Available scopes: {FIT_SCOPES}."
            )

    @staticmethod
    def _validate_error_structure(error_structure: str):
        if error_structure not in ERROR_STRUCTURES:
            raise ValueError(
                f"Unsupported error structure '{error_structure}'. "
                f"Available structures: {ERROR_STRUCTURES}."
            )

    def _agpd_fit_relative_dir(
        self,
        *,
        fit_scope: str,
        model_name: str,
        material: str | None = None,
        parameterization: str | None = None,
        error_structure: str | None = None,
    ) -> Path:
        self._validate_fit_scope(fit_scope)

        if fit_scope == "individual":
            if material is None:
                raise ValueError("Individual fit paths require a material.")
            return Path("individual") / material / model_name

        if parameterization is None or error_structure is None:
            raise ValueError(
                "All-material fit paths require parameterization and error structure."
            )
        self._validate_error_structure(error_structure)
        return Path("all_materials") / parameterization / error_structure / model_name

    def agpd_posterior_output_dir(self, **fit_specification) -> Path:
        return self.agpd_posterior_root / self._agpd_fit_relative_dir(**fit_specification)

    def agpd_posterior_dir(self, *, require_posterior=True, **fit_specification) -> Path:
        directory = self.agpd_posterior_output_dir(**fit_specification)
        posterior_path = directory / "posterior.nc"

        if require_posterior and not posterior_path.exists():
            raise FileNotFoundError(f"Posterior not found: {posterior_path}")

        return directory

    def agpd_prior_predictive_output_dir(self, **fit_specification) -> Path:
        return self.agpd_prior_predictive_root / self._agpd_fit_relative_dir(**fit_specification)

    def agpd_model_comparison_dir(
        self,
        *,
        fit_scope: str,
        comparison_name: str,
        material: str | None = None,
    ) -> Path:
        self._validate_fit_scope(fit_scope)
        comparison_name = str(comparison_name).strip()
        if not comparison_name:
            raise ValueError("Model comparisons require a non-empty comparison name.")

        if fit_scope == "individual":
            if material is None:
                raise ValueError("Individual model-comparison paths require a material.")
            base = self.agpd_posterior_root / "individual" / material
        else:
            base = self.agpd_posterior_root / "all_materials"

        return base / "model_comparison" / comparison_name

    def agpd_validation_output_dir(
        self,
        *,
        scheme: str,
        model_name: str,
        parameterization: str,
        error_structure: str,
        material: str,
        koh_M: float | None = None,
        co_mole_fraction: float | None = None,
    ) -> Path:
        if scheme not in {"loco", "lomo"}:
            raise ValueError("Validation scheme must be 'loco' or 'lomo'.")
        self._validate_error_structure(error_structure)

        base = (
            self.agpd_validation_root
            / "all_materials"
            / parameterization
            / error_structure
            / model_name
            / scheme
            / material
        )

        if scheme == "lomo":
            if koh_M is not None or co_mole_fraction is not None:
                raise ValueError("LOMO paths do not use KOH or CO condition labels.")
            return base

        if koh_M is None or co_mole_fraction is None:
            raise ValueError("LOCO paths require KOH and CO condition labels.")

        return base / f"KOH_{koh_M:g}_CO_{co_mole_fraction:g}"
