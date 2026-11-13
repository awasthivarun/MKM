"""Canonical repository paths for AgPd CO-oxidation workflows."""

from dataclasses import dataclass, replace
import os
from pathlib import Path


FIT_SCOPES = ("individual", "all_materials")
ERROR_STRUCTURES = ("shared", "material")
AGPD_DATA_VARIANTS = {
    "base": "AgPd_COOx_basic",
    "maxtof": "AgPd_COOx_basic_maxtof",
}
AGPD_DATA_VARIANT_ENV = "MKM_AGPD_DATA_VARIANT"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    agpd_data_variant: str = "base"

    def __post_init__(self):
        if self.agpd_data_variant not in AGPD_DATA_VARIANTS:
            raise ValueError(
                f"Unknown AgPd data variant '{self.agpd_data_variant}'. "
                f"Available variants: {tuple(AGPD_DATA_VARIANTS)}."
            )

    @classmethod
    def discover(cls, anchor: str | Path) -> "ProjectPaths":
        path = Path(anchor).resolve()
        start = path if path.is_dir() else path.parent
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").exists():
                variant = os.environ.get(AGPD_DATA_VARIANT_ENV, "base").strip() or "base"
                return cls(root=candidate, agpd_data_variant=variant)

        raise FileNotFoundError(f"Could not locate repository root from '{path}'.")

    def with_agpd_data_variant(self, variant: str) -> "ProjectPaths":
        return replace(self, agpd_data_variant=variant)

    @property
    def agpd_dataset_name(self) -> str:
        return AGPD_DATA_VARIANTS[self.agpd_data_variant]

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def agpd_raw_dir(self) -> Path:
        # Both preprocessing variants derive from the same raw workbooks.
        return self.raw_data_dir / "AgPd_COOx_basic"

    @property
    def agpd_processed_dir(self) -> Path:
        return self.processed_data_dir / self.agpd_dataset_name

    @property
    def agpd_standardized_dir(self) -> Path:
        return self.agpd_processed_dir / "standardized"

    @property
    def agpd_analysis_dir(self) -> Path:
        return self.agpd_processed_dir / "analysis"

    @property
    def agpd_standardized_path(self) -> Path:
        return self.agpd_standardized_dir / f"{self.agpd_dataset_name}_replicates.parquet"

    @property
    def agpd_full_path(self) -> Path:
        return self.agpd_analysis_dir / f"{self.agpd_dataset_name}_full.parquet"

    @property
    def agpd_selected_path(self) -> Path:
        return self.agpd_analysis_dir / f"{self.agpd_dataset_name}_selected.parquet"

    @property
    def agpd_summary_path(self) -> Path:
        return self.agpd_analysis_dir / f"{self.agpd_dataset_name}_summary.parquet"

    @property
    def agpd_truncation_path(self) -> Path:
        return self.agpd_analysis_dir / f"{self.agpd_dataset_name}_truncation.parquet"

    @property
    def agpd_delta_oh_path(self) -> Path:
        return self.agpd_analysis_dir / f"{self.agpd_dataset_name}_delta_OH.parquet"

    @property
    def agpd_delta_co_path(self) -> Path:
        return self.agpd_analysis_dir / f"{self.agpd_dataset_name}_delta_CO.parquet"

    @property
    def agpd_delta_co_replicates_path(self) -> Path:
        return self.agpd_analysis_dir / f"{self.agpd_dataset_name}_delta_CO_replicates.parquet"

    @property
    def agpd_model_config_path(self) -> Path:
        return self.config_dir / "models" / "agpd_basic.yaml"

    @property
    def agpd_prior_config_path(self) -> Path:
        return self.config_dir / "priors" / "agpd_basic.yaml"

    @property
    def agpd_experimental_model_config_path(self) -> Path:
        return self.config_dir / "models" / "experimental" / "agpd_caps.yaml"

    @property
    def agpd_model_config_paths(self) -> tuple[Path, ...]:
        return (
            self.agpd_model_config_path,
            self.agpd_prior_config_path,
            self.agpd_experimental_model_config_path,
        )

    @property
    def agpd_preprocessing_config_path(self) -> Path:
        filename = "agpd_basic.yaml" if self.agpd_data_variant == "base" else "agpd_basic_maxtof.yaml"
        return self.config_dir / "preprocessing" / filename

    @property
    def agpd_preprocessing_figure_dir(self) -> Path:
        return self.figures_dir / "preprocessing" / self.agpd_dataset_name

    @property
    def agpd_results_root(self) -> Path:
        return self.results_dir / self.agpd_dataset_name

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
            raise ValueError(f"Unknown fit scope '{fit_scope}'. Available scopes: {FIT_SCOPES}.")

    @staticmethod
    def _validate_error_structure(error_structure: str):
        if error_structure not in ERROR_STRUCTURES:
            raise ValueError(
                f"Unsupported error structure '{error_structure}'. Available structures: {ERROR_STRUCTURES}."
            )

    def _agpd_fit_relative_dir(
        self,
        *,
        fit_scope: str,
        model_name: str,
        material: str | None = None,
        parameterization: str | None = None,
        error_structure: str | None = None,
        excluded_materials=(),
    ) -> Path:
        self._validate_fit_scope(fit_scope)

        if fit_scope == "individual":
            if material is None:
                raise ValueError("Individual fit paths require a material.")
            return Path("individual") / material / model_name
        if parameterization is None or error_structure is None:
            raise ValueError("All-material fit paths require parameterization and error structure.")
        self._validate_error_structure(error_structure)
        excluded_materials = tuple(excluded_materials)
        leaf = model_name
        if excluded_materials:
            leaf = f"{model_name}__no_{'_'.join(excluded_materials)}"
        return Path("all_materials") / parameterization / error_structure / leaf

    def agpd_posterior_output_dir(self, **fit_specification) -> Path:
        return self.agpd_posterior_root / self._agpd_fit_relative_dir(**fit_specification)

    def agpd_posterior_dir(self, *, require_posterior=True, **fit_specification) -> Path:
        directory = self.agpd_posterior_output_dir(**fit_specification)
        posterior_path = directory / "posterior.nc"

        if require_posterior and not posterior_path.exists():
            raise FileNotFoundError(f"Posterior not found: {posterior_path}")

        return directory

    @staticmethod
    def fit_tables_dir(fit_dir: str | Path) -> Path:
        return Path(fit_dir) / "tables"

    @staticmethod
    def fit_figures_dir(fit_dir: str | Path) -> Path:
        return Path(fit_dir) / "figures"

    @staticmethod
    def fit_drc_dir(fit_dir: str | Path) -> Path:
        return Path(fit_dir) / "drc"

    @staticmethod
    def fit_composition_dir(fit_dir: str | Path) -> Path:
        return Path(fit_dir) / "composition"

    @staticmethod
    def fit_validation_dir(fit_dir: str | Path) -> Path:
        return Path(fit_dir) / "validation"

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
        fit_dir = self.agpd_posterior_output_dir(
            fit_scope="all_materials",
            model_name=model_name,
            parameterization=parameterization,
            error_structure=error_structure,
        )
        base = self.fit_validation_dir(fit_dir) / scheme / material

        if scheme == "lomo":
            if koh_M is not None or co_mole_fraction is not None:
                raise ValueError("LOMO paths do not use KOH or CO condition labels.")
            return base
        if koh_M is None or co_mole_fraction is None:
            raise ValueError("LOCO paths require KOH and CO condition labels.")

        return base / f"KOH_{koh_M:g}_CO_{co_mole_fraction:g}"

    @staticmethod
    def _validate_ptru_medium(medium: str):
        if medium not in ("acidic", "basic"):
            raise ValueError(f"Unknown PtRu medium '{medium}'. Expected 'acidic' or 'basic'.")

    @staticmethod
    def ptru_dataset_name(medium: str) -> str:
        ProjectPaths._validate_ptru_medium(medium)
        return f"PtRu_COOx_{medium}"

    def ptru_raw_dir(self, medium: str) -> Path:
        return self.raw_data_dir / self.ptru_dataset_name(medium)

    def ptru_processed_dir(self, medium: str) -> Path:
        return self.processed_data_dir / self.ptru_dataset_name(medium)

    def ptru_standardized_dir(self, medium: str) -> Path:
        return self.ptru_processed_dir(medium) / "standardized"

    def ptru_analysis_dir(self, medium: str) -> Path:
        return self.ptru_processed_dir(medium) / "analysis"

    def ptru_standardized_path(self, medium: str) -> Path:
        dataset = self.ptru_dataset_name(medium)
        return self.ptru_standardized_dir(medium) / f"{dataset}_replicates.parquet"

    def ptru_full_path(self, medium: str) -> Path:
        dataset = self.ptru_dataset_name(medium)
        return self.ptru_analysis_dir(medium) / f"{dataset}_full.parquet"

    def ptru_selected_path(self, medium: str) -> Path:
        dataset = self.ptru_dataset_name(medium)
        return self.ptru_analysis_dir(medium) / f"{dataset}_selected.parquet"

    def ptru_truncation_path(self, medium: str) -> Path:
        dataset = self.ptru_dataset_name(medium)
        return self.ptru_analysis_dir(medium) / f"{dataset}_truncation.parquet"

    def ptru_em_path(self, medium: str, sample_id: str, stage: str = "full") -> Path:
        if sample_id not in ("PtC_20pct", "PtC_20pct_1mg"):
            raise ValueError(f"Unknown PtRu Em sample: {sample_id}")
        if stage not in ("full", "replicates"):
            raise ValueError(f"Unknown PtRu Em stage: {stage}")
        dataset = self.ptru_dataset_name(medium)
        folder = self.ptru_standardized_dir(medium) if stage == "replicates" else self.ptru_analysis_dir(medium)
        return folder / f"{dataset}_Em_{sample_id}_{stage}.parquet"

    @property
    def ptru_model_config_path(self) -> Path:
        return self.config_dir / "models" / "pt100.yaml"

    @property
    def ptru_prior_config_path(self) -> Path:
        return self.config_dir / "priors" / "pt100.yaml"

    @property
    def ptru_preprocessing_config_path(self) -> Path:
        return self.config_dir / "preprocessing" / "ptru_todd.yaml"

    def ptru_preprocessing_figure_dir(self, medium: str) -> Path:
        return self.figures_dir / "preprocessing" / self.ptru_dataset_name(medium)

    def ptru_results_root(self, medium: str) -> Path:
        return self.results_dir / self.ptru_dataset_name(medium)

    def ptru_posterior_root(self, medium: str) -> Path:
        return self.ptru_results_root(medium) / "posterior"

    def ptru_prior_predictive_root(self, medium: str) -> Path:
        return self.ptru_results_root(medium) / "prior_predictive"

    def ptru_posterior_output_dir(self, medium: str, model_name: str, material: str = "Pt100") -> Path:
        if material != "Pt100":
            raise ValueError("Only Pt100 Bayesian fits are currently implemented.")
        if model_name not in ("CO_LH", "CO_ER", "CO_ER_LH"):
            raise ValueError(f"Unknown Pt100 mechanism: {model_name}")
        return self.ptru_posterior_root(medium) / "individual" / material / model_name

    def ptru_posterior_dir(
        self, medium: str, model_name: str, material: str = "Pt100", *, require_posterior=True,
    ) -> Path:
        directory = self.ptru_posterior_output_dir(medium, model_name, material)
        posterior = directory / "posterior.nc"
        if require_posterior and not posterior.exists():
            raise FileNotFoundError(f"Posterior not found: {posterior}")
        return directory

    def ptru_prior_predictive_output_dir(self, medium: str, model_name: str) -> Path:
        self.ptru_posterior_output_dir(medium, model_name)
        return self.ptru_prior_predictive_root(medium) / "individual" / "Pt100" / model_name

    def ptru_model_comparison_dir(self, medium: str, comparison_name: str) -> Path:
        name = str(comparison_name).strip()
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("A model comparison requires a valid single-directory name.")
        return self.ptru_posterior_root(medium) / "individual" / "Pt100" / "model_comparison" / name

    def ptru_observable_path(
        self, medium: str, product: str, *, selected: bool = False, sample_id: str | None = None,
    ) -> Path:
        valid_products = {
            "alpha_replicates", "alpha_summary", "delta_CO", "delta_Hplus", "delta_OH",
        }
        if product not in valid_products:
            raise ValueError(f"Unknown PtRu observable product: {product}")

        if product in ("delta_Hplus", "delta_OH"):
            expected = "delta_Hplus" if medium == "acidic" else "delta_OH"
            if product != expected:
                raise ValueError(f"{product} is not the electrolyte-ion observable for {medium}.")

        if sample_id is not None:
            if sample_id not in ("PtC_20pct", "PtC_20pct_1mg"):
                raise ValueError(f"Unknown PtRu Em sample: {sample_id}")
            if selected or product in ("delta_Hplus", "delta_OH"):
                raise ValueError("Em observable products do not support selection or ion orders.")

        dataset = self.ptru_dataset_name(medium)
        source = "" if sample_id is None else f"_Em_{sample_id}"
        suffix = "_selected" if selected else ""
        filename = f"{dataset}{source}_{product}{suffix}.parquet"
        return self.ptru_analysis_dir(medium) / filename
