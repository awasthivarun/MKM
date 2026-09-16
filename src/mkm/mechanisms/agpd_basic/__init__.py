"""AgPd CO-oxidation mechanism primitives and evaluators.

The public API is re-exported here so existing imports from
``mkm.mechanisms.agpd_basic`` remain stable while the implementation is split
by responsibility.
"""

from .electrochem import (
    electrochemical_activation_energy,
    electrochemical_free_energy,
    log_equilibrium_constant,
    log_eyring_prefactor_s_inv,
    log_tst_rate_constant,
    thermal_energy_eV,
)
from .finite_co import (
    AgPdCOBFERRLHParameters,
    AgPdCOBFERRLHResult,
    AgPdCOBFERParameters,
    AgPdCOBFParameters,
    AgPdCOBFLHParameters,
    AgPdCOERLHParameters,
    AgPdCOERParameters,
    AgPdCOLHParameters,
    AgPdCOPathwayResult,
    PdCOSSACoverages,
    evaluate_agpd_co_bf,
    evaluate_agpd_co_bf_er,
    evaluate_agpd_co_bf_er_lh,
    evaluate_agpd_co_bf_er_lh_ag10_no_bf,
    evaluate_agpd_co_bf_er_lh_ag10_no_er,
    evaluate_agpd_co_bf_er_lh_capped,
    evaluate_agpd_co_bf_er_lh_capped_ag10_no_bf,
    evaluate_agpd_co_bf_lh,
    evaluate_agpd_co_er,
    evaluate_agpd_co_er_lh,
    evaluate_agpd_co_lh,
    solve_pd_co_ssa_qea_oh,
)
from .legacy_qea import (
    AgPdBFERRLHParameters,
    AgPdBFERRLHResult,
    AgPdERParameters,
    AgPdERResult,
    calculate_log_rate_bf,
    calculate_log_rate_er,
    calculate_log_rate_lh,
    evaluate_agpd_bf_er_lh,
    evaluate_agpd_er,
)
from .state import AgPdPointState, build_agpd_point_state
from .surface import (
    AgQEACoverages,
    PdQEACoverages,
    calculate_ag_qea_coverages,
    calculate_pd_qea_coverages,
    log_surface_fraction,
    logsumexp_pathways,
)

__all__ = [name for name in globals() if not name.startswith("_")]
