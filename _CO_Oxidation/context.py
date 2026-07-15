# context.py
from .config import _resolve_config
from .core import CoreContext
from .processing import ProcessingMixin
from .plotting import PlottingMixin
from .posteriors import PosteriorsMixin
from .drc import DRCMixin

class PlottingContext(CoreContext, ProcessingMixin, PlottingMixin, PosteriorsMixin, DRCMixin):
    """
    Unified context object combining core state, data processing, 
    and plotting routines.
    """
    pass

def build_context(env_type):
    cfg = _resolve_config(env_type)
    return PlottingContext(cfg)