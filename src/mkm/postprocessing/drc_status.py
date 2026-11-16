"""Record whether DRC artifacts were recalculated or retained from an earlier invocation."""

import json
import shutil
from datetime import datetime, timezone


DRC_STATUS_FILENAME = "drc_status.json"


def record_drc_status(drc_dir, *, recomputed, options=None, figures_dir=None):
    """Update a DRC sidecar without deleting existing calculation products.

    A null last_computed_at_utc means that existing outputs predate this sidecar
    and their calculation time is unknown; it does not mean they were recomputed.
    """
    drc_dir.mkdir(parents=True, exist_ok=True)
    path = drc_dir / DRC_STATUS_FILENAME
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    products = sorted(p.name for p in drc_dir.glob("drc_transition_state*") if p.is_file())
    figures = (sorted(p.relative_to(figures_dir).as_posix()
                      for p in figures_dir.glob("*/drc_transition_states.png") if p.is_file())
               if figures_dir is not None and figures_dir.exists() else [])
    status = ("computed_this_invocation" if recomputed else
              "retained_from_previous_invocation" if products or figures else "skipped_no_existing_products")
    record = {
        "status": status,
        "last_invocation_at_utc": timestamp,
        "last_computed_at_utc": timestamp if recomputed else previous.get("last_computed_at_utc"),
        "last_computation_options": dict(options or {}) if recomputed else previous.get("last_computation_options"),
        "products_present": products,
        "figures_present": figures,
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def reset_figures_preserving_drc(figures_dir):
    """Clear ordinary plots while retaining earlier DRC figures across skipped calculations."""
    preserved = {}
    if figures_dir.exists():
        preserved = {path.relative_to(figures_dir): path.read_bytes()
                     for path in figures_dir.glob("*/drc_transition_states.png") if path.is_file()}
        shutil.rmtree(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for relative, contents in preserved.items():
        destination = figures_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
