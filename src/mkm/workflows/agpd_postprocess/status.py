"""Status and failure-isolation helpers for AgPd postprocessing."""

import pandas as pd

POSTPROCESSING_STATUS_COLUMNS = (
    "stage",
    "status",
    "message",
    "warning_count",
    "warnings",
    "good_k",
    "max_pareto_k",
    "n_pareto_k_above_good_k",
    "plot_attempts",
    "plot_failures",
)
LOO_PIT_MAX_PARETO_K = 1.0


def clean_message(value):
    return " ".join(str(value).split())


def warning_text(caught_warnings):
    messages = []
    for caught in caught_warnings:
        message = clean_message(caught.message)
        if message not in messages:
            messages.append(message)
    return "; ".join(messages)


def status_row(stage, status, message="", **values):
    row = {column: "" for column in POSTPROCESSING_STATUS_COLUMNS}
    row.update({"stage": stage, "status": status, "message": clean_message(message)})
    row.update(values)
    return row


def write_postprocessing_status(tables_dir, status_rows):
    pd.DataFrame(status_rows, columns=POSTPROCESSING_STATUS_COLUMNS).to_csv(
        tables_dir / "postprocessing_status.csv",
        index=False,
    )


def safe_plot(status_rows, name, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except Exception as error:
        status_rows.append(status_row(f"plot:{name}", "error", f"{type(error).__name__}: {error}"))
        return False
    status_rows.append(status_row(f"plot:{name}", "complete"))
    return True
