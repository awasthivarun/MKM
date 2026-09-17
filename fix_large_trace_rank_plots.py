from pathlib import Path

TARGET = Path("src/mkm/postprocessing/plotting/posterior.py")

OLD_TRACE = '''    with plt.rc_context({"axes.prop_cycle": DEFAULT_COLOR_CYCLE, "font.weight": "normal"}):
        pc = azp.plot_trace(
            data,
            var_names=plotted_names,
            group="posterior",
            backend="matplotlib",
            aes_by_visuals={"divergence": ["color"]},
            visuals={"divergence": {"marker": "|", "size": 30}},
            col_wrap=3,
            figure_kwargs={
                "figsize": (TRACE_FIGURE_WIDTH_IN, TRACE_PANEL_HEIGHT_IN * nrows + TRACE_EXTRA_HEIGHT_IN),
                "layout": "none",
            },
        )
'''

NEW_TRACE = '''    with plt.rc_context({"axes.prop_cycle": DEFAULT_COLOR_CYCLE, "font.weight": "normal"}):
        with azb.rc_context({"plot.max_subplots": max(40, len(plotted_names))}):
            pc = azp.plot_trace(
                data,
                var_names=plotted_names,
                group="posterior",
                backend="matplotlib",
                aes_by_visuals={"divergence": ["color"]},
                visuals={"divergence": {"marker": "|", "size": 30}},
                col_wrap=3,
                figure_kwargs={
                    "figsize": (TRACE_FIGURE_WIDTH_IN, TRACE_PANEL_HEIGHT_IN * nrows + TRACE_EXTRA_HEIGHT_IN),
                    "layout": "none",
                },
            )
'''

OLD_RANK = '''    with plt.rc_context({"axes.prop_cycle": DEFAULT_COLOR_CYCLE, "font.weight": "normal"}):
        pc = azp.plot_rank(
            data,
            var_names=plotted_names,
            group="posterior",
            backend="matplotlib",
            envelope_prob=0.99,
            stats={"ecdf_pit": {"n_simulations": 1000}},
            col_wrap=3,
            visuals={
                "xlabel": False,
                "credible_interval": {"color": "#D9D9D9", "alpha": 0.42},
            },
            figure_kwargs={
                "figsize": (RANK_FIGURE_WIDTH_IN, RANK_PANEL_HEIGHT_IN * nrows + RANK_EXTRA_HEIGHT_IN),
                "layout": "none",
            },
        )
'''

NEW_RANK = '''    with plt.rc_context({"axes.prop_cycle": DEFAULT_COLOR_CYCLE, "font.weight": "normal"}):
        with azb.rc_context({"plot.max_subplots": max(40, len(plotted_names))}):
            pc = azp.plot_rank(
                data,
                var_names=plotted_names,
                group="posterior",
                backend="matplotlib",
                envelope_prob=0.99,
                stats={"ecdf_pit": {"n_simulations": 1000}},
                col_wrap=3,
                visuals={
                    "xlabel": False,
                    "credible_interval": {"color": "#D9D9D9", "alpha": 0.42},
                },
                figure_kwargs={
                    "figsize": (RANK_FIGURE_WIDTH_IN, RANK_PANEL_HEIGHT_IN * nrows + RANK_EXTRA_HEIGHT_IN),
                    "layout": "none",
                },
            )
'''


def main():
    if not TARGET.exists():
        raise FileNotFoundError(f"Run this script from the repository root; missing {TARGET}.")

    raw = TARGET.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8").replace("\r\n", "\n")

    replacements = ((OLD_TRACE, NEW_TRACE, "trace"), (OLD_RANK, NEW_RANK, "rank"))
    for old, new, label in replacements:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f"Expected exactly one {label} plotting block, found {count}; file may have changed.")
        text = text.replace(old, new, 1)

    TARGET.write_text(text.replace("\n", newline), encoding="utf-8", newline="")
    print(f"Updated {TARGET}: trace and rank plots now raise ArviZ's subplot limit to the parameter count.")


if __name__ == "__main__":
    main()
