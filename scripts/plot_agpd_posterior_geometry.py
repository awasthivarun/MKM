from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

POSTERIOR_PATH = (
    ROOT
    / "results"
    / "AgPd_COOx_basic"
    / "posterior"
    / "Ag10Pd90"
    / "CO_BF_ER_LH"
    / "posterior.nc"
)

OUTPUT_DIR = POSTERIOR_PATH.parent / "diagnostics" / "geometry"

PARAMETERS = [
    "Gact2_BF_0",
    "deltaG5_0",
    "q",
    "Gact2_ER_0",
    "beta_2_ER",
]


def main():
    idata = az.from_netcdf(POSTERIOR_PATH)
    posterior = idata.posterior

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in PARAMETERS:
        values = np.asarray(posterior[name])

        fig, ax = plt.subplots(figsize=(8, 4))

        for chain in range(values.shape[0]):
            ax.plot(values[chain], linewidth=0.7, label=f"chain {chain}")

        ax.set_xlabel("draw")
        ax.set_ylabel(name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"trace_{name}.png", dpi=180)
        plt.close(fig)

    pairs = [
        ("Gact2_BF_0", "Gact2_ER_0"),
        ("Gact2_ER_0", "beta_2_ER"),
        ("Gact2_BF_0", "deltaG5_0"),
        ("deltaG5_0", "q"),
    ]

    for x_name, y_name in pairs:
        x = np.asarray(posterior[x_name])
        y = np.asarray(posterior[y_name])

        fig, ax = plt.subplots(figsize=(5, 5))

        for chain in range(x.shape[0]):
            ax.scatter(
                x[chain],
                y[chain],
                s=5,
                alpha=0.35,
                label=f"chain {chain}",
            )

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"pair_{x_name}__{y_name}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()