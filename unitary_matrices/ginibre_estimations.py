#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Callable, Dict, List, Tuple
from pandas import DataFrame, Series

from unitary_matrices.estimate_pi_qmc import qmc_from_ginibre, estimate_pi, cmc_points

# =====================================================================
# TYPE DEFINITIONS
# =====================================================================

PointGenerator = Callable[[int, int | None], np.ndarray]
"""
A function that takes:
    - R: number of points
    - seed: optional integer seed
and returns an (R, 2) NumPy array of 2D points.
"""

# =====================================================================
# POINT GENERATORS
# =====================================================================

# registry for dynamic dispatch
POINT_GENERATORS: Dict[str, PointGenerator] = {
    "ginibre": lambda R, seed: qmc_from_ginibre(R, seed=seed),
    "cmc":     lambda R, seed: cmc_points(R, seed=seed),
}


# =====================================================================
# CORE PIPELINE
# =====================================================================

def run_single(method: str, R: int, seed: int) -> Tuple[float, float]:
    """
    Run a single π estimation using the requested method.
    Returns:
        (estimate, error)
    """
    generator = POINT_GENERATORS[method]
    pts: np.ndarray = generator(R, seed)
    est: float = estimate_pi(pts)
    err: float = abs(est - float(np.pi))
    return est, err


def run_experiment(
    method: str,
    R: int,
    n_runs: int = 100,
    base_seed: int = 10000
) -> Tuple[DataFrame, float]:
    """
    Run n_runs π estimations for a fixed R and method.
    Returns:
        (DataFrame of runs, variance_of_error)
    """
    rows: List[dict] = []

    for i in range(n_runs):
        seed = base_seed + i
        est, err = run_single(method, R, seed)
        rows.append({"run": i, "seed": seed, "estimate": est, "error": err})

    df: DataFrame = pd.DataFrame(rows)

    errors: Series = df["error"]
    var_err: float = float(errors.var(ddof=1))
    mean_err: float = float(errors.mean())
    std_err: float = float(errors.std(ddof=1))

    print(
        f"[{method}] R={R:5d} | mean={mean_err:.8f} | "
        f"var={var_err:.8e} | std={std_err:.8f}"
    )

    return df, var_err


def sweep_R_values(
    method: str,
    R_values: List[int],
    n_runs: int = 100,
    base_seed: int = 10000
) -> DataFrame:
    """
    Sweep across multiple sample sizes R and compute
    error variance for each.
    Returns:
        DataFrame indexed by R with column 'variance'.
    """
    rows: List[dict] = []

    for R in R_values:
        _, var_err = run_experiment(
            method=method,
            R=R,
            n_runs=n_runs,
            base_seed=base_seed + R * 1000,
        )
        rows.append({"R": R, "variance": var_err})

    df_var: DataFrame = pd.DataFrame(rows).set_index("R")
    return df_var


def plot_variance(df_var: DataFrame, method: str, output: str | None = None) -> None:
    """
    Plot variance vs R for a method.
    """
    if output is None:
        output = f"variance_vs_R_{method}.png"

    plt.figure(figsize=(6, 4))
    plt.plot(df_var.index, df_var["variance"], marker="o")
    plt.xlabel("R (sample size)")
    plt.ylabel("Variance of |π_est − π|")
    plt.title(f"{method.upper()} ensemble: Error variance vs R")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved plot -> {output}")
    plt.show()


# =====================================================================
# MAIN SCRIPT
# =====================================================================
if __name__ == "__main__":

    R_values: List[int] = [100, 500, 1000, 1500]
    n_runs: int = 200

    for method in ["ginibre", "cmc"]:
        print(f"\n=== Running variance sweep for {method.upper()} ===")

        df_var: DataFrame = sweep_R_values(
            method=method,
            R_values=R_values,
            n_runs=n_runs,
        )

        csv_path: str = f"variance_vs_R_{method}.csv"
        df_var.to_csv(csv_path, float_format="%.10f")
        print(f"Saved table -> {csv_path}")

        plot_variance(df_var, method)
