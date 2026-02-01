from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt

import pandas as pd
from tqdm.auto import tqdm

from unitary_matrices.computation.computation import (
    estimate_call_mc,
    call_bs,
    gen_points_original as gen_points,
)
from unitary_matrices.config.config import CALL_ESTIMATION_OUTPUT_DIR

def plot_convergence(df, bs_price, out_path):
    plt.figure(figsize=(8, 5))

    for method in ["CMC", "kostlan", "ginibre"]:
        plt.plot(
            df.index,
            df[f"{method}_val"],
            marker="o",
            label=method,
        )

    plt.axhline(
        bs_price,
        linestyle="--",
        linewidth=2,
        label="Black–Scholes",
    )

    plt.xscale("log")
    plt.xlabel("Sample size N")
    plt.ylabel("Call price estimate")
    plt.title("Monte Carlo convergence of call option price")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_experiment(
    Rs=(10, 50, 100, 200, 500, 1000, 2000, 4000, 6000),
    out=CALL_ESTIMATION_OUTPUT_DIR,
):
    # ---- option params ----
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    bs_price = call_bs(S0, K, r, sigma, T)
    print("Real price:", bs_price)

    out = Path(out)
    out.mkdir(exist_ok=True)

    rows = []
    methods = [("CMC", 101), ("kostlan", 1107), ("ginibre", 3)]

    with tqdm(total=len(Rs) * len(methods), desc="Total progress", position=0) as pbar_global:
        for R in Rs:
            print(f"\nEstimations for R={R}")

            with tqdm(total=len(methods), desc=f"R={R}", position=1, leave=False) as pbar_R:
                vals = {}
                times = {}

                for name, base_seed in methods:
                    t0 = perf_counter()
                    pts = gen_points(name, R, seed=base_seed + R)
                    times[name] = perf_counter() - t0

                    z = pts[:, 0]  # interpret first coord as N(0,1)
                    vals[name] = estimate_call_mc(z, S0, K, r, sigma, T)

                    pbar_R.update(1)
                    pbar_global.update(1)

            rows.append({
                "R": R,

                "CMC_val": vals["CMC"],
                "CMC_err": abs(vals["CMC"] - bs_price),
                "CMC_gen_time_s": times["CMC"],

                "kostlan_val": vals["kostlan"],
                "kostlan_err": abs(vals["kostlan"] - bs_price),
                "kostlan_gen_time_s": times["kostlan"],

                "ginibre_val": vals["ginibre"],
                "ginibre_err": abs(vals["ginibre"] - bs_price),
                "ginibre_gen_time_s": times["ginibre"],
            })

    df = pd.DataFrame(rows).set_index("R")
    print("\nResults:\n", df.round(6))

    csv_path = out / "call_estimates_with_generation_times.csv"
    df.to_csv(csv_path, float_format="%.10f")
    print(f"Saved table -> {csv_path}")

    fig_path = out / "call_convergence_vs_N.png"
    plot_convergence(df, bs_price, fig_path)
    print(f"Saved convergence plot -> {fig_path}")


if __name__ == "__main__":
    run_experiment()
