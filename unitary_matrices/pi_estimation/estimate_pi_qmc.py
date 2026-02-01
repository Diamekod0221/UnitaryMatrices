from math import pi
from pathlib import Path
from time import perf_counter

import pandas as pd
from tqdm.auto import tqdm

from unitary_matrices.computation.computation import (
    estimate_pi,
    gen_points_original as gen_points,
)
from unitary_matrices.config.config import PI_ESTIMATION_OUTPUT_DIR
from unitary_matrices.plotting.plotting import plot_three_panels


def run_experiment(
    Rs=(10, 50, 100, 200, 500, 1000, 2000, 4000),
    R_chosen=500,
    out=PI_ESTIMATION_OUTPUT_DIR,
):
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

                    vals[name] = estimate_pi(pts)

                    pbar_R.update(1)
                    pbar_global.update(1)

            rows.append({
                "R": R,

                "CMC_val": vals["CMC"],
                "CMC_err": abs(vals["CMC"] - pi),
                "CMC_gen_time_s": times["CMC"],

                "kostlan_val": vals["kostlan"],
                "kostlan_err": abs(vals["kostlan"] - pi),
                "kostlan_gen_time_s": times["kostlan"],

                "ginibre_val": vals["ginibre"],
                "ginibre_err": abs(vals["ginibre"] - pi),
                "ginibre_gen_time_s": times["ginibre"],
            })

    df = pd.DataFrame(rows).set_index("R")
    print("\nResults:\n", df.round(6))

    csv_path = out / "pi_estimates_with_generation_times.csv"
    df.to_csv(csv_path, float_format="%.10f")
    print(f"Saved table -> {csv_path}")

    # ---- visualization (unchanged) ----
    pts_cmc = gen_points("CMC", R_chosen, seed=42)
    pts_haar = gen_points("kostlan", R_chosen, seed=43)
    pts_gin = gen_points("ginibre", R_chosen, seed=44)

    titles = [
        f"PCG64 uniform (R={R_chosen})",
        f"Kostlan (R={R_chosen})",
        f"Ginibre (R={R_chosen})",
    ]

    fig_path = out / "pi_three_panels_kostlan_ginibre.png"
    plot_three_panels([pts_cmc, pts_haar, pts_gin], titles, fig_path)
    print(f"Saved figure -> {fig_path}")


if __name__ == "__main__":
    run_experiment()
