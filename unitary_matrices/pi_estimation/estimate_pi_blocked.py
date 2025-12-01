from pathlib import Path
import time
import pandas as pd
from tqdm.auto import tqdm

from unitary_matrices.config.config import PI_ESTIMATION_OUTPUT_DIR

from unitary_matrices.computation.computation import (
    estimate_pi,
    gen_points_original,
    gen_points_blocked,
)


def run_experiment_blocked(
    Rs=(100, 400, 900, 1600),
    outdir=PI_ESTIMATION_OUTPUT_DIR,
):
    out = Path(outdir)
    out.mkdir(exist_ok=True)

    methods = [("CMC", 101), ("ginibre", 202)]
    rows = []

    with tqdm(total=len(Rs) * len(methods), desc="Total progress") as pbar:
        for R in Rs:
            print(f"\nR={R}")

            vals = {}
            times = {}

            for name, base_seed in methods:
                seed = base_seed + R

                # ---- time original ----
                t0 = time.perf_counter()
                pts_o = gen_points_original(name, R, seed)
                t1 = time.perf_counter()
                pi_o = estimate_pi(pts_o)

                # ---- time blocked ----
                t2 = time.perf_counter()
                pts_b = gen_points_blocked(name, R, seed)
                t3 = time.perf_counter()
                pi_b = estimate_pi(pts_b)

                vals[f"{name}_orig"] = pi_o
                vals[f"{name}_block"] = pi_b
                times[f"{name}_orig_time"] = t1 - t0
                times[f"{name}_block_time"] = t3 - t2

                pbar.update(1)

            row = {"R": R}
            row.update(vals)
            row.update(times)
            rows.append(row)

    df = pd.DataFrame(rows).set_index("R")
    out_csv = out / "blocked_results.csv"
    df.to_csv(out_csv)

    print(df)
    print("Saved →", out_csv)


if __name__ == "__main__":
    run_experiment_blocked()
