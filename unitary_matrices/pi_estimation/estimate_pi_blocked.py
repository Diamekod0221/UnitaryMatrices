from math import ceil, sqrt
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from estimate_pi_qmc import (
    qmc_from_haar,
    qmc_from_ginibre,
    cmc_points,
    estimate_pi,
    gen_points as gen_points_original,
)

from unitary_matrices.config.config import PI_ESTIMATION_OUTPUT_DIR
# ----------------------------------------------------
# BLOCKED GENERATOR
# ----------------------------------------------------

def gen_points_blocked(method, R, seed):
    """
    Blocked generator that calls existing methods repeatedly
    but never redefines them.

    Produces ~sqrt(R) blocks of size ~sqrt(R),
    giving O(R^2) total eigen cost.
    """

    s = int(ceil(sqrt(R)))  # block size
    blocks = []

    for i in range(s):
        block_seed = None if seed is None else seed + i

        if method == "CMC":
            pts = cmc_points(s, seed=block_seed)
        elif method == "haar":
            pts = qmc_from_haar(s, seed=block_seed)
        elif method == "ginibre":
            pts = qmc_from_ginibre(s, seed=block_seed)
        else:
            raise ValueError("Unknown method")

        blocks.append(pts)

    big = np.vstack(blocks)
    return big[:R]

# ----------------------------------------------------
# EXPERIMENT
# ----------------------------------------------------

def run_experiment_blocked(
    Rs=(100, 400, 900, 1600),
    outdir=PI_ESTIMATION_OUTPUT_DIR,
):
    out = Path(outdir)
    out.mkdir(exist_ok=True)

    methods = [("CMC", 101), ("haar", 202), ("ginibre", 303)]
    rows = []

    with tqdm(total=len(Rs) * len(methods), desc="Total progress") as pbar:
        for R in Rs:
            print(f"\nR={R}")

            vals = {}
            times = {}

            for name, base_seed in methods:
                seed = base_seed + R

                # time original
                t0 = time.perf_counter()
                pts_o = gen_points_original(name, R, seed)
                t1 = time.perf_counter()
                pi_o = estimate_pi(pts_o)

                # time blocked
                t2 = time.perf_counter()
                pts_b = gen_points_blocked(name, R, seed)
                t3 = time.perf_counter()
                pi_b = estimate_pi(pts_b)

                vals[f"{name}_orig"] = pi_o
                vals[f"{name}_block"] = pi_b
                times[f"{name}_orig_t"] = t1 - t0
                times[f"{name}_block_t"] = t3 - t2

                pbar.update(1)

            row = {"R": R}
            row.update(vals)
            row.update(times)
            rows.append(row)

    df = pd.DataFrame(rows).set_index("R")
    df.to_csv(out / "blocked_results.csv")

    print(df)
    print("Saved →", out / "blocked_results.csv")


if __name__ == "__main__":
    run_experiment_blocked()
