import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path

from scipy.stats import qmc
from tqdm.auto import tqdm  # progress bars

# ---------- Fallback implementations (used only if your versions aren't present) ----------
def _haar_unitary(n, rng):
    Z = (rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))) / np.sqrt(2.0)
    Q, R = la.qr(Z)
    d = np.diag(R)
    Q = Q * (d / np.abs(d))
    return Q


def _qmc_from_haar(n, seed=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(np.random.PCG64(seed if seed is not None else 12345))

    def haar_angles(n):
        U = _haar_unitary(n, rng)
        eig = la.eigvals(U)
        theta = np.angle(eig) % (2 * np.pi)
        shift = rng.random()
        return (theta / (2 * np.pi) + shift) % 1.0

    x = haar_angles(n)
    y = haar_angles(n)
    return np.column_stack([x, y])


def _qmc_from_ginibre(n, seed=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(np.random.PCG64(seed if seed is not None else 12345))
    G = (rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))) / np.sqrt(2.0)
    lam = la.eigvals(G) / np.sqrt(n)
    r = np.abs(lam)
    theta = (np.angle(lam) % (2*np.pi))
    u = np.clip(r**2, 0.0, 1.0)
    v = theta / (2*np.pi)
    return np.column_stack([u, v])
# -----------------------------------------------------------------------------------------

def cmc_points(n, seed=None):
    rng = np.random.default_rng(np.random.PCG64(seed if seed is not None else 12345))
    return rng.random((n, 2))

def estimate_pi(points):
    x, y = points[:,0], points[:,1]
    inside = (x*x + y*y) <= 1.0
    return 4.0 * np.count_nonzero(inside) / points.shape[0]


def gen_points(method, R, seed):

    generators = {
        "CMC": lambda: cmc_points(R, seed=seed),
        "haar": lambda: (
            qmc_from_haar(R, seed=seed)
            if "qmc_from_haar" in globals()
            else _qmc_from_haar(R, seed=seed)
        ),
        "ginibre": lambda: (
            qmc_from_ginibre(R, seed=seed)
            if "qmc_from_ginibre" in globals()
            else _qmc_from_ginibre(R, seed=seed)
        ),
        "sobol": lambda: qmc.Sobol(d=2, scramble=True, seed=seed).random(R),
        "halton": lambda: qmc.Halton(d=2, scramble=True, seed=seed).random(R),
    }

    try:
        return generators[method]()
    except KeyError:
        raise ValueError(f"Unknown method: {method}")

import time

def run_experiment(Rs=(50, 100),
                   R_chosen=100,
                   outdir="out_pi"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    methods = [("CMC", 101), ("haar", 202), ("ginibre", 303), ("sobol", 404), ("halton", 505)]
    rows = []

    # ----- compute estimations -----
    with tqdm(total=len(Rs) * len(methods), desc="Total progress", position=0) as pbar_global:
        for R in Rs:
            vals = {}
            times = {}
            with tqdm(total=len(methods), desc=f"R={R}", position=1, leave=False) as pbar_R:
                for name, base_seed in methods:
                    t0 = time.time()
                    pts = gen_points(name, R, seed=base_seed + R)
                    vals[name] = estimate_pi(pts)
                    times[name] = time.time() - t0
                    pbar_R.update(1)
                    pbar_global.update(1)

            row = {
                "R": R,
                **{f"{m[0]}_val": vals[m[0]] for m in methods},
                **{f"{m[0]}_err": abs(vals[m[0]] - pi) for m in methods},
                **{f"{m[0]}_time_s": times[m[0]] for m in methods},
            }
            rows.append(row)

    df = pd.DataFrame(rows).set_index("R").round(6)
    csv_path = out / "pi_estimates.csv"
    df.to_csv(csv_path, float_format="%.10f")

    print("\nResults:\n", df)
    print(f"Saved table -> {csv_path}")

    # ----- plot results for chosen R -----
    plot_point_sets(methods, R_chosen, out)

    return df


# ==========================================================
# Plotting helper
# ==========================================================

def plot_point_sets(methods, R_chosen, outdir):
    """Plot point distributions for different quasi-random methods."""
    out = Path(outdir)

    # Generate sample points for each method
    pts_all = [gen_points(name, R_chosen, seed=base_seed + 999) for name, base_seed in methods]
    titles = [f"{name} (R={R_chosen})" for name, _ in methods]

    # Circle boundary
    x_curve = np.linspace(0.0, 1.0, 700)
    y_curve = np.sqrt(1.0 - x_curve * x_curve)

    # Create 2×3 grid (with one empty if 5 methods)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()

    for ax, pts, title in zip(axes, pts_all, titles):
        ax.plot(x_curve, y_curve, color="black", linewidth=1)
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="gray", linewidth=0.8)
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.7)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)

    # Hide unused subplot (6th slot)
    for ax in axes[len(pts_all):]:
        ax.axis("off")

    plt.tight_layout()
    fig_path = out / "pi_pointsets.png"
    fig.savefig(fig_path, dpi=150)
    plt.show()
    print(f"Saved figure -> {fig_path}")


if __name__ == "__main__":
    run_experiment()
