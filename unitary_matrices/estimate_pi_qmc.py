import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path
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


def qmc_from_ginibre(n, seed=None, rng=None):
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
    if method == "CMC":
        return cmc_points(R, seed=seed)
    elif method == "haar":
        try:
            return qmc_from_haar(R, seed=seed)  # your function if present
        except NameError:
            return _qmc_from_haar(R, seed=seed)
    elif method == "ginibre":
        try:
            return qmc_from_ginibre(R, seed=seed)  # your function if present
        except NameError:
            return qmc_from_ginibre(R, seed=seed)
    else:
        raise ValueError("Unknown method")

def run_experiment(Rs=(10,50,100,200,500,1000, 2000), R_chosen=500, outdir="out_pi"):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)

    rows = []
    methods = [("CMC", 101), ("haar", 202), ("ginibre", 303)]

    # ----- global progress -----
    with tqdm(total=len(Rs) * len(methods), desc="Total progress", position=0) as pbar_global:
        for R in Rs:
            print(f"\nEstimations for R={R}")
            # per-R progress bar: three steps (one per method)
            with tqdm(total=len(methods), desc=f"R={R}", position=1, leave=False) as pbar_R:
                # compute each estimator
                vals = {}
                for name, base_seed in methods:
                    pts = gen_points(name, R, seed=base_seed + R)
                    vals[name] = estimate_pi(pts)
                    pbar_R.update(1)
                    pbar_global.update(1)

            rows.append({
                "R": R,
                "CMC_val": vals["CMC"],     "CMC_err": abs(vals["CMC"] - pi),
                "haar_val": vals["haar"],   "haar_err": abs(vals["haar"] - pi),
                "ginibre_val": vals["ginibre"], "ginibre_err": abs(vals["ginibre"] - pi),
            })

    df = pd.DataFrame(rows).set_index("R")
    print("\nResults:\n", df.round(6))
    csv_path = out / "pi_estimates.csv"
    df.to_csv(csv_path, float_format="%.10f")
    print(f"Saved table -> {csv_path}")

    # ----- one figure with 3 side-by-side panels -----
    pts_cmc  = gen_points("CMC", R_chosen, seed=42)
    pts_haar = gen_points("haar", R_chosen, seed=43)
    pts_gin  = gen_points("ginibre", R_chosen, seed=44)

    titles = [f"CMC uniform (R={R_chosen})",
              f"Haar-unitary (R={R_chosen})",
              f"Ginibre (R={R_chosen})"]
    pts_all = [pts_cmc, pts_haar, pts_gin]

    x_curve = np.linspace(0.0, 1.0, 700)
    y_curve = np.sqrt(1.0 - x_curve*x_curve)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, pts, title in zip(axes, pts_all, titles):
        ax.plot(x_curve, y_curve)
        ax.plot([0,1,1,0,0], [0,0,1,1,0])
        ax.scatter(pts[:,0], pts[:,1], s=8)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title(title)
    plt.tight_layout()
    fig_path = out / "pi_three_panels.png"
    fig.savefig(fig_path, dpi=150)
    plt.show()
    print(f"Saved figure -> {fig_path}")

if __name__ == "__main__":
    run_experiment()
