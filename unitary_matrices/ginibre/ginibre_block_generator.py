from __future__ import annotations

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from typing import List

from unitary_matrices.config.config import GINIBRE_OUTPUT_DIR


def ginibre_eigenvalues(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.PCG64(seed))
    G = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2.0)
    lam = la.eigvals(G) / np.sqrt(n)
    return lam


def angle_colors_from_complex(z: np.ndarray) -> np.ndarray:
    theta = (np.angle(z) + 2*np.pi) % (2*np.pi)
    return theta / (2*np.pi)


def plot_scatter_complex(sizes: List[int], seed: int) -> None:
    out_path = GINIBRE_OUTPUT_DIR / "ginibre_scatter_corrected.png"

    fig, axes = plt.subplots(1, len(sizes), figsize=(14, 4), constrained_layout=True)
    cmap = "hsv"
    last_sc = None

    for ax, n in zip(axes, sizes):
        lam = ginibre_eigenvalues(n, seed)
        cols = angle_colors_from_complex(lam)
        sc = ax.scatter(
            lam.real, lam.imag,
            c=cols, cmap=cmap,
            s=18, alpha=0.85, linewidths=0
        )
        last_sc = sc
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"Ginibre eigenvalues (N={n})")

    fig.colorbar(last_sc, ax=axes.ravel().tolist())
    fig.savefig(out_path, dpi=200)
    print(f"Saved scatter -> {out_path}")
    plt.show()


def plot_radial_histograms(sizes: List[int], seed: int) -> None:
    out_path = GINIBRE_OUTPUT_DIR / "ginibre_radial_corrected.png"

    fig, axes = plt.subplots(1, len(sizes), figsize=(14, 4), constrained_layout=True)
    r_theor = np.linspace(0, 1, 200)
    f_r = 2*r_theor  # correct radial density for uniform disk

    for ax, n in zip(axes, sizes):
        lam = ginibre_eigenvalues(n, seed)
        r = np.abs(lam)

        ax.hist(r, bins=40, density=True, color="steelblue", alpha=0.8)
        ax.plot(r_theor, f_r, "k-", linewidth=2)
        ax.set_title(f"N={n}   mean={r.mean():.4f}   max={r.max():.4f}")
        ax.set_xlim(0, 1.2)
        ax.set_xlabel("radius r")
        ax.set_ylabel("density")

    fig.savefig(out_path, dpi=200)
    print(f"Saved radial hist -> {out_path}")
    plt.show()


if __name__ == "__main__":
    sizes = [50, 100, 300]
    seed = 2024

    plot_scatter_complex(sizes, seed)
    plot_radial_histograms(sizes, seed)
