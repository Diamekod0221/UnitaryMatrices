#!/usr/bin/env python3
from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from config import GINIBRE_OUTPUT_DIR
from unitary_matrices.ginibre.ginibre_iterative_fill import make_ginibre_matrix


# ---------------------------
# Raw Ginibre eigenvalues
# ---------------------------
def ginibre_eigenvalues(n: int, seed: int | None = None) -> np.ndarray:
    G = make_ginibre_matrix(n, seed)
    lam = la.eigvals(G) / np.sqrt(n)  # scale so circle radius ~1
    return lam


# ---------------------------
# Helpers
# ---------------------------
def angle_colors_from_complex(z: np.ndarray) -> np.ndarray:
    theta = np.angle(z)  # (-pi, pi]
    theta = (theta + 2 * np.pi) % (2 * np.pi)
    return theta / (2 * np.pi)


# ---------------------------
# A: Scatterplots (complex plane)
# ---------------------------
def plot_scatter_complex(sizes: List[int], seed: int = 2024,
                         out: str = GINIBRE_OUTPUT_DIR / 'ginibre_scatter_corrected.png') -> None:
    fig, axes = plt.subplots(1, len(sizes), figsize=(14, 4), constrained_layout=True)
    cmap = "hsv"
    last_sc = None

    for ax, n in zip(axes, sizes):
        lam = ginibre_eigenvalues(n, seed)
        cols = angle_colors_from_complex(lam)
        sc = ax.scatter(lam.real, lam.imag, c=cols, cmap=cmap, s=18, alpha=0.85, linewidths=0)
        last_sc = sc
        ax.set_title(f"Ginibre eigenvalues (N={n})")
        ax.set_aspect("equal", adjustable="box")
        lim = 1.05
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axvline(0, color="#bbbbbb", lw=0.5)
        ax.axhline(0, color="#bbbbbb", lw=0.5)

    cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist())
    cbar.set_label("angle (θ / 2π)")

    fig.savefig(out, dpi=200)
    print(f"Saved scatter -> {out}")
    plt.show()


# ---------------------------
# B: Radial histograms with theoretical overlay
# ---------------------------
def plot_radial_histograms(sizes: List[int], seed: int = 2024, bins: int = 40,
                           out: str = GINIBRE_OUTPUT_DIR / 'ginibre_radial_corrected.png') -> None:
    fig, axes = plt.subplots(1, len(sizes), figsize=(14, 4), constrained_layout=True)

    r_theor = np.linspace(0.0, 1.0, 200)
    f_r = 2.0 * r_theor  # theoretical radial density for uniform disk (0<=r<=1)

    for ax, n in zip(axes, sizes):
        lam = ginibre_eigenvalues(n, seed)
        r = np.abs(lam)

        ax.hist(r, bins=bins, density=True, alpha=0.8, color="steelblue", label="empirical")
        ax.plot(r_theor, f_r, color="k", lw=2, label="theoretical $f(r)=2r$ (0≤r≤1)")
        ax.set_title(f"Radial |λ| (N={n})\nmean={r.mean():.4f}, max={r.max():.4f}")
        ax.set_xlabel("radius r")
        ax.set_ylabel("density")
        ax.set_xlim(0, 1.2)
        ax.legend()

    fig.savefig(out, dpi=200)
    print(f"Saved radial hist -> {out}")
    plt.show()


# ---------------------------
# Quick numerical diagnostics
# ---------------------------
def summary_stats(sizes: List[int], seed: int = 2024) -> None:
    for n in sizes:
        lam = ginibre_eigenvalues(n, seed)
        r = np.abs(lam)
        print(f"N={n:4d} | mean radius = {r.mean():.6f} | std = {r.std():.6f} | max = {r.max():.6f}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    sizes = [50, 100, 2000]
    seed = 2024

    print("Running corrected diagnostics (raw eigenvalues scaled by sqrt(n))\n")
    summary_stats(sizes, seed)

    plot_scatter_complex(sizes, seed)
    plot_radial_histograms(sizes, seed)
