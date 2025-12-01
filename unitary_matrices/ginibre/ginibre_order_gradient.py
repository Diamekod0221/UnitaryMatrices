#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

from typing import Tuple
from unitary_matrices.estimate_pi_qmc import qmc_from_ginibre


# ============================================================
# Uniform PCG64 points
# ============================================================
def cmc_points(n: int, seed: int) -> np.ndarray:
    """Uniform PCG64 sampling."""
    rng = np.random.default_rng(np.random.PCG64(seed))
    return rng.random((n, 2))


# ============================================================
# Convert points to angle-based colors
# ============================================================
def angle_colors(points: np.ndarray) -> np.ndarray:
    """
    Given (x,y) points in [0,1]^2, produce colors based on polar angle.
    Angle is computed around center = (0.5, 0.5).
    Returned values are normalized to [0,1].
    """
    cx, cy = 0.5, 0.5
    x = points[:, 0] - cx
    y = points[:, 1] - cy

    theta = np.arctan2(y, x)  # [-pi, pi]
    theta = (theta + 2 * np.pi) % (2 * np.pi)  # [0, 2pi)

    return theta / (2 * np.pi)  # normalize -> [0,1]


def generate_angle_colored(generator, R: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    pts = generator(R, seed)
    colors = angle_colors(pts)
    return pts, colors


# ============================================================
# Plot side-by-side: Ginibre vs CMC
# ============================================================
def plot_angle_comparison(
    R: int,
    seed: int = 12345,
    cmap: str = "hsv",
    output: str = "ginibre_angle_coloring.png",
) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    norm = plt.Normalize(0.0, 1.0)

    # -------------------------------------
    # Ginibre
    # -------------------------------------
    gin_pts, gin_cols = generate_angle_colored(qmc_from_ginibre, R, seed)
    sc1 = axes[0].scatter(
        gin_pts[:, 0], gin_pts[:, 1],
        c=gin_cols, cmap=cmap, norm=norm,
        s=12, alpha=0.8, linewidths=0
    )
    axes[0].set_aspect("equal")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"Ginibre (angle-colored)  R={R}, seed={seed}")

    # -------------------------------------
    # CMC uniform
    # -------------------------------------
    cmc_pts, cmc_cols = generate_angle_colored(cmc_points, R, seed)
    sc2 = axes[1].scatter(
        cmc_pts[:, 0], cmc_pts[:, 1],
        c=cmc_cols, cmap=cmap, norm=norm,
        s=12, alpha=0.8, linewidths=0
    )
    axes[1].set_aspect("equal")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"CMC uniform (angle-colored)  R={R}, seed={seed}")

    # Shared colorbar
    cbar = fig.colorbar(sc1, ax=axes.ravel().tolist())
    cbar.set_label("Angle (θ / 2π)")

    fig.savefig(output, dpi=200)
    print(f"Saved -> {output}")
    plt.show()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    plot_angle_comparison(R=20, seed=2024)
