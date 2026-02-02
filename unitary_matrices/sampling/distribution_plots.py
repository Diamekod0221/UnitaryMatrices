import matplotlib.pyplot as plt
import numpy as np
from pted import pted

from scipy.stats import ks_2samp
from unitary_matrices.config.config import CIRCLES_OUTPUT_DIR
from unitary_matrices.sampling.sample_ginibre import ginibre_eigs_scaled


def kostlan_ginibre_with_order(n, rng):
    k = np.arange(1, n + 1)                 # generation index
    r2 = rng.gamma(shape=k, scale=1.0)      # Gamma(k,1)
    theta = rng.uniform(0, 2*np.pi, size=n)

    z = np.sqrt(r2) * np.exp(1j * theta)
    z /= np.sqrt(n)

    return z, k

import numpy as np
import matplotlib.pyplot as plt

def plot_kostlan_with_radial_grid(z, k, outdir):
    fig, ax = plt.subplots(figsize=(6, 6))

    sc = ax.scatter(
        z.real, z.imag,
        c=k,
        cmap="viridis",
        s=18,
        alpha=0.85
    )

    # Radial reference circles
    for r in np.linspace(0.1, 1.0, 10):
        circle = plt.Circle(
            (0, 0),
            r,
            color="black",
            linewidth=0.6,
            alpha=0.15,
            fill=False
        )
        ax.add_artist(circle)

    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.colorbar(sc, ax=ax, label="generation index k")
    ax.set_title("Kostlan Ginibre points with radial grid (Δr = 0.1)")
    if outdir is not None:
        out = outdir / f"kostlan_generation_colored_{k.size}.png"
        fig.savefig(out, dpi=150)

    plt.show()
    

def linear_stats(z):
        r = np.abs(z)
        return {
            "mean_r2": np.mean(r ** 2),
            "mean_r4": np.mean(r ** 4),
            "mean_exp": np.mean(np.exp(-5 * r ** 2)),
        }


def pted_in_R2(eigvals: np.ndarray, comparison_pts: np.ndarray):
    """
    Two-sample PTED test in R^2.

    Parameters
    ----------
    eigvals : np.ndarray
        Complex eigenvalues, shape (n,)
    comparison_pts : np.ndarray
        Either:
        - complex array of shape (m,), or
        - real array of shape (m, 2) with (x, y)

    Returns
    -------
    float
        p-value from PTED
    """

    # --- Ginibre eigenvalues → R^2 ---
    X = np.column_stack([eigvals.real, eigvals.imag])

    # --- Comparison points → R^2 ---
    if np.iscomplexobj(comparison_pts):
        Y = np.column_stack([comparison_pts.real, comparison_pts.imag])
    else:
        comparison_pts = np.asarray(comparison_pts)
        if comparison_pts.ndim != 2 or comparison_pts.shape[1] != 2:
            raise ValueError("comparison_pts must be complex or shape (m, 2)")
        Y = comparison_pts

    #Pval
    return pted(X, Y)

def compare_ginibre_vs_kostlan(
            ginibre: np.ndarray,
            kostlan: np.ndarray,
            n: int,
            outdir=None,
    ):
        """
        Compare Ginibre eigenvalues and Kostlan points.

        - Runs KS test on radii
        - Produces side-by-side scatter plots
        - Optionally saves figure
        """

        # --- Radial KS test ---
        r_g = np.abs(ginibre)
        r_k = np.abs(kostlan)
        ks = ks_2samp(r_g, r_k)
        print("Radial KS test:", ks)

        print("Linear stat for ginibre:", linear_stats(ginibre))
        print("Linear stat for kostlan :", linear_stats(kostlan))

        pted_result = pted_in_R2(ginibre, kostlan)
        print("PTED result p-value:", pted_result)

        # --- Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].scatter(ginibre.real, ginibre.imag, s=20)
        axes[0].set_title(f"Ginibre eigenvalues (n={n})")
        axes[0].set_xlim(-1.1, 1.1)
        axes[0].set_ylim(-1.1, 1.1)
        axes[0].set_aspect("equal")

        axes[1].scatter(kostlan.real, kostlan.imag, s=20)
        axes[1].set_title(f"Kostlan points (n={n})")
        axes[1].set_xlim(-1.1, 1.1)
        axes[1].set_ylim(-1.1, 1.1)
        axes[1].set_aspect("equal")

        fig.tight_layout()

        # --- Save if requested ---
        if outdir is not None:
            out = outdir / f"ginibre_vs_kostlan_{n}.png"
            fig.savefig(out, dpi=150)

        plt.show()

        return ks



def demo(n=1000, outdir=CIRCLES_OUTPUT_DIR):
    rng = np.random.default_rng(12345)
    ginibre = ginibre_eigs_scaled( n, rng)

    # Haar-based points (your current method)
    kost_points, kost_order = kostlan_ginibre_with_order( n, rng)

    compare_ginibre_vs_kostlan(ginibre, kost_points, n, outdir)
    plot_kostlan_with_radial_grid(kost_points, kost_order, outdir)


if __name__ == "__main__":
    demo()
