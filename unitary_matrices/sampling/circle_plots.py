import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import ks_2samp
from unitary_matrices.config.config import CIRCLES_OUTPUT_DIR
from unitary_matrices.sampling.sample_ginibre import ginibre_eigs_scaled


def kostlan_ginibre(n, rng):
    k = np.arange(1, n + 1)
    r2 = rng.gamma(shape=k, scale=1.0)     # exact Kostlan law
    theta = rng.uniform(0, 2*np.pi, size=n)

    z = np.sqrt(r2) * np.exp(1j * theta)   # complex plane
    return z / np.sqrt(n)                  # same scaling as Ginibre


def demo(n=1000, outfile=CIRCLES_OUTPUT_DIR):
    rng = np.random.default_rng(12345)
    ginibre = ginibre_eigs_scaled( n, rng)

    # Haar-based points (your current method)
    kost_points = kostlan_ginibre( n, rng)


    r_g = np.abs(ginibre)
    r_k = np.abs(kost_points)

    print(ks_2samp(r_g, r_k))

    plt.figure(figsize=(8, 4))

    # --- Left: CMC ---
    plt.subplot(1, 2, 1)
    plt.scatter(ginibre.real, ginibre.imag, s=20)
    plt.title(f"Ginibre eigenvals (n={n})")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    # --- Right: kostlan ---
    plt.subplot(1, 2, 2)
    plt.scatter(kost_points.real, kost_points.imag, s=20)
    plt.title(f"Kostlan points (n={n})")
    plt.axis("equal")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    plt.tight_layout()

    out = outfile / f"ginibre_vs_kostlan_{n}"
    plt.savefig(out, dpi=150)
    plt.show()


if __name__ == "__main__":
    demo()
