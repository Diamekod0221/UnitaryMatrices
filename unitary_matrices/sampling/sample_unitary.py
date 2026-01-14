import matplotlib.pyplot as plt
from unitary_matrices.computation.computation import (
    gen_points_original as gen_points,
)
from unitary_matrices.config.config import HAAR_OUTPUT_DIR


def demo(n=500, seed=123, outfile=HAAR_OUTPUT_DIR):
    # CMC points (baseline)
    cmc = gen_points("CMC", n, seed=seed)

    # Haar-based points (your current method)
    haar = gen_points("haar", n, seed=seed)
    haar_y = gen_points("haar", n, seed=seed + 134)

    plt.figure(figsize=(8, 4))

    # --- Left: CMC ---
    plt.subplot(1, 2, 1)
    plt.scatter(cmc[:, 0], cmc[:, 1], s=20)
    plt.title(f"CMC samples (n={n})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # --- Right: Haar ---
    plt.subplot(1, 2, 2)
    plt.scatter(haar[:, 0], haar[:, 1], s=20)
    plt.title(f"Eigenvector-based samples (n={n})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.tight_layout()

    out = outfile / f"cmc_vs_haar_{n}"
    plt.savefig(out, dpi=150)
    plt.show()


if __name__ == "__main__":
    demo()
