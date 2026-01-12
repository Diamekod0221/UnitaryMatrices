import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from unitary_matrices.computation.computation import (
    estimate_pi,
    gen_points_original as gen_points,
)
from unitary_matrices.config.config import HAAR_OUTPUT_DIR

def demo(n=500, seed=123, outfile = HAAR_OUTPUT_DIR):
    eigs = gen_points("haar", n, seed=seed)
    eigs_y = gen_points("haar", n, seed=seed+134)

    plt.figure(figsize=(4,4))
    plt.scatter(eigs[:, 0], eigs[:,1], s=20)

    plt.title(f"Eigenvalues of random Haar unitary (n={n})")
    out = outfile / f"haar_sample_{n}"
    plt.savefig(out, dpi=150)
    plt.show()

if __name__ == "__main__":
    demo()
