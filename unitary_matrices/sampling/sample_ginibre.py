import numpy as np
import matplotlib.pyplot as plt

def ginibre_eigs_scaled(n, rng):
    # complex Ginibre with Var(entry)=1
    G = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2.0)
    vals = np.linalg.eigvals(G) / np.sqrt(n)  # scale by 1/sqrt(n)
    return vals

def demo(ns=(25, 100, 1000), seed=12345):
    rng = np.random.default_rng(np.random.PCG64(seed))
    fig, axes = plt.subplots(1, len(ns), figsize=(12, 3.5))
    if len(ns) == 1:
        axes = [axes]
    for ax, n in zip(axes, ns):
        z = ginibre_eigs_scaled(n, rng)
        ax.scatter(np.real(z), np.imag(z), s=10)
        ax.set_title(f"n={n}")
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo()  # change ns to match your panels if needed
