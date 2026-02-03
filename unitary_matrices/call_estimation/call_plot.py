import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.lines import Line2D
from unitary_matrices.config.config import CALL_ESTIMATION_OUTPUT_DIR

# -----------------------------
# Parameters
# -----------------------------
K = 1.5
rho = 0.8
n_grid = 150
n_samples = 40_000

def run_experiment(out):
    # -----------------------------
    # Payoff surface
    # -----------------------------
    u = np.linspace(0.0, 1.0, n_grid)
    U1, U2 = np.meshgrid(u, u)

    S1 = np.exp(U1)
    S2 = np.exp(U2)
    payoff = np.maximum(0.5 * (S1 + S2) - K, 0.0)

    # -----------------------------
    # Gaussian copula samples
    # -----------------------------
    U = np.random.rand(n_samples, 2)
    Z = norm.ppf(U)
    L = np.linalg.cholesky([[1.0, rho], [rho, 1.0]])
    Uc = norm.cdf(Z @ L.T)

    S1_s = np.exp(Uc[:, 0])
    S2_s = np.exp(Uc[:, 1])
    payoff_s = np.maximum(0.5 * (S1_s + S2_s) - K, 0.0)

    # -----------------------------
    # Exercise boundary: e^{u1} + e^{u2} = 2K
    # -----------------------------
    u1_line = np.linspace(0.0, 1.0, 500)
    u2_line = np.log(2 * K - np.exp(u1_line))
    mask = (u2_line >= 0.0) & (u2_line <= 1.0)
    u1_line = u1_line[mask]
    u2_line = u2_line[mask]

    # -----------------------------
    # Plot
    # -----------------------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # payoff surface
    ax.plot_surface(
        U1, U2, payoff,
        alpha=0.35,
        linewidth=0,
        antialiased=True
    )

    # copula mass
    ax.scatter(
        Uc[:, 0], Uc[:, 1], payoff_s,
        s=2,
        alpha=0.25
    )

    # exercise boundary (orange)
    ax.plot(
        u1_line,
        u2_line,
        np.zeros_like(u1_line),
        color="orange",
        linewidth=3
    )

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Payoff")

    # -----------------------------
    # Legend (manual for 3D)
    # -----------------------------
    legend_elements = [
        Line2D([0], [0], color="blue", lw=6, alpha=0.4, label="Payoff surface"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="gray", markersize=6, label="Copula mass"),
        Line2D([0], [0], color="orange", lw=3, label="Exercise boundary")
    ]

    ax.legend(handles=legend_elements)

    plt.savefig(out, dpi=250)
    plt.show()

if __name__ == "__main__":
    run_experiment(CALL_ESTIMATION_OUTPUT_DIR)
