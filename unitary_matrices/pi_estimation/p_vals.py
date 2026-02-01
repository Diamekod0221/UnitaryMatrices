#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare, kstest, uniform

from unitary_matrices.computation.computation import (
    qmc_from_ginibre,
    qmc_from_ginibre_kostlan,
    cmc_points,
)

# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 500
N_RUNS = 500
BASE_SEED = 10000
N_BINS = 10  # per axis for 2D chi^2

# ============================================================
# POINT GENERATORS (assumed to exist)
# ============================================================

POINT_GENERATORS = {
    "cmc": cmc_points,
    "kostlan": qmc_from_ginibre_kostlan,
    "ginibre": qmc_from_ginibre,
}


# ============================================================
# TESTS
# ============================================================

def chi2_2d_uniform(points, n_bins):
    hist, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=n_bins,
        range=[[0, 1], [0, 1]]
    )

    observed = hist.ravel()
    n = observed.sum()
    k = observed.size

    expected = np.full(k, n / k)

    _, pval = chisquare(observed, expected)
    return pval


def ks_marginals(points):
    """
    KS tests for x and y marginals
    """
    px = kstest(points[:, 0], uniform.cdf).pvalue
    py = kstest(points[:, 1], uniform.cdf).pvalue
    return px, py


# ============================================================
# RUN EXPERIMENT
# ============================================================

results = {
    name: {
        "chi2_2d": [],
        "ks_x": [],
        "ks_y": [],
    }
    for name in POINT_GENERATORS
}

for run in range(N_RUNS):
    print(f"Run {run}")
    seed = BASE_SEED + run

    for name, gen in POINT_GENERATORS.items():
        points = gen(SAMPLE_SIZE, seed=seed)

        results[name]["chi2_2d"].append(
            chi2_2d_uniform(points, N_BINS)
        )

        px, py = ks_marginals(points)
        results[name]["ks_x"].append(px)
        results[name]["ks_y"].append(py)


# ============================================================
# VISUALIZATION
# ============================================================

def plot_pvals(pvals, title):
    plt.hist(pvals, bins=20, range=(0, 1), density=True, alpha=0.7)
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("p-value")
    plt.ylabel("density")
    plt.title(title)


plt.figure(figsize=(14, 9))

i = 1
for name in POINT_GENERATORS:
    plt.subplot(3, len(POINT_GENERATORS), i)
    plot_pvals(results[name]["chi2_2d"], f"{name} – chi² 2D")
    i += 1

    plt.subplot(3, len(POINT_GENERATORS), i)
    plot_pvals(results[name]["ks_x"], f"{name} – KS x")
    i += 1

    plt.subplot(3, len(POINT_GENERATORS), i)
    plot_pvals(results[name]["ks_y"], f"{name} – KS y")
    i += 1
plt.savefig(f"p_vals_tests.png", dpi=300)
plt.tight_layout()
plt.show()

# ============================================================
# META-TEST: are p-values uniform?
# ============================================================

print("\nMeta KS test on p-values (should be ~uniform):\n")

for name in POINT_GENERATORS:
    for test in ["chi2_2d", "ks_x", "ks_y"]:
        pvals = np.array(results[name][test])
        meta_p = kstest(pvals, uniform.cdf).pvalue
        print(f"{name:8s} | {test:8s} | meta-p = {meta_p:.4f}")
