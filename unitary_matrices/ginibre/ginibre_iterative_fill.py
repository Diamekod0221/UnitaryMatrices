#!/usr/bin/env python3
from __future__ import annotations

import sys
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from unitary_matrices.config.config import GINIBRE_OUTPUT_DIR

# Try to import Hungarian solver; if not available we'll fallback to greedy
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore

    _HUNGARIAN_AVAILABLE = True
except Exception:
    _HUNGARIAN_AVAILABLE = False


def make_ginibre_matrix(N: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate an N x N complex Ginibre matrix (normalized)."""
    rng = np.random.default_rng(np.random.PCG64(seed if seed is not None else 12345))
    G = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2.0)
    return G


def eigvals_principal_submatrix(G: np.ndarray, k: int) -> np.ndarray:
    """Compute scaled eigenvalues for the top-left k x k principal submatrix of G."""
    A = G[:k, :k]
    vals = la.eigvals(A) / np.sqrt(k)  # scale to circular law radius ~1
    return vals


def pairwise_cost_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Euclidean distance cost between complex arrays a and b.
    Returns an (len(a), len(b)) real matrix.
    """
    A = a.reshape(-1, 1)
    B = b.reshape(1, -1)
    # distance on complex plane = abs(a - b)
    D = np.abs(A - B)
    return D


def match_indices(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match elements of a (size m) to elements of b (size n), where n >= m is typical.
    Returns (matched_idx_in_a, matched_idx_in_b, unmatched_idx_in_b).
    - If scipy Hungarian is available, we solve the optimal matching for m->n by padding.
    - Else we perform a greedy nearest neighbor matching (stable and simple).
    """
    m = a.shape[0]
    n = b.shape[0]

    if m == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.arange(n)

    D = pairwise_cost_matrix(a, b)  # shape (m, n)

    if _HUNGARIAN_AVAILABLE:
        # pad to square if needed with large costs so Hungarian returns a minimal matching for m->n
        if m <= n:
            pad = n - m
            D_pad = np.vstack([D, np.full((pad, n), D.max() + 1e6)])
            row_ind, col_ind = linear_sum_assignment(D_pad)
            # only keep rows < m (real matches)
            mask = row_ind < m
            row_ind = row_ind[mask]
            col_ind = col_ind[mask]
            unmatched_b = np.setdiff1d(np.arange(n), col_ind, assume_unique=True)
            return row_ind.astype(int), col_ind.astype(int), unmatched_b.astype(int)
        else:
            # rare: m > n (shouldn't happen since sizes grow). Do Hungarian with transposed cost.
            pad = m - n
            D_pad = np.hstack([D, np.full((m, pad), D.max() + 1e6)])
            row_ind, col_ind = linear_sum_assignment(D_pad)
            mask = col_ind < n
            row_ind = row_ind[mask]
            col_ind = col_ind[mask]
            unmatched_b = np.array([], dtype=int)
            return row_ind.astype(int), col_ind.astype(int), unmatched_b
    else:
        # Greedy nearest neighbor: for each a[i] find nearest unmatched b[j]
        unmatched_b_list = list(range(n))
        matched_a = []
        matched_b = []
        # iterate a in arbitrary order
        for i in range(m):
            if len(unmatched_b_list) == 0:
                break
            dists = D[i, unmatched_b_list]
            j_local = int(np.argmin(dists))
            j = unmatched_b_list[j_local]
            matched_a.append(i)
            matched_b.append(j)
            unmatched_b_list.pop(j_local)
        unmatched_b = np.array(unmatched_b_list, dtype=int)
        return np.array(matched_a, dtype=int), np.array(matched_b, dtype=int), unmatched_b


def build_trajectories(
        G: np.ndarray,
        sizes: List[int]
) -> Dict[int, List[complex]]:
    """
    Build trajectories of eigenvalues across the increasing principal submatrix sizes.
    Returns a dict mapping trajectory_id -> list of complex eigenvalues (one entry per encountered size).
    Trajectory ids are integers starting from 0. Also track 'birth_size' per trajectory via the first size where it appears.
    """
    # compute eigenvalues for each size
    eig_by_size: Dict[int, np.ndarray] = {}
    for k in sizes:
        eig_by_size[k] = eigvals_principal_submatrix(G, k)

    # trajectories: each trajectory is a list of (size, complex value)
    trajectories: List[List[Tuple[int, complex]]] = []

    prev_vals = np.array([], dtype=complex)
    prev_ids: np.ndarray = np.array([], dtype=int)

    for k in sizes:
        cur_vals = eig_by_size[k]
        if prev_vals.size == 0:
            # initialize trajectories from first size
            for v in cur_vals:
                trajectories.append([(k, v)])
            prev_vals = cur_vals.copy()
            prev_ids = np.arange(len(cur_vals), dtype=int)
            continue

        # match prev_vals -> cur_vals
        matched_prev_idx, matched_cur_idx, unmatched_cur_idx = match_indices(prev_vals, cur_vals)

        # extend matched trajectories
        # prev_ids maps matched_prev_idx -> trajectory id in trajectories
        for pi, ci in zip(matched_prev_idx, matched_cur_idx):
            traj_id = int(prev_ids[pi])
            trajectories[traj_id].append((k, cur_vals[ci]))

        # new trajectories for unmatched cur values
        for ci in unmatched_cur_idx:
            new_id = len(trajectories)
            trajectories.append([(k, cur_vals[ci])])

        # build new prev arrays
        # new prev_vals should be cur_vals (for next iteration)
        prev_vals = cur_vals.copy()
        # we must map cur_vals indices to trajectory ids:
        # create array cur_ids of length len(cur_vals)
        cur_ids = np.full(cur_vals.shape[0], -1, dtype=int)
        # fill matched
        for pi, ci in zip(matched_prev_idx, matched_cur_idx):
            cur_ids[ci] = int(prev_ids[pi])  # same trajectory id
        # fill unmatched with newly assigned ids (they were appended in order)
        next_new_id = len(trajectories) - 1
        # but we appended new trajectories in order of unmatched_cur_idx; assign accordingly:
        # find indices of newly added trajectories: they occupy the last len(unmatched_cur_idx) indexes
        if len(unmatched_cur_idx) > 0:
            num_new = len(unmatched_cur_idx)
            new_ids = np.arange(len(trajectories) - num_new, len(trajectories), dtype=int)
            for j, ci in enumerate(unmatched_cur_idx):
                cur_ids[ci] = int(new_ids[j])

        prev_ids = cur_ids

    # convert trajectories list-of-(size,value) to dict mapping traj_id -> complex list (with associated birth size)
    traj_dict: Dict[int, List[complex]] = {}
    for tid, seq in enumerate(trajectories):
        # store only the complex values (in order of increasing size)
        traj_dict[tid] = [val for (_size, val) in seq]

    return traj_dict, trajectories  # also return detailed (size,val) sequences


def plot_trajectories(
        trajectories_detailed: List[List[Tuple[int, complex]]],
        sizes: List[int],
        seed: int,
        output: str = GINIBRE_OUTPUT_DIR / 'ginibre_principal_minor_trajectories.png'
) -> None:
    """
    Plot trajectories (detailed sequences of (size, complex value)).
    Color trajectories by their birth size (the first size in their sequence).
    """
    # determine birth size for each traj
    birth_sizes = [seq[0][0] for seq in trajectories_detailed]
    unique_births = sorted(set(birth_sizes))
    # color map over unique birth sizes
    cmap = plt.get_cmap("viridis")
    birth_to_color = {b: cmap(i / max(1, len(unique_births) - 1)) for i, b in enumerate(unique_births)}

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

    # plot each trajectory as a line
    for seq in trajectories_detailed:
        birth = seq[0][0]
        col = birth_to_color[birth]
        pts = np.array([val for (_s, val) in seq])
        ax.plot(pts.real, pts.imag, "-", linewidth=0.8, color=col, alpha=0.9)
        # mark last point more prominently
        ax.scatter(pts[-1].real, pts[-1].imag, s=10, color=col, edgecolors="none")

    # decorate
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title(f"Ginibre principal-minor eigenvalue trajectories (seed={seed})")
    # legend: show birth sizes
    handles = [plt.Line2D([0], [0], color=birth_to_color[b], lw=3) for b in unique_births]
    labels = [f"birth k={b}" for b in unique_births]
    ax.legend(handles, labels, loc="upper right")
    ax.grid(True, alpha=0.25)

    fig.savefig(output, dpi=220)
    print(f"Saved -> {output}")
    plt.show()


def main():
    # user-configurable
    sizes = [50, 100, 300]  # principal submatrix sizes to inspect (must be increasing)
    N_max = max(sizes)
    seed = 2024

    # generate one large Ginibre matrix
    G = make_ginibre_matrix(N_max, seed=seed)

    # build trajectories
    traj_dict, trajectories_detailed = build_trajectories(G, sizes)

    # trajectories_detailed: list indexed by traj_id, each is list[(size, complex_val), ...]
    plot_trajectories(trajectories_detailed, sizes, seed=seed)


if __name__ == "__main__":
    if not _HUNGARIAN_AVAILABLE:
        print("scipy.optimize.linear_sum_assignment not found — using greedy matching fallback.", file=sys.stderr)
    main()
