
from math import ceil, sqrt
import numpy as np
import numpy.linalg as la

# -----------------------------
#  Basic primitives
# -----------------------------

def haar_unitary(n, rng):
    Z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2.0)
    Q, R = la.qr(Z)
    d = np.diag(R)
    Q = Q * (d / np.abs(d))
    return Q

def qmc_from_haar(n, seed=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(np.random.PCG64(seed if seed is not None else 12345))

    def haar_angles(n):
        U = haar_unitary(n, rng)
        eig = la.eigvals(U)
        theta = np.angle(eig) % (2 * np.pi)
        shift = rng.random()
        return (theta / (2 * np.pi) + shift) % 1.0

    x = haar_angles(n)
    y = haar_angles(n)
    return np.column_stack([x, y])

def qmc_from_ginibre(n, seed=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(np.random.PCG64(seed if seed is not None else 12345))
    G = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2.0)
    lam = la.eigvals(G) / np.sqrt(n)
    r = np.abs(lam)
    theta = (np.angle(lam) % (2 * np.pi))
    u = np.clip(r**2, 0.0, 1.0)
    v = theta / (2 * np.pi)
    return np.column_stack([u, v])

def cmc_points(n, seed=None):
    rng = np.random.default_rng(np.random.PCG64(seed if seed is not None else 12345))
    return rng.random((n, 2))

def estimate_pi(points):
    x, y = points[:, 0], points[:, 1]
    inside = (x * x + y * y) <= 1.0
    return 4.0 * np.count_nonzero(inside) / points.shape[0]

# -----------------------------
#  Original & blocked generators
# -----------------------------

def gen_points_original(method, R, seed):
    if method == "CMC":
        return cmc_points(R, seed=seed)
    elif method == "haar":
        return qmc_from_haar(R, seed=seed)
    elif method == "ginibre":
        return qmc_from_ginibre(R, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")

def gen_points_blocked(method, R, seed):
    """
    Blocked ~sqrt(R) x sqrt(R) point generation.
    Produces ~R points but with O(R^2) eigencost instead of O(R^3).
    """
    s = int(ceil(sqrt(R)))
    blocks = []

    for i in range(s):
        block_seed = None if seed is None else seed + i

        if method == "CMC":
            pts = cmc_points(s, seed=block_seed)
        elif method == "haar":
            pts = qmc_from_haar(s, seed=block_seed)
        elif method == "ginibre":
            pts = qmc_from_ginibre(s, seed=block_seed)
        else:
            raise ValueError(f"Unknown method: {method}")

        blocks.append(pts)

    big = np.vstack(blocks)
    return big[:R]
