import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group

def haar_unitary_eigs(n, rng):
    U = unitary_group.rvs(n, random_state=rng)
    eigs = np.linalg.eigvals(U)
    return eigs

def demo(n=100, seed=123):
    rng = np.random.default_rng(np.random.PCG64(seed))
    eigs = haar_unitary_eigs(n, rng)
    theta = np.angle(eigs)

    plt.figure(figsize=(4,4))
    plt.scatter(np.real(eigs), np.imag(eigs), s=20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.title(f"Eigenvalues of random Haar unitary (n={n})")
    plt.show()

if __name__ == "__main__":
    demo()
