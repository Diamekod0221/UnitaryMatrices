# plotting.py

import numpy as np
import matplotlib.pyplot as plt

def plot_three_panels(pts_list, titles, outfile):
    """
    pts_list: [pts_method1, pts_method2, pts_method3]
    titles:   list of 3 strings
    outfile:  path-like for saving
    """

    x_curve = np.linspace(0.0, 1.0, 700)
    y_curve = np.sqrt(np.clip(1.0 - x_curve * x_curve, 0.0, None))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, pts, title in zip(axes, pts_list, titles):
        ax.plot(x_curve, y_curve)
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0])
        ax.scatter(pts[:, 0], pts[:, 1], s=8)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)

    plt.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
