from typing import Any, Sequence, Generator

import numpy as np
import matplotlib.pyplot as plt


def one_dimension_scater(y_axis: Sequence[float], title: str) -> None:
    x_axis = tuple(i for i in range(len(y_axis)))

    plt.figure(figsize=(10, 4))
    plt.scatter(x_axis, y_axis, s=5)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


def plot_sequence(func: Generator[Any, None, None], n: int = 500, title: str = ""):
    y_axis = tuple(next(func) for _ in range(n))
    one_dimension_scater(y_axis, title)
    plt.show()


def default_scatter_plot(ax_vals, ay_vals, title: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ax_vals, ay_vals, s=0.5, label="Samples")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_aspect("equal")
    return ax


def add_quarter_circle(ax):
    theta = np.linspace(0, np.pi / 2, 200)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.plot(circle_x, circle_y, "r-", linewidth=2, label="Quarter unit circle")
    ax.legend()
