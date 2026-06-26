import numpy as np
from matplotlib import pyplot as plt


def plot_coincidence_events(taus: np.ndarray, bins: int) -> None:
    plt.hist(taus, bins)
    plt.title("tau - coincidence events")
    plt.xlabel("tau")
    plt.ylabel("coincidence events")
    plt.show()


def plot_via_histogram(hist: np.ndarray, edges: np.ndarray) -> None:
    plt.stairs(hist, edges, fill=True, color="y")
    plt.title("tau - coincidence events")
    plt.xlabel("tau")
    plt.ylabel("coincidence events")
    plt.show()


def plot_adjacent(
    taus: np.ndarray, bins: int, hist: np.ndarray, edges: np.ndarray
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(taus, bins)
    ax1.set_title("tau - coincidence events (pre np.histogram)")
    ax1.set_xlabel("tau")
    ax1.set_ylabel("coincidence events")

    ax2.stairs(hist, edges, fill=True, color="y")
    ax2.set_title("tau - coincidence events (post np.histogram)")
    ax2.set_xlabel("tau")
    ax2.set_ylabel("coincidence events")

    plt.tight_layout()
    plt.show()
