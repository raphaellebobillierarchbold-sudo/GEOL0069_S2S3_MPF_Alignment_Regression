"""Plotting utilities for spatial MPF prediction maps and regression diagnostics."""

from pathlib import Path
import matplotlib.pyplot as plt


def plot_spatial_map(x, y, values, title, path, label="Value", vmin=None, vmax=None, s=10):
    """Create and save a coordinate scatter map."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(x, y, c=values, s=s, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, label=label)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred, title, path):
    """Create and save an actual-vs-predicted regression plot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "--")
    plt.xlabel("Actual MPF")
    plt.ylabel("Predicted MPF")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
