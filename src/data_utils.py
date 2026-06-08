"""Utility functions for loading and summarising project datasets."""

from pathlib import Path
import numpy as np


def load_npz(path):
    """Load a NumPy `.npz` file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the `.npz` file.

    Returns
    -------
    numpy.lib.npyio.NpzFile
        Opened NumPy archive.
    """
    return np.load(Path(path))


def print_npz_summary(name, data):
    """Print keys, shapes, data types and finite ranges for an `.npz` object."""
    print(f"\n{name}")
    print("-" * len(name))
    for key in data.files:
        arr = data[key]
        finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.number) else []
        if len(finite) > 0:
            print(f"{key}: shape={arr.shape}, dtype={arr.dtype}, min={finite.min():.6g}, max={finite.max():.6g}")
        else:
            print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
