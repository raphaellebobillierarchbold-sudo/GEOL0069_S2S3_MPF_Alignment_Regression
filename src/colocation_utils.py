"""KDTree colocation utilities for Sentinel-2/Sentinel-3 MPF datasets."""

import numpy as np
from scipy.spatial import KDTree


def colocate_s2_mpf_to_s3(s2_x, s2_y, mpf, s3_x, s3_y):
    """Average Sentinel-2-derived MPF values onto the nearest Sentinel-3 grid points.

    Parameters
    ----------
    s2_x, s2_y : array-like
        Sentinel-2 coordinate arrays.
    mpf : array-like
        Sentinel-2-derived melt pond fraction values.
    s3_x, s3_y : array-like
        Sentinel-3 coordinate arrays.

    Returns
    -------
    mpf_target : numpy.ndarray
        Mean MPF assigned to each Sentinel-3 point.
    s2_count : numpy.ndarray
        Number of Sentinel-2 pixels assigned to each Sentinel-3 point.
    mean_distance : numpy.ndarray
        Mean S2-to-S3 colocation distance for each Sentinel-3 point.
    """
    s2_points = np.column_stack([s2_x, s2_y])
    s3_points = np.column_stack([s3_x, s3_y])

    distances, indices = KDTree(s3_points).query(s2_points)

    mpf_target = np.full(len(s3_points), np.nan)
    s2_count = np.zeros(len(s3_points), dtype=int)
    mean_distance = np.full(len(s3_points), np.nan)

    for s3_idx in np.unique(indices):
        mask = indices == s3_idx
        mpf_target[s3_idx] = np.nanmean(mpf[mask])
        s2_count[s3_idx] = int(mask.sum())
        mean_distance[s3_idx] = np.nanmean(distances[mask])

    return mpf_target, s2_count, mean_distance
