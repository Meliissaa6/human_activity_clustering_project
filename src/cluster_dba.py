# src/cluster_dba.py
from typing import Tuple, Dict, Optional
import numpy as np
from tslearn.clustering import TimeSeriesKMeans


def _to_tslearn_shape(X: np.ndarray) -> np.ndarray:
    """
    Convert time series data from shape (n_samples, n_features, n_timesteps) to (n_samples, n_timesteps, n_features) 
    required by the tslearn library.

    Args
    ----
    X : np.ndarray
        Input time series data with shape (n_samples, n_features, n_timesteps).

    Returns
    -------
    np.ndarray
        Transposed time series data with shape (n_samples, n_timesteps, n_features).

    Raises
    ------
    AssertionError
        If X is not 3-dimensional.
    """
    assert X.ndim == 3, "X must be 3D: (n_samples, n_features, n_timesteps)."
    return np.transpose(X, (0, 2, 1))


def run_dba_kmeans(
    X: np.ndarray,
    n_clusters: int = 6,
    n_init: int = 10,
    max_iter: int = 50,
    random_state: int = 42,
    metric_params: Optional[Dict] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, TimeSeriesKMeans, np.ndarray]:
    """
    Run TimeSeriesKMeans with DTW metric (DBA barycenters) on time series data.

    This function first reshapes the input data to (n_samples, n_timesteps, n_features) 
    and then fits a k-means model using Dynamic Time Warping (DTW) as the metric. 
    The cluster centers are computed using the DBA (DTW Barycenter Averaging) method.

    Args
    ----
    X : np.ndarray
        Input time series data. Shape is expected to be (n_samples, n_features, n_timesteps). 
        Data should ideally be z-normalized for shape-based clustering.
    n_clusters : int, optional
        The number of clusters to form. Defaults to 6.
    n_init : int, optional
        Number of times the k-means algorithm will be run with different centroid seeds. Defaults to 10.
    max_iter : int, optional
        Maximum number of iterations of the k-means algorithm for a single run. Defaults to 50.
    random_state : int, optional
        Determines random number generation for centroid initialization. Defaults to 42.
    metric_params : dict or None, optional
        Additional parameters passed to the DTW metric calculation, 
        e.g., {"sakoe_chiba_radius": 5} for constrained DTW. Defaults to None.
    verbose : bool, optional
        Controls the verbosity of the tslearn model fitting. Defaults to False.

    Returns
    -------
    y_pred : np.ndarray
        Cluster labels for each sample, shape (n_samples,).
    model : TimeSeriesKMeans
        The fitted tslearn model object, containing attributes like `.cluster_centers_`.
    centers : np.ndarray
        The computed cluster centroids. Shape is (n_clusters, n_timesteps, n_features).
    """
    if metric_params is None:
        metric_params = {}

    X_ts = _to_tslearn_shape(X)  # (n, T, d)

    model = TimeSeriesKMeans(
        init="k-means++",
        n_clusters=n_clusters,
        metric="dtw",
        metric_params=metric_params,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
    )
    y_pred = model.fit_predict(X_ts)
    centers = model.cluster_centers_  # (k, T, d)
    return y_pred, model, centers