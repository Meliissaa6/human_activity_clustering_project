import hdbscan
import numpy as np
from typing import Tuple


def run_hdbscan(
    X: np.ndarray, 
    min_cluster_size: int, 
    min_samples: int, 
    metric: str
) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
    """
    Execute the HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications 
    with Noise) algorithm on the input data.

    This algorithm is particularly effective for finding clusters of varying densities 
    and shapes in data, automatically classifying points not belonging to any cluster 
    as noise (labeled -1).

    Args
    ----
    X : np.ndarray
        The input data matrix of shape (n_samples, n_features). 
        This is often a low-dimensional embedding like UMAP or t-SNE.
    min_cluster_size : int
        The smallest size grouping to consider a cluster. Must be >= 2.
    min_samples : int
        The number of samples in a neighborhood for a point to be considered as a 
        core point. Larger values result in more conservative clustering (more points 
        classified as noise).
    metric : str
        The distance metric to use (e.g., 'euclidean', 'manhattan', 'cosine').

    Returns
    -------
    y_pred : np.ndarray
        The predicted cluster labels for each sample, shape (n_samples,). 
        Noise points are labeled as -1.
    clusterer : hdbscan.HDBSCAN
        The fitted HDBSCAN clusterer object.
    """
    # Initialize the HDBSCAN clusterer with specified parameters
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric
    )
    # Fit the model and predict cluster labels
    y_pred = clusterer.fit_predict(X)
    
    return y_pred, clusterer