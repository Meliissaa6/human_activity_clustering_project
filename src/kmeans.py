# src/kmeans.py
from sklearn.cluster import KMeans
from typing import Tuple
import numpy as np
from sklearn.base import ClusterMixin 


def run_kmeans(
    X: np.ndarray, 
    n_clusters: int, 
    n_init: int, 
    max_iter: int, 
    random_state: int
) -> Tuple[np.ndarray, ClusterMixin, np.ndarray]:
    """
    Execute the standard K-Means clustering algorithm.

    This function fits the K-Means model and returns the cluster assignments 
    and the computed centroids.

    Args
    ----
    X : np.ndarray
        The input data matrix of shape (n_samples, n_features).
    n_clusters : int
        The number of clusters to form, K.
    n_init : int
        Number of times the k-means algorithm will be run with different centroid seeds.
    max_iter : int
        Maximum number of iterations of the k-means algorithm for a single run.
    random_state : int
        Determines random number generation for centroid initialization.

    Returns
    -------
    y_pred : np.ndarray
        The predicted cluster labels for each sample, shape (n_samples,).
    model : sklearn.cluster.KMeans
        The fitted K-Means model object.
    centers : np.ndarray
        The computed cluster centroids, shape (n_clusters, n_features).
    """
    # Initialize the K-Means model with specified parameters
    model = KMeans(
        n_clusters=n_clusters, 
        init='k-means++',         # Use the smart initialization method
        n_init=n_init, 
        max_iter=max_iter, 
        random_state=random_state,
        verbose=0                 # Set verbosity to 0
    )
    
    # Fit the model to the data
    model.fit(X)
    
    # Extract the predicted labels and cluster centroids
    y_pred = model.labels_
    centers = model.cluster_centers_ 

    return y_pred, model, centers


