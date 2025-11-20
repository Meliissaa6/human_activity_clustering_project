# src/eval.py
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple, Any


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the clustering Purity score using the Hungarian algorithm for optimal cluster-to-label mapping.

    Purity is a measure of the extent to which clusters contain objects of a single class.
    It ranges from 0 (poor) to 1 (perfect).

    Args
    ----
    y_true : np.ndarray
        True class labels, shape (n_samples,).
    y_pred : np.ndarray
        Predicted cluster labels, shape (n_samples,).

    Returns
    -------
    float
        The calculated Purity score.
    """
    # Build the contingency matrix C[i, j] = count of points in cluster 'i' and true label 'j'
    labels = np.unique(y_true)
    clusters = np.unique(y_pred)
    C = np.zeros((clusters.size, labels.size), dtype=int)
    for i, c in enumerate(clusters):
        for j, l in enumerate(labels):
            # Count the number of samples belonging to cluster c AND true label l
            C[i, j] = np.sum((y_pred == c) & (y_true == l))
    
    # Use the Hungarian algorithm (linear_sum_assignment) to find the optimal assignment 
    # that maximizes the sum of counts. We maximize C, so we minimize C.max() - C.
    row_ind, col_ind = linear_sum_assignment(C.max() - C)
    
    # The sum of assigned counts is the number of correctly classified samples under 
    # the optimal mapping. Purity = (correctly mapped samples) / (total samples).
    return C[row_ind, col_ind].sum() / y_true.size


def external_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard external clustering evaluation scores (require true labels).

    Args
    ----
    y_true : np.ndarray
        True class labels, shape (n_samples,).
    y_pred : np.ndarray
        Predicted cluster labels, shape (n_samples,).

    Returns
    -------
    dict
        A dictionary containing "ARI", "NMI", and "Purity" scores.
    """
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic"),
        "Purity": purity_score(y_true, y_pred),
    }


def silhouette_by_metric(X: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> float:
    """
    Compute the Silhouette score using a given distance metric.

    The Silhouette score is an internal evaluation metric that measures how well 
    each object is clustered. Values closer to +1 indicate good separation.

    Args
    ----
    X : np.ndarray
        The data matrix (features, embeddings, etc.) of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster labels assigned to each sample, shape (n_samples,).
    metric : str, optional
        The distance metric to use for score calculation (e.g., 'euclidean', 'manhattan', 'cosine'). 
        Defaults to "euclidean".

    Returns
    -------
    float
        The calculated mean Silhouette score over all samples.
    """
    # The silhouette_score function automatically computes the pairwise distance matrix 
    # using the specified metric and then calculates the score.
    return silhouette_score(X, labels, metric=metric)


def confusion_with_hungarian(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[Any, Any]]:
    """
    Compute the confusion matrix and remap predicted cluster IDs to their most 
    corresponding true labels using the Hungarian algorithm (optimal assignment).

    This is essential for visual comparison when cluster IDs (0, 1, 2, ...) do not 
    naturally correspond to true labels (A, B, C, ...).

    Args
    ----
    y_true : np.ndarray
        True labels, shape (n_samples,).
    y_pred : np.ndarray
        Predicted cluster IDs, shape (n_samples,).
        (y_true and y_pred must have the same length and can be filtered, 
        e.g., to exclude HDBSCAN noise points).

    Returns
    -------
    y_pred_mapped : np.ndarray
        The predicted cluster IDs remapped to true labels based on optimal assignment, 
        shape (n_samples,).
    C : np.ndarray
        The confusion matrix of shape (n_true_labels, n_clusters). 
        Rows = True labels, Cols = Predicted clusters.
    cluster_to_label : dict
        A dictionary mapping the original cluster IDs to the optimally assigned true labels: 
        {cluster_id -> true_label}.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = np.unique(y_true)     # True class labels (e.g., [1, 2, 3, 4, 5, 6] or a subset)
    clusters = np.unique(y_pred)   # Predicted cluster IDs (e.g., [0, 1, 2, 3, 4])

    n_labels = len(labels)
    n_clusters = len(clusters)

    # Degenerate case: no true labels or no clusters found
    if n_labels == 0 or n_clusters == 0:
        C = np.zeros((n_labels, n_clusters), dtype=int)
        return y_pred.copy(), C, {}

    # Contingency matrix C[i, j] = number of points with true label=labels[i] and cluster=clusters[j]
    C = np.zeros((n_labels, n_clusters), dtype=int)
    for i, lab in enumerate(labels):
        for j, clu in enumerate(clusters):
            C[i, j] = np.sum((y_true == lab) & (y_pred == clu))

    # Apply Hungarian algorithm on C. We want to maximize the sum on the assigned elements, 
    # so we minimize the cost matrix (C.max() - C).
    row_ind, col_ind = linear_sum_assignment(C.max() - C)

    # Create the mapping: original cluster ID -> optimal true label
    cluster_to_label = {
        clusters[j]: labels[i]
        for i, j in zip(row_ind, col_ind)
    }

    # Remap the predictions using the calculated mapping
    # Note: .get(c, c) ensures that any cluster ID not in the mapping (e.g., noise label if present) 
    # remains unchanged.
    y_pred_mapped = np.vectorize(lambda c: cluster_to_label.get(c, c))(y_pred)

    return y_pred_mapped, C, cluster_to_label

