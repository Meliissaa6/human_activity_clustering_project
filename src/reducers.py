import umap
from sklearn.decomposition import PCA
from typing import Literal
import numpy as np


def apply_umap(
    X: np.ndarray, 
    n_components: int, 
    n_neighbors: int, 
    min_dist: float, 
    metric: str, 
    random_state: int
) -> np.ndarray:
    """
    Create, fit, and transform the input data using the UMAP (Uniform Manifold 
    Approximation and Projection) dimensionality reduction algorithm.

    UMAP is a manifold learning technique often used for visualization and as 
    a preprocessing step for clustering due to its ability to preserve local 
    and global data structure.

    Args
    ----
    X : np.ndarray
        The input data matrix of shape (n_samples, n_features).
    n_components : int
        The dimension of the space to embed into (e.g., 2 for visualization).
    n_neighbors : int
        The size of the local neighborhood (controls the balance between local 
        and global structure preservation).
    min_dist : float
        The minimum distance between embedded points (controls how tightly points 
        are packed together).
    metric : str
        The distance metric to use in the input space (e.g., 'euclidean', 'cosine').
    random_state : int
        The seed for reproducibility.

    Returns
    -------
    X_umap : np.ndarray
        The low-dimensional embedding of the data, shape (n_samples, n_components).
    """
    # Create the UMAP model with predefined parameters
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    # Fit the model and transform the data
    X_umap = umap_model.fit_transform(X)
    return X_umap


def apply_pca(
    X: np.ndarray, 
    n_components: float | int, 
    svd_solver: Literal["auto", "full", "arpack", "randomized"] = "auto"
) -> np.ndarray:
    """
    Apply Principal Component Analysis (PCA) for linear dimensionality reduction.

    PCA is often used for noise reduction and decorrelating features. The number 
    of components can be specified directly or as a proportion of variance to retain.

    Args
    ----
    X : np.ndarray
        The input data matrix of shape (n_samples, n_features).
    n_components : float or int
        Number of components to keep. If $0 < n\_components < 1$, it is treated as 
        the proportion of variance that should be explained (e.g., 0.95). 
        If integer, it is the exact number of components.
    svd_solver : str, optional
        The solver to use for the SVD computation. Defaults to "auto".

    Returns
    -------
    X_pca : np.ndarray
        The data projected onto the principal components, shape (n_samples, n_components).
    """
    # Create the PCA model with specified number of components (or variance)
    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    
    # Fit the model and transform the data
    X_pca = pca.fit_transform(X)
    
    return X_pca