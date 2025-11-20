# src/cluster_kshape.py
from typing import Tuple, Optional, Literal
import numpy as np
from tslearn.clustering import KShape
from sklearn.decomposition import PCA


def to_univariate(
    X: np.ndarray,
    mode: Literal["pca1", "var", "mean"] = "pca1",
    var_idx: Optional[int] = None
) -> np.ndarray:
    """
    Convert multivariate time series data (n_samples, n_features, n_timesteps) to 
    univariate data (n_samples, n_timesteps) for k-Shape clustering.

    k-Shape is inherently a univariate clustering algorithm, requiring a dimensionality 
    reduction step for multivariate inputs.

    Args
    ----
    X : np.ndarray
        Input multivariate time series data with shape (n, d, T).
    mode : Literal["pca1", "var", "mean"], optional
        The reduction method:
        - "pca1": Projects the 'd' features onto the first Principal Component (PC1) 
          at each time step 't'. (Default)
        - "var": Selects a single variable/channel for clustering. Requires 'var_idx'.
        - "mean": Computes the average across the 'd' features at each time step 't'.
    var_idx : Optional[int], optional
        The index of the variable/channel to use when 'mode' is "var". Required in this case.

    Returns
    -------
    X_uni : np.ndarray
        The resulting univariate time series data with shape (n, T).

    Raises
    ------
    AssertionError
        If 'mode' is "var" but 'var_idx' is not provided or is invalid.
    """
    n, d, T = X.shape
    if mode == "var":
        # Select a single variable/channel
        assert var_idx is not None and 0 <= var_idx < d, "var_idx must be valid for mode='var'."
        return X[:, var_idx, :]
    elif mode == "mean":
        # Compute the mean across features (axis=1)
        return X.mean(axis=1)
    else:  # "pca1"
        # Project 'd' features onto the first Principal Component (PC1) independently at each time step 't'
        X_uni = np.zeros((n, T), dtype=float)
        # We calculate PCA independently for each timestamp (d -> 1). This is stable if d is small (e.g., 9).
        for t in range(T):
            pca = PCA(n_components=1, random_state=0)
            Xt = X[:, :, t]  # Slice of all samples at time t, shape (n, d)
            comp1 = pca.fit_transform(Xt)  # Projection onto PC1, shape (n, 1)
            X_uni[:, t] = comp1[:, 0]
        # Re-Z-normalize each series after PCA
        X_uni = z_norm_univariate(X_uni)
        return X_uni

def z_norm_univariate(X_uni: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-normalize each univariate series independently along the time axis.

    The z-normalization subtracts the mean and divides by the standard deviation 
    for *each individual time series*.

    Args
    ----
    X_uni : np.ndarray
        Univariate time series data of shape (n_samples, n_timesteps).
    eps : float, optional
        Small constant to prevent division by zero for series with zero standard deviation. 
        Defaults to 1e-8.

    Returns
    -------
    np.ndarray
        The Z-normalized time series data with shape (n_samples, n_timesteps).
    """
    mu = X_uni.mean(axis=1, keepdims=True)  # Mean across time for each series
    sd = X_uni.std(axis=1, keepdims=True)   # Std dev across time for each series
    # Handle zero standard deviation
    sd[sd == 0] = eps
    return (X_uni - mu) / sd

def run_kshape(
    X: np.ndarray,
    n_clusters: int = 6,
    n_init: int = 10,
    max_iter: int = 50,
    random_state: int = 42,
    reduce_mode: Literal["pca1", "var", "mean"] = "pca1",
    var_idx: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, KShape, np.ndarray, np.ndarray]:
    """
    Run k-Shape clustering on time series data.

    If the input data 'X' is multivariate (n, d, T), it is first reduced to 
    univariate (n, T) using the specified 'reduce_mode'. Each series is 
    then Z-normalized before fitting the KShape model.

    Args
    ----
    X : np.ndarray
        Input time series data. Expected shape is (n, d, T) for multivariate or 
        (n, T) for already univariate data.
    n_clusters : int, optional
        The number of clusters to form (cf. KShape in tslearn). Defaults to 6.
    n_init : int, optional
        Number of random initializations (cf. KShape in tslearn). Defaults to 10.
    max_iter : int, optional
        Maximum number of iterations for the k-Shape updates (cf. KShape in tslearn). Defaults to 50.
    random_state : int, optional
        Reproducibility seed (cf. KShape in tslearn). Defaults to 42.
    reduce_mode : Literal["pca1", "var", "mean"], optional
        Method used to convert multivariate data to univariate. Used only if X is 3D. 
        Defaults to "pca1".
    var_idx : Optional[int], optional
        The index of the variable to use if reduce_mode="var". Required in this case.
    verbose : bool, optional
        Controls the verbosity of the tslearn model fitting. Defaults to False.

    Returns
    -------
    y_pred : np.ndarray
        Cluster labels for each sample, shape (n,).
    model : KShape
        The fitted tslearn KShape model object.
    centers : np.ndarray
        The computed cluster centroids. Shape (n_clusters, T, 1) as returned by tslearn.
    X_uni : np.ndarray
        The univariate representation of the data used for clustering, shape (n, T).

    Raises
    ------
    ValueError
        If the input array 'X' does not have 2 or 3 dimensions.
    """
    # 1) Reduce to univariate if necessary
    if X.ndim == 3:
        # Convert (n, d, T) -> (n, T)
        X_uni = to_univariate(X, mode=reduce_mode, var_idx=var_idx)
    elif X.ndim == 2:
        # Already (n, T)
        X_uni = X
    else:
        raise ValueError("X must be (n, d, T) or (n, T).")

    # 2) Explicitly z-normalize each series (k-Shape assumes this; we align explicitly)
    X_uni = z_norm_univariate(X_uni)

    # 3) Reshape to tslearn format: (n, T, 1)
    X_ts = X_uni[:, :, None]

    # 4) Fit the KShape model
    model = KShape(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose
    )
    y_pred = model.fit_predict(X_ts)
    centers = model.cluster_centers_  # (k, T, 1)
    return y_pred, model, centers, X_uni