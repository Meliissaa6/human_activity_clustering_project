import numpy as np 


def z_norm_per_series(X: np.ndarray) -> np.ndarray:
    """
    Apply Z-normalization (standardization) to each individual time series 
    independently across the time axis.

    The Z-normalization transforms each series such that its mean is 0 and its 
    standard deviation is 1. This is crucial for shape-based time series analysis 
    as it removes amplitude and offset differences.

    Args
    ----
    X : np.ndarray
        Input time-series tensor of shape (n_samples, n_features, n_timesteps). 
        The normalization is applied along axis=2 (n_timesteps).

    Returns
    -------
    np.ndarray
        The Z-normalized tensor with the same shape as X.
    """
    X = X.astype(float)
    
    # Calculate the mean across the time axis (axis=2) for each sample and feature
    mu = X.mean(axis=2, keepdims=True)
    
    # Calculate the standard deviation across the time axis (axis=2)
    sigma = X.std(axis=2, keepdims=True)
    
    # Avoid division by zero by setting zero standard deviations to a small epsilon
    sigma[sigma == 0] = 1e-8
    
    # Apply the Z-normalization formula: (X - mean) / std_dev
    X_norm = (X - mu) / sigma
    
    return X_norm