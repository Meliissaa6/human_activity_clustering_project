from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import skew, kurtosis
from typing import List


def add_derivative_channels(X: np.ndarray) -> np.ndarray:
    """
    Compute the discrete derivative of the time series and concatenate it with 
    the original data.

    The original data is truncated by one time step to match the length of the derivative.

    Args
    ----
    X : np.ndarray
        Time-series tensor of shape (n_samples, n_variables, n_timesteps).

    Returns
    -------
    np.ndarray
        The augmented tensor of shape (n_samples, 2 * n_variables, n_timesteps - 1), 
        containing [Original Data, Derivative Data].
    """
    # Compute the discrete derivative along the time axis (axis=2)
    dX = np.diff(X, axis=2)  # Shape: (n_samples, n_variables, n_timesteps - 1)
    
    # Truncate the original data to match the length of the derivative (removing the first time step)
    X_cut = X[:, :, 1:]
    
    # Concatenate the truncated original data and the derivative along the variable axis (axis=1)
    X_aug = np.concatenate([X_cut, dX], axis=1)
    
    return X_aug


def concatenate_features(X_time: np.ndarray, X_freq: np.ndarray) -> np.ndarray:
    """
    Concatenate time-domain and frequency-domain feature matrices.

    Args
    ----
    X_time : np.ndarray
        Time-domain feature matrix, shape (n_samples, n_features_time).
    X_freq : np.ndarray
        Frequency-domain feature matrix, shape (n_samples, n_features_freq).

    Returns
    -------
    np.ndarray
        The combined feature matrix, shape (n_samples, n_features_time + n_features_freq).
    """
    return np.concatenate([X_time, X_freq], axis=1)


def scale_features(X_feat: np.ndarray) -> np.ndarray:
    """
    Apply standard scaling (Z-score normalization) to the feature matrix.

    Standardization centers the data around zero (mean=0) and scales it to unit variance (std=1).

    Args
    ----
    X_feat : np.ndarray
        The input feature matrix of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        The standardized feature matrix.
    """
    scaler = StandardScaler()
    # fit_transform calculates the mean and standard deviation on the input data 
    # and then applies the transformation.
    return scaler.fit_transform(X_feat)


def compute_time_features(X: np.ndarray) -> np.ndarray:
    """
    Compute a comprehensive set of time-domain statistical features for each sample.

    Features are computed for each individual variable and for the overall signal magnitude.

    Args
    ----
    X : np.ndarray
        Time-series tensor of shape (n_samples, n_variables, n_timesteps).

    Returns
    -------
    X_time : np.ndarray
        The resulting time-domain feature matrix of shape (n_samples, n_features_time). 
        The number of features is (n_variables * 7 + 3).

    Raises
    ------
    AssertionError
        If X is not 3-dimensional.
    """
    assert X.ndim == 3
    n, d, T = X.shape
    feats: List[np.ndarray] = []
    
    # --- 1. Per-variable statistical features (7 features * d variables) ---
    # Calculated across the time axis (axis=2). Result shape: (n_samples, n_variables)
    mean_    = X.mean(axis=2)
    std_     = X.std(axis=2)
    min_     = X.min(axis=2)
    max_     = X.max(axis=2)
    energy_  = (X**2).mean(axis=2)
    
    # Skewness and Kurtosis (unbiased estimate: bias=False)
    skew_    = skew(X, axis=2, bias=False)
    kurt_    = kurtosis(X, axis=2, bias=False)
    
    # --- 2. Global Magnitude features (3 features) ---
    # Compute signal magnitude across variables (axis=1). Result shape: (n_samples, n_timesteps)
    mag = np.linalg.norm(X, axis=1) 
    
    # Compute mean, std, and energy for the magnitude series
    # keepdims=True ensures the shape is (n_samples, 1) for concatenation
    mag_mean   = mag.mean(axis=1, keepdims=True)
    mag_std    = mag.std(axis=1, keepdims=True)
    mag_energy = (mag**2).mean(axis=1, keepdims=True)
    
    # --- 3. Concatenate all features ---
    feats.append(mean_)
    feats.append(std_)
    feats.append(min_)
    feats.append(max_)
    feats.append(energy_)
    feats.append(skew_)
    feats.append(kurt_)
    feats.append(mag_mean)
    feats.append(mag_std)
    feats.append(mag_energy)
    
    # Concatenate all feature arrays along the feature axis (axis=1)
    X_time = np.concatenate(feats, axis=1)  # Shape: (n_samples, d*7 + 3)
    
    return X_time