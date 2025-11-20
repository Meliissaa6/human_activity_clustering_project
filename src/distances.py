# src/distances.py 
import numpy as np
from tslearn.metrics import cdist_dtw
from numpy.linalg import norm
from numpy.fft import fft, ifft
from sklearn.metrics.pairwise import pairwise_distances
from typing import Optional, Tuple


def dtw_distance_matrix(X_tslearn: np.ndarray, sakoe_chiba_radius: int = 5) -> np.ndarray:
    """
    Compute the pairwise Dynamic Time Warping (DTW) distance matrix.

    This uses the fast C-optimized implementation from tslearn.

    Args
    ----
    X_tslearn : np.ndarray
        Input time series array with shape (n_samples, n_timesteps, n_features). 
        (Assumed to be already transposed for tslearn format).
    sakoe_chiba_radius : int, optional
        Warping window size for the Sakoe-Chiba band constraint. 
        A smaller value speeds up computation. Defaults to 5.

    Returns
    -------
    D : np.ndarray
        The pairwise DTW distance matrix of shape (n_samples, n_samples).
    """
    return cdist_dtw(
        X_tslearn,
        global_constraint="sakoe_chiba",
        sakoe_chiba_radius=sakoe_chiba_radius
    )


# ---------- SBD (Shape-Based Distance) helpers ----------

def roll_zeropad(a: np.ndarray, shift: int, axis: Optional[int] = None) -> np.ndarray:
    """
    Roll elements of an array with zero padding (non-circular roll).

    Args
    ----
    a : np.ndarray
        The input array.
    shift : int
        The number of places by which elements are shifted. 
        Positive shift means rolling to the right/down.
    axis : int or None, optional
        The axis along which elements are shifted. If None, the array is flattened 
        before shifting. Defaults to None.

    Returns
    -------
    np.ndarray
        The shifted array with zeros padded in the vacated positions.
    """
    a = np.asanyarray(a)
    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    
    # Absolute shift greater than array size results in all zeros
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    # Negative shift (roll left/up)
    elif shift < 0:
        shift += n # Convert negative shift to positive shift in opposite direction
        # Zeros are padded at the end
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        # Take from the end, then append zeros
        res = np.concatenate((a.take(np.arange(n-shift, n), axis), zeros), axis)
    # Positive shift (roll right/down)
    else:
        # Zeros are padded at the beginning
        zeros = np.zeros_like(a.take(np.arange(n-shift, n), axis))
        # Prepend zeros, then take from the beginning
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
        
    if reshape:
        return res.reshape(a.shape)
    else:
        return res


def _ncc_c(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Normalized Cross-Correlation (NCC) between two 1D signals x and y,
    computed efficiently via Fast Fourier Transform (FFT).

    The signals are assumed to be zero-mean and unit-variance (Z-normalized) 
    prior to calling the outer SBD function.

    Args
    ----
    x : np.ndarray
        The first 1D signal.
    y : np.ndarray
        The second 1D signal.

    Returns
    -------
    np.ndarray
        A vector of correlation values for all possible shifts between x and y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Normalization factor
    den = np.array(norm(x) * norm(y))
    # Handle zero norm (flat, non-z-normalized series)
    if den == 0:
        den = np.Inf

    x_len = len(x)
    # Compute the size of the FFT window for convolution
    fft_size = 1 << (2 * x_len - 1).bit_length()
    
    # Compute cross-correlation using the Convolution Theorem: ifft(fft(x) * conj(fft(y)))
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    
    # Reorder the correlation results to represent shifts from -(N-1) to N-1
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
    
    # Return the real part of the normalized cross-correlation
    return np.real(cc) / den


def _sbd(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute the Shape-Based Distance (SBD) between two 1D series x and y.

    SBD is defined as $1 - \max(NCC(x, y))$ where NCC is the Normalized Cross-Correlation.

    Args
    ----
    x : np.ndarray
        The first 1D time series (should be z-normalized).
    y : np.ndarray
        The second 1D time series (should be z-normalized).

    Returns
    -------
    dist : float
        The SBD distance value.
    yshift : np.ndarray
        The shifted version of 'y' that achieves the best alignment (maximum NCC) 
        with 'x', padded with zeros.

    Examples
    --------
    >>> # SBD between identical, normalized series should be close to 0
    >>> _sbd([1,1,1]/norm([1,1,1]), [1,1,1]/norm([1,1,1]))[0]
    0.0
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Compute normalized cross-correlation for all shifts
    ncc = _ncc_c(x, y)
    
    # Find the index corresponding to the maximum correlation
    idx = ncc.argmax()
    
    # SBD is 1 minus the maximum NCC
    dist = 1.0 - ncc[idx]
    
    # Compute the required roll shift value
    shift_val = (idx + 1) - max(len(x), len(y))
    
    # Apply the optimal shift to y (with zero padding)
    yshift = roll_zeropad(y, shift_val)
    
    return dist, yshift


def sbd_distance_matrix(X_uni: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Shape-Based Distance (SBD) matrix for univariate time series.

    The computation iterates through all unique pairs and calculates the SBD.

    Args
    ----
    X_uni : np.ndarray
        Array of univariate time series with shape (n_samples, n_timesteps). 
        The series must be **z-normalized** for correct SBD calculation.

    Returns
    -------
    D : np.ndarray
        The pairwise distance matrix of shape (n_samples, n_samples), which is 
        symmetric with zeros on the diagonal.
    """
    X_uni = np.asarray(X_uni, dtype=float)
    n, T = X_uni.shape
    D = np.zeros((n, n), dtype=float)

    for i in range(n):
        D[i, i] = 0.0
        xi = X_uni[i]
        for j in range(i + 1, n):
            xj = X_uni[j]
            # SBD is symmetric, so we only compute it once per pair
            dist, _ = _sbd(xi, xj)
            D[i, j] = dist
            D[j, i] = dist # Fill the symmetric element
    return D


def distance_matrix(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Compute the pairwise distance matrix using a standard metric (e.g., Euclidean).

    This function is primarily used for feature-based clustering on representations 
    like Fourier coefficients.

    Args
    ----
    X : np.ndarray
        Data matrix of shape (n_samples, n_features). 
        For example, a matrix of Fourier coefficients.
    metric : str, optional
        The distance metric to use (e.g., "euclidean", "cosine", "manhattan"). 
        Defaults to "euclidean".

    Returns
    -------
    D : np.ndarray
        The pairwise distance matrix of shape (n_samples, n_samples).
    """
    # Use scikit-learn's optimized pairwise distance function
    return pairwise_distances(X, metric=metric)

