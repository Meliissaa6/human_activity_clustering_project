# dans src/cluster_fourier_kmeans.py
import numpy as np
from typing import Tuple


def compute_fft_spectrum(
    X: np.ndarray,
    sampling_rate: float,
    center: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the amplitude spectra (FFT) for each time series sample and each variable.

    The Real Fast Fourier Transform (RFFT) is used to efficiently compute the 
    spectrum for real-valued input, returning only the non-negative frequency bins.

    Parameters
    ----------
    X : np.ndarray
        Input time-series tensor of shape (n_samples, n_variables, n_timesteps).
    sampling_rate : float
        The sampling frequency (in Hz) of the signals (e.g., 50 Hz).
    center : bool, optional
        If True, the mean of each individual time series (per sample and variable) 
        is subtracted before computing the FFT. This removes the DC component (freq=0). 
        Defaults to True.

    Returns
    -------
    freqs : np.ndarray
        Array of shape (n_freqs,) containing the positive frequency bins in Hz.
    amp : np.ndarray
        Amplitude spectra of shape (n_samples, n_variables, n_freqs). 
        Contains only non-negative frequencies (output of rfft).

    Raises
    ------
    AssertionError
        If X is not 3-dimensional.
    """
    assert X.ndim == 3, "X must be (n_samples, n_variables, n_timesteps)"
    n, d, T = X.shape
    X_proc = X.astype(float)
    
    if center:
        # Subtract the mean (DC component) per sample and variable along the time axis (axis=2)
        mu = X_proc.mean(axis=2, keepdims=True)
        X_proc = X_proc - mu
        
    # Compute RFFT along the time axis (axis=2). rfft returns only non-negative frequencies.
    amp_complex = np.fft.rfft(X_proc, axis=2)  # (n, d, n_freqs) complex
    amp = np.abs(amp_complex)                  # Get the amplitude spectrum
    
    # Compute the frequency bins corresponding to the rfft output
    freqs = np.fft.rfftfreq(T, d=1.0 / sampling_rate)  # (n_freqs,)
    
    return freqs, amp


def extract_fourier_features(amp_spectra: np.ndarray, n_coeffs: int) -> np.ndarray:
    """
    Extract and flatten the top 'n_coeffs' amplitude spectrum coefficients 
    for use as features in clustering.

    The coefficient corresponding to frequency 0 (DC component/mean) is typically 
    ignored as it often contains limited shape information.

    Args
    ----
    amp_spectra : np.ndarray
        Amplitude spectra array of shape (n_samples, n_variables, n_freqs), 
        as returned by compute_fft_spectrum.
    n_coeffs : int
        The number of low-frequency coefficients to extract per variable.

    Returns
    -------
    X_features_2D : np.ndarray
        The flattened feature matrix of shape (n_samples, n_variables * n_coeffs).
    """
    
    # 1. Select the first n_coeffs positive frequencies (index 1 to n_coeffs).
    # We ignore the 0th frequency (index 0) which corresponds to the mean (DC component).
    X_features_3D = amp_spectra[:, :, 1 : n_coeffs + 1]
    
    N_samples = X_features_3D.shape[0]
    
    # 2. Flatten the features: (n_samples, n_variables, n_coeffs) -> (n_samples, n_variables * n_coeffs)
    X_features_2D = X_features_3D.reshape(N_samples, -1)
    
    return X_features_2D


