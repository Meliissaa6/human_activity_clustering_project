# src/io.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

# Define the expected names and order of the 9 variables (channels)
VAR_NAMES = [
    "accm_x", "accm_y", "accm_z",
    "acce_x", "acce_y", "acce_z",
    "vit_x", "vit_y", "vit_z"
]


def _load_txt_matrix(path: str) -> np.ndarray:
    """
    Load a .txt file (space/tab separated) into a float matrix.
    
    This function handles multiple spaces/tabs as separators and ensures the 
    resulting array is at least 2D (even for single-line files). It also checks 
    for missing files and potential NaN values.

    Args
    ----
    path : str
        The full path to the .txt file.

    Returns
    -------
    np.ndarray
        The data matrix of shape (n_samples, n_timesteps).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If reading fails or if NaN values are detected in the data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    try:
        # Load data, robustly handling multiple spaces and empty lines
        arr = np.loadtxt(path, dtype=float)
    except Exception as e:
        raise ValueError(f"Failed to read (float) for {path}: {e}")
    
    if arr.ndim == 1:
        # If the file is a single line -> force 2D shape (1, T)
        arr = arr[None, :]
        
    if np.isnan(arr).any():
        # Check for NaN values
        where = np.argwhere(np.isnan(arr))
        raise ValueError(f"NaNs detected in {path} at positions {where[:5]} (…)")
        
    return arr  # Shape (n_samples, n_timesteps)

def _load_labels_txt(path: str, n_expected: int) -> np.ndarray:
    """
    Load labels from a space-separated lab.txt file, handling both numeric and 
    textual labels, and ensuring consistency.

    Args
    ----
    path : str
        The full path to the lab.txt file.
    n_expected : int
        The expected number of labels (must match the number of time series windows).

    Returns
    -------
    np.ndarray
        The label vector of shape (n_samples,). Labels are converted to integers 
        (0 to K-1) if they were originally strings.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If reading fails, the label count is inconsistent, or NaN values are found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    # Read as strings first for tolerance (avoids parsing NaNs)
    try:
        raw = np.loadtxt(path, dtype=str)
    except Exception as e:
        raise ValueError(f"Failed to read labels (str) for {path}: {e}")

    raw = np.ravel(raw)  # Flatten the array (row or column format)
    # Remove any potential empty fields
    raw = np.array([s for s in raw if s != "" and s.strip() != ""], dtype=str)

    if raw.size != n_expected:
        raise ValueError(
            f"lab.txt contains {raw.size} labels, but {n_expected} windows were read. "
            "Check the file format (row/column) and empty lines."
        )

    # Attempt numeric conversion
    try:
        y_float = raw.astype(float)
        if np.isnan(y_float).any():
            raise ValueError("NaN after float conversion")
        
        # If they are integers coded as floats (1.0, 2.0, ...) -> cast to int
        if np.allclose(y_float, np.round(y_float)):
            y = y_float.astype(int)
        else:
            y = y_float
            
    except Exception:
        # Textual labels -> factorization into integers 0..K-1
        classes, y_idx = np.unique(raw, return_inverse=True)
        y = y_idx  # Integers 0..K-1
        
    return y  # Shape (n_samples,)

def load_raw_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the 9 variable matrices and the labels from the specified directory.

    The matrices are stacked to form a single 3D tensor, and labels are loaded 
    and validated for consistency.

    Args
    ----
    data_dir : str
        The path to the directory containing 'accm_x.txt', 'lab.txt', etc.

    Returns
    -------
    X : np.ndarray
        The time series tensor of shape (n_samples, 9, n_timesteps).
    y : np.ndarray
        The label vector of shape (n_samples,).

    Raises
    ------
    ValueError
        If the number of samples (rows) is inconsistent across the variable files.
    """
    matrices = []
    n_samples = None
    # Load each variable file
    for var in VAR_NAMES:
        path = os.path.join(data_dir, f"{var}.txt")
        arr = _load_txt_matrix(path)  # Shape (n, T)
        
        if n_samples is None:
            n_samples = arr.shape[0]
        elif arr.shape[0] != n_samples:
            # Check for consistency in the number of samples
            raise ValueError(
                f"Inconsistent number of rows for {var}.txt: "
                f"{arr.shape[0]} vs {n_samples} in other files."
            )
        matrices.append(arr)

    # Stack the matrices along a new axis (axis=1) -> (n, 9, T)
    X = np.stack(matrices, axis=1)

    # Load labels using the determined number of samples
    y_path = os.path.join(data_dir, "lab.txt")
    y = _load_labels_txt(y_path, n_expected=n_samples)

    return X, y

def train_test_split_data(
    X: np.ndarray, 
    y: np.ndarray, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data (windows) into training and testing sets.

    Stratification is used based on 'y' labels unless all labels are identical, 
    to ensure proportional representation of classes in both splits.

    Args
    ----
    X : np.ndarray
        The data tensor, shape (n_samples, n_variables, n_timesteps).
    y : np.ndarray
        The label vector, shape (n_samples,).
    test_size : float, optional
        The proportion of the data to use for the test set. Defaults to 0.2.
    random_state : int, optional
        The random seed for reproducibility. Defaults to 42.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_test, y_train, y_test.
    """
    # Disable stratification if there is only one unique class to avoid sklearn error
    strat = y if np.unique(y).size > 1 else None
    idx = np.arange(len(y))
    
    # Split the indices
    tr_idx, te_idx = train_test_split(
        idx, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=strat
    )
    
    # Use the indices to split the data
    return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

def load_dataset(
    data_dir: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete data pipeline: load raw data and split into train/test sets.

    Args
    ----
    data_dir : str
        The path to the directory containing the data files.
    test_size : float, optional
        The proportion of the data to use for the test set. Defaults to 0.2.
    random_state : int, optional
        The random seed for reproducibility. Defaults to 42.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_test, y_train, y_test.
    """
    # 1. Load the raw data and labels
    X, y = load_raw_data(data_dir)
    
    # 2. Split the data
    return train_test_split_data(X, y, test_size=test_size, random_state=random_state)