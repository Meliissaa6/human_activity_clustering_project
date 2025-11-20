import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import seaborn as sns
from sklearn.manifold import TSNE, MDS
from typing import Dict, Optional, Tuple, List, Any


def plot_one_sample_per_label(
    X: np.ndarray, 
    y: np.ndarray, 
    label_to_activity: Dict[int, str], 
    n_cols: int = 2, 
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Display one representative sample (window) for each unique label in a grid of subplots.
    Each subplot shows the time-series profile of all 9 sensor variables over time 
    for the corresponding activity.

    Args
    ----
    X : np.ndarray
        Dataset of shape (n_samples, n_variables, n_timesteps).
    y : np.ndarray
        Corresponding labels of shape (n_samples,).
    label_to_activity : Dict[int, str]
        Mapping from integer label to human-readable activity name.
    n_cols : int, optional
        Number of columns in the subplot grid. Defaults to 2.
    figsize : Tuple[int, int], optional
        Overall figure size (width, height). Defaults to (15, 10).
    """
    unique_labels = np.unique(y)
    n_labels = len(unique_labels)
    n_rows = int(np.ceil(n_labels / n_cols))

    # Select one representative sample per label (first occurrence)
    one_index_per_label = {lab: np.where(y == lab)[0][0] for lab in unique_labels}

    # Compute consistent Y-axis limits for better amplitude comparison
    y_min = X.min()
    y_max = X.max()

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, lab in enumerate(unique_labels):
        idx = one_index_per_label[lab]
        ax = axes[i]
        # Plot all 9 variables for the selected sample
        for v in range(X.shape[1]):
            ax.plot(X[idx, v, :], label=f"Var {v}", linewidth=1)
        
        # Apply consistent Y-axis scale
        ax.set_ylim(y_min, y_max)
        activity = label_to_activity.get(lab, "Unknown")
        ax.set_title(f"Label {lab} - {activity} (sample {idx})", fontsize=10)
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Value")
        ax.grid(True)
        
    # Hide unused subplots
    for j in range(n_labels, len(axes)):
        axes[j].axis("off")
        
    # Shared legend and layout
    fig.legend([f"Var {v}" for v in range(X.shape[1])], loc="lower center", ncol=X.shape[1], fontsize=8)
    fig.suptitle("One representative sample per label (all variables)", fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust for legend
    plt.show()


def plot_activity_variability(
    X: np.ndarray, 
    y: np.ndarray, 
    label_to_activity: Dict[int, str], 
    var_idx: int = 0
):
    """
    Plot the mean ± standard deviation profile of a selected sensor signal 
    over time for each activity label.

    Parameters
    ----------
    X : np.ndarray
        3D tensor of shape (n_samples, n_variables, n_timesteps).
    y : np.ndarray
        Label vector of shape (n_samples,).
    label_to_activity : Dict[int, str]
        Mapping from numeric label to activity name.
    var_idx : int, optional
        Index of the variable to plot (0 to n_variables-1). Defaults to 0.
    """
    plt.figure(figsize=(10, 6))
    for lab in np.unique(y):
        # Select all windows for this activity and variable
        X_lab = X[y == lab, var_idx, :]
        mean = X_lab.mean(axis=0)  # Average time profile
        std = X_lab.std(axis=0)    # Variability around the mean
        
        plt.plot(mean, label=f"{label_to_activity.get(lab, str(lab))}")
        # Plot the region of $\mu \pm \sigma$
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
        
    plt.title(f"Temporal variability – Variable {var_idx}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_all_activity_variability(
    X: np.ndarray, 
    y: np.ndarray, 
    label_to_activity: Dict[int, str], 
    var_names: Optional[List[str]] = None
):
    """
    Plot the mean ± std profile for all sensor variables in a grid of subplots.

    Parameters
    ----------
    X : np.ndarray
        3D tensor of shape (n_samples, n_variables, n_timesteps).
    y : np.ndarray
        Label vector of shape (n_samples,).
    label_to_activity : Dict[int, str]
        Mapping from numeric label to activity name.
    var_names : Optional[List[str]], optional
        List of variable names for subplot titles. Length must equal n_variables. 
        Defaults to None.
    """
    n_vars = X.shape[1]
    n_rows, n_cols = 3, 3 # Assuming 9 variables
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.flatten()

    for v in range(n_vars):
        ax = axes[v]
        for lab in np.unique(y):
            # Calculate mean and std for the current variable (v) and activity (lab)
            X_lab = X[y == lab, v, :]
            mean = X_lab.mean(axis=0)
            std = X_lab.std(axis=0)
            
            ax.plot(mean, label=label_to_activity.get(lab, str(lab)))
            ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
            
        title = var_names[v] if var_names is not None and v < len(var_names) else f"Var {v}"
        ax.set_title(title, fontsize=10)
        ax.grid(True)
        # Add Y-label only to the first column
        if v % n_cols == 0:
            ax.set_ylabel("Amplitude")
        # Add X-label only to the bottom row
        if v >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Time (samples)")

    # Unified legend (get handles/labels from the last plot)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(np.unique(y)), fontsize=8)
    fig.suptitle("Temporal variability of all variables by activity", fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust for legend
    plt.show()


def plot_fft_per_activity(
    X: np.ndarray, 
    y: np.ndarray, 
    label_to_activity: Dict[int, str], 
    var_idx: int = 0, 
    fs: float = 50
):
    """
    Plot the average magnitude frequency spectrum (FFT) of a selected sensor variable 
    for each activity label.

    Parameters
    ----------
    X : np.ndarray
        The dataset tensor of shape (n_samples, n_variables, n_timesteps).
    y : np.ndarray
        Array of integer labels of shape (n_samples,).
    label_to_activity : Dict[int, str]
        Dictionary mapping label integers to human-readable activity names.
    var_idx : int, optional
        Index of the variable (0 to n_variables-1) to analyze. Defaults to 0.
    fs : float, optional
        Sampling frequency of the signals in Hz. Defaults to 50.
    """
    plt.figure(figsize=(10, 6))

    for lab in np.unique(y):
        # Select all windows for the given activity and variable
        X_lab = X[y == lab, var_idx, :]
        
        # Compute RFFT magnitude for each window along the time axis (axis=1)
        fft_vals = np.abs(rfft(X_lab, axis=1))
        
        # Compute corresponding frequency bins
        freq = rfftfreq(X_lab.shape[1], d=1/fs)
        
        # Average FFT magnitudes across all windows of this activity
        mean_fft = fft_vals.mean(axis=0)
        
        # Plot mean spectrum
        plt.plot(freq, mean_fft, label=label_to_activity.get(int(lab), f"Label {lab}"))
        
    plt.title(f"Average Frequency Spectrum - Variable {var_idx}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.legend(title="Activity", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_fft_grid(
    X: np.ndarray, 
    y: np.ndarray, 
    label_to_activity: Dict[int, str], 
    var_names: Optional[List[str]] = None, 
    fs: float = 50, 
    n_cols: int = 3, 
    figsize: Tuple[int, int] = (14, 10), 
    normalize: bool = False, 
    legend_out: bool = True
):
    """
    Plot the average FFT magnitude spectrum for all variables in a grid of subplots.

    Parameters
    ----------
    X : np.ndarray
        Time-series tensor of shape (n_samples, n_variables, n_timesteps).
    y : np.ndarray
        Integer labels of shape (n_samples,).
    label_to_activity : Dict[int, str]
        Mapping from numeric labels to human-readable activity names.
    var_names : Optional[List[str]], optional
        Names of variables. Defaults to ['Var 0', 'Var 1', ...].
    fs : float, optional
        Sampling frequency (Hz). Defaults to 50.
    n_cols : int, optional
        Number of columns in the subplot grid. Defaults to 3.
    figsize : Tuple[int, int], optional
        Matplotlib figure size (width, height). Defaults to (14, 10).
    normalize : bool, optional
        If True, normalize spectra per variable (per subplot) by the maximum amplitude. 
        Defaults to False.
    legend_out : bool, optional
        If True, draws one global legend outside the grid. Defaults to True.
    """
    n_samples, n_vars, n_timesteps = X.shape
    labels = np.unique(y)
    if var_names is None or len(var_names) != n_vars:
        var_names = [f"Var {i}" for i in range(n_vars)]

    # Frequency axis (same for all plots)
    freq = rfftfreq(n_timesteps, d=1/fs)
    
    # Grid size
    n_rows = int(np.ceil(n_vars / n_cols))
    # squeeze=False ensures axes is always 2D, even if n_rows=1 or n_cols=1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    global_handles = []
    global_labels = []

    for v in range(n_vars):
        ax = axes[v]
        mean_spectra: List[Tuple[Any, np.ndarray]] = []
        
        # 1. Compute and collect mean spectra for all activities
        for lab in labels:
            X_lab = X[y == lab, v, :]
            if X_lab.size == 0:
                continue
            fft_vals = np.abs(rfft(X_lab, axis=1))
            mean_fft = fft_vals.mean(axis=0)
            mean_spectra.append((lab, mean_fft))
            
        # 2. Optional normalization per subplot
        if normalize and len(mean_spectra) > 0:
            max_val = max(s[1].max() for s in mean_spectra if s[1].size > 0)
            if max_val > 0:
                # Apply normalization: divide by max amplitude found across all activities for this variable
                mean_spectra = [(lab, spec / max_val) for lab, spec in mean_spectra]

        # 3. Plot all mean spectra
        for lab, spec in mean_spectra:
            lbl = label_to_activity.get(int(lab), f"Label {lab}")
            h, = ax.plot(freq, spec, label=lbl, linewidth=1.5)
            # Collect handles/labels for the global legend
            if lbl not in global_labels:
                global_handles.append(h)
                global_labels.append(lbl)

        ax.set_title(var_names[v])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude" + (" (norm.)" if normalize else ""))
        ax.grid(True)
        # Limit X-axis to the Nyquist frequency (fs/2)
        ax.set_xlim(0, fs / 2)

    # Hide unused axes
    for k in range(n_vars, len(axes)):
        axes[k].axis("off")

    # Global legend
    if legend_out:
        # Place legend above the subplots
        fig.legend(global_handles, global_labels, loc="upper center", ncol=len(global_labels)//2 or 1)
        plt.subplots_adjust(top=0.88)  # Leave room for the legend
    else:
        # Place local legends on each subplot
        for ax in axes[:n_vars]:
            ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()


def plot_correlation_matrix(X: np.ndarray):
    """
    Compute the correlation matrix between the time series variables and display it 
    as a heatmap (lower triangle only).

    The correlation is calculated across all samples and all time steps.

    Parameters
    ----------
    X : np.ndarray
        Time-series tensor of shape (n_samples, n_variables, n_timesteps). 
        Expected n_variables is 9.

    Raises
    ------
    ValueError
        If X does not have the expected 3D shape with 9 variables.
    """
    if X.ndim != 3 or X.shape[1] != 9:
        raise ValueError("Expected X to have shape (n_samples, 9, timesteps).")

    # Reshape the data from (n_samples, 9, timesteps) to (n_samples * timesteps, 9) 
    # to compute correlation across all data points
    X_flat = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
    # Compute correlation matrix between the 9 variables (columns)
    corr = np.corrcoef(X_flat.T)

    # Create mask for the upper triangle (to display only the lower triangle)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Plot lower triangle heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        xticklabels=[f"Var {i}" for i in range(9)],
        yticklabels=[f"Var {i}" for i in range(9)],
        square=True, cbar_kws={"shrink": .8}
    )
    plt.title("Correlation between the 9 sensor variables (lower triangle)")
    plt.tight_layout()
    plt.show()


def plot_cluster_centroids(
    centers: np.ndarray, 
    vars_idx: Tuple[int, ...] = (0, 3, 6), 
    suptitle: str = "DBA Cluster Centroids"
):
    """
    Plot the cluster centroids for a subset of variables in a grid (K rows x V columns).

    Parameters
    ----------
    centers : np.ndarray
        Cluster centroids of shape (n_clusters, n_timesteps, n_variables).
    vars_idx : Tuple[int, ...], optional
        Indices of the variables to plot. Defaults to (0, 3, 6).
    suptitle : str, optional
        Title for the entire figure. Defaults to "DBA Cluster Centroids".
    """
    k, T, d = centers.shape
    n_cols = len(vars_idx)
    
    # Create subplot grid
    fig, axes = plt.subplots(k, n_cols, figsize=(4 * n_cols, 2.2 * k), sharex=True)
    # Ensure axes is always 2D, even if k=1 (single row)
    if k == 1: axes = np.atleast_2d(axes) 
    
    for ci in range(k): # Loop over clusters (rows)
        for j, v in enumerate(vars_idx): # Loop over selected variables (columns)
            ax = axes[ci, j]
            ax.plot(centers[ci, :, v])
            ax.grid(True, alpha=0.3)
            
            # Titles/Labels
            if ci == 0: ax.set_title(f"Var {v}") # Variable name at the top
            if j == 0: ax.set_ylabel(f"Cluster {ci}") # Cluster index on the left
            
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def embed_tsne_from_dist(
    D: np.ndarray, 
    perplexity: int = 30, 
    random_state: int = 42
) -> np.ndarray:
    """
    Apply t-SNE (t-distributed Stochastic Neighbor Embedding) based on a precomputed 
    distance matrix (D) to generate a 2D embedding.

    Parameters
    ----------
    D : np.ndarray
        Precomputed distance matrix of shape (n_samples, n_samples).
    perplexity : int, optional
        The parameter related to the effective number of nearest neighbors. Defaults to 30.
    random_state : int, optional
        Seed for reproducibility. Defaults to 42.

    Returns
    -------
    np.ndarray
        The 2D embedding of the data, shape (n_samples, 2).
    """
    # Initialize and fit t-SNE using the precomputed distance matrix
    return TSNE(
        n_components=2, 
        metric="precomputed",
        perplexity=perplexity, 
        init="random", # Use random initialization for t-SNE on precomputed distances
        random_state=random_state
    ).fit_transform(D)


def embed_mds_from_dist(
    D: np.ndarray, 
    random_state: int = 42
) -> np.ndarray:
    """
    Apply Multidimensional Scaling (MDS) based on a precomputed distance matrix (D) 
    to generate a 2D embedding.

    Parameters
    ----------
    D : np.ndarray
        Precomputed distance matrix of shape (n_samples, n_samples).
    random_state : int, optional
        Seed for reproducibility. Defaults to 42.

    Returns
    -------
    np.ndarray
        The 2D embedding of the data, shape (n_samples, 2).
    """
    # Initialize and fit MDS using the precomputed distance matrix (dissimilarity)
    return MDS(
        n_components=2, 
        dissimilarity="precomputed",
        random_state=random_state
    ).fit_transform(D)


def scatter_2d(
    X2: np.ndarray, 
    labels: np.ndarray, 
    title: str = "2D embedding", 
    y_true: Optional[np.ndarray] = None, 
    show: bool = True
) -> plt.Figure:
    """
    Create a scatter plot of a 2D embedding, colored by cluster labels. 
    Optionally, true class boundaries can be highlighted.

    Parameters
    ----------
    X2 : np.ndarray
        The 2D embedding data, shape (n_samples, 2).
    labels : np.ndarray
        Cluster labels used for coloring the points, shape (n_samples,).
    title : str, optional
        Title of the plot. Defaults to "2D embedding".
    y_true : Optional[np.ndarray], optional
        Optional true class labels used to overlay black circles indicating 
        true class boundaries. Defaults to None.
    show : bool, optional
        If True, display the plot immediately. Defaults to True.

    Returns
    -------
    plt.Figure
        The Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    # Scatter plot, colored by predicted labels
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=labels, s=20, alpha=0.85, cmap="tab10")
    
    ax.set_title(title); ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label("Cluster")
    
    # Optional: Overlay black circles around points belonging to the same true label
    if y_true is not None:
        for lab in np.unique(y_true):
            idx = (y_true == lab)
            # Plot transparent circles with black edges
            ax.scatter(X2[idx, 0], X2[idx, 1], facecolors="none", edgecolors="k", s=40, linewidths=0.8)
            
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_confusion_matrix(
    C: np.ndarray, 
    labels_true: List[Any], 
    labels_pred: List[Any], 
    label_to_activity: Optional[Dict] = None, 
    normalize: bool = True, 
    title: str = "Confusion Matrix (Aligned)", 
    show: bool = True
) -> plt.Figure:
    """
    Display a confusion matrix C as a heatmap, typically after aligning cluster 
    IDs to true labels using the Hungarian algorithm.

    Parameters
    ----------
    C : np.ndarray
        The confusion matrix of shape (n_true_labels, n_clusters).
    labels_true : List[Any]
        List of true labels corresponding to the rows of C.
    labels_pred : List[Any]
        List of predicted cluster IDs corresponding to the columns of C.
    label_to_activity : Optional[Dict], optional
        Mapping to convert numeric true labels to activity names for the Y-axis. 
        Defaults to None.
    normalize : bool, optional
        If True, normalize the matrix rows (percentage of true class that fell into 
        each cluster). Defaults to True.
    title : str, optional
        Title of the plot. Defaults to "Confusion Matrix (Aligned)".
    show : bool, optional
        If True, display the plot immediately. Defaults to True.

    Returns
    -------
    plt.Figure
        The Matplotlib Figure object.
    """
    C_display = C.astype(float)
    if normalize:
        # Normalize by row sum (i.e., by the total count of each true class)
        row_sums = C_display.sum(axis=1, keepdims=True)
        # Avoid division by zero for empty rows
        row_sums[row_sums == 0] = 1e-8
        C_display = C_display / row_sums
        
    # Prepare tick labels for the Y-axis (True Classes)
    if label_to_activity:
        tick_labels = [label_to_activity.get(int(l), str(l)) for l in labels_true]
    else:
        tick_labels = [str(l) for l in labels_true]
        
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Create the heatmap
    sns.heatmap(
        C_display, 
        annot=True, 
        fmt=".2f" if normalize else "d", # Format as float if normalized, integer otherwise
        cmap="Blues",
        xticklabels=labels_pred, 
        yticklabels=tick_labels, 
        ax=ax
    )
    
    ax.set_xlabel("Predicted Clusters"); 
    ax.set_ylabel("True Classes"); 
    ax.set_title(title)
    
    fig.tight_layout()
    if show:
        plt.show()
    return fig