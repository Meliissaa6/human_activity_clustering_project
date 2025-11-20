# src/utils_io.py
import os, json
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _to_python_scalars(obj: Any) -> Any:
    """
    Recursively cast numpy scalars within a nested structure (dict, list, tuple) 
    to native Python types (int, float) so json.dump() can serialize them without errors.

    Args
    ----
    obj : Any
        The input object (dictionary, list, tuple, or scalar).

    Returns
    -------
    Any
        The object with NumPy scalars converted to native Python types.
    """
    if isinstance(obj, dict):
        # Recursively process dictionary keys and values
        return { _to_python_scalars(k): _to_python_scalars(v) for k, v in obj.items() }
    if isinstance(obj, (list, tuple)):
        # Recursively process list/tuple elements
        return [ _to_python_scalars(x) for x in obj ]
    if isinstance(obj, (np.floating, np.integer)):
        # Convert NumPy scalar to native Python type (e.g., np.int64 to int)
        return obj.item()
    return obj


def _ensure_dir(p: str) -> str:
    """
    Ensure the specified directory path exists. Creates it if necessary.

    Args
    ----
    p : str
        The path to the directory.

    Returns
    -------
    str
        The original path 'p'.
    """
    os.makedirs(p, exist_ok=True); return p


def save_clustering_results(
    method_name: str,
    y_pred: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    scores: Optional[Dict] = None,
    centers: Optional[np.ndarray] = None,
    mapping: Optional[Dict] = None,
    embedding_2d: Optional[np.ndarray] = None,
    dist_matrix: Optional[np.ndarray] = None,
    confusion: Optional[np.ndarray] = None,
    labels_true: Optional[List[Any]] = None, 
    labels_pred: Optional[List[Any]] = None,
    params: Optional[Dict] = None,
    output_dir: str = "results",
    run_id: Optional[str] = None,
) -> Tuple[str, str, str, str]:
    """
    Save the results of a clustering run, including predictions, metrics, 
    parameters, and auxiliary arrays, into a structured directory hierarchy.

    The structure is: output_dir / method_name / run_id / {arrays, figures, *.json, groups.csv}.

    Args
    ----
    method_name : str
        Name of the clustering method (e.g., "DBA-KMeans", "K-Shape").
    y_pred : np.ndarray
        Predicted cluster labels, shape (n_samples,).
    y_true : Optional[np.ndarray], optional
        True class labels, shape (n_samples,). Defaults to None.
    scores : Optional[Dict], optional
        External/internal clustering metrics. Defaults to None.
    centers : Optional[np.ndarray], optional
        Cluster centroids. Defaults to None.
    mapping : Optional[Dict], optional
        Mapping from cluster ID to true label (e.g., from Hungarian matching). Defaults to None.
    embedding_2d : Optional[np.ndarray], optional
        2D embedding for visualization (e.g., UMAP, PCA). Defaults to None.
    dist_matrix : Optional[np.ndarray], optional
        Pairwise distance matrix used for clustering. Defaults to None.
    confusion : Optional[np.ndarray], optional
        Confusion matrix (rows=true labels, cols=clusters). Defaults to None.
    labels_true : Optional[List[Any]], optional
        True labels corresponding to confusion matrix rows. Defaults to None.
    labels_pred : Optional[List[Any]], optional
        Predicted cluster IDs corresponding to confusion matrix columns. Defaults to None.
    params : Optional[Dict], optional
        Parameters used for the clustering run. Defaults to None.
    output_dir : str, optional
        Root directory for saving results. Defaults to "results".
    run_id : Optional[str], optional
        Unique identifier for the run. If None, it is generated based on timestamp and random_state.

    Returns
    -------
    Tuple[str, str, str, str]
        root_dir, arrays_dir, figures_dir, run_id.
    """
    # 1. Create run_id if absent
    if run_id is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        seed = params.get("random_state") if isinstance(params, dict) else None
        run_id = f"{stamp}" + (f"_seed{seed}" if seed is not None else "")

    # 2. Setup directory structure
    root = _ensure_dir(os.path.join(output_dir, method_name, run_id))
    arr_dir = _ensure_dir(os.path.join(root, "arrays"))
    fig_dir = _ensure_dir(os.path.join(root, "figures"))

    # 3. Save JSON files (parameters, metrics, mapping)
    if params is not None:
        with open(os.path.join(root, "params.json"), "w") as f:
            # Convert NumPy types to native Python before dumping
            json.dump(_to_python_scalars(params), f, indent=4)
    if scores is not None:
        with open(os.path.join(root, "metrics.json"), "w") as f:
            json.dump(_to_python_scalars(scores), f, indent=4)
    if mapping is not None:
        with open(os.path.join(root, "mapping.json"), "w") as f:
            # Ensure keys (cluster IDs) and values (labels) are integers for JSON
            json.dump({int(k): int(v) for k,v in mapping.items()}, f, indent=4)

    # 4. Save NumPy arrays (predictions, data, matrices) using compressed format (.npz)
    np.savez_compressed(os.path.join(arr_dir, "y_pred.npz"), y_pred=y_pred)
    if y_true is not None: np.savez_compressed(os.path.join(arr_dir, "y_true.npz"), y_true=y_true)
    if centers is not None: np.savez_compressed(os.path.join(arr_dir, "centers.npz"), centers=centers)
    if embedding_2d is not None: np.savez_compressed(os.path.join(arr_dir, "embedding_2d.npz"), X2=embedding_2d)
    if dist_matrix is not None: np.savez_compressed(os.path.join(arr_dir, "distance_matrix.npz"), D=dist_matrix)
    if confusion is not None:
        np.savez_compressed(os.path.join(arr_dir, "confusion_matrix.npz"), C=confusion)
        if labels_true is not None: np.savez_compressed(os.path.join(arr_dir, "confusion_true_labels.npz"), labels_true=np.asarray(labels_true))
        if labels_pred is not None: np.savez_compressed(os.path.join(arr_dir, "confusion_pred_labels.npz"), labels_pred=np.asarray(labels_pred))

    # 5. Save cluster assignments to CSV (requires pandas)
    try:
        import pandas as pd
        
        # Base DataFrame with indices and predicted clusters
        df = {"index": np.arange(len(y_pred)), "cluster": y_pred}
        if y_true is not None: df["true_label"] = y_true
        
        # Include mapped labels if a mapping is provided
        if mapping is not None:
            mp = {int(k): int(v) for k,v in mapping.items()}
            # Apply the mapping: cluster ID -> mapped label. Default to -1 if ID is not in mapping.
            y_map = np.vectorize(lambda c: mp.get(int(c), -1))(y_pred)
            df["mapped_label"] = y_map
            
        pd.DataFrame(df).to_csv(os.path.join(root, "groups.csv"), index=False)
    except Exception:
        # Ignore if pandas is not available or if mapping/writing fails
        pass

    print(f"✅ Saved run to: {root}")
    return root, arr_dir, fig_dir, run_id