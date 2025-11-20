import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def compute_group_pivots(
    X: np.ndarray, 
    y: np.ndarray, 
    label_to_activity: Dict[int, str], 
    group_to_indices: Dict[str, List[int]]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute per-activity statistics (mean, standard deviation, energy) aggregated 
    by predefined sensor groups and return the results as pivot tables.

    The statistics are calculated across all time steps and all samples belonging 
    to a specific activity, for each variable, and then averaged over the variables 
    within a sensor group.

    Args
    ----
    X : np.ndarray
        Time-series tensor of shape (n_samples, n_variables, n_timesteps).
    y : np.ndarray
        Label vector of shape (n_samples,). Labels must be integers mapping to activities.
    label_to_activity : Dict[int, str]
        Mapping from numeric labels (e.g., 1..6) to human-readable activity names.
    group_to_indices : Dict[str, List[int]]
        Dictionary mapping a group name (e.g., "accm", "acce", "vit") to the 
        list of variable indices belonging to that group in the 'X' array.

    Returns
    -------
    mean_pivot : pd.DataFrame
        Pivot table of mean values per activity (index) and sensor group (columns).
    std_pivot : pd.DataFrame
        Pivot table of standard deviations per activity (index) and sensor group (columns).
    energy_pivot : pd.DataFrame
        Pivot table of signal energy (mean of squared signal) per activity (index) and 
        sensor group (columns).
    """
    rows = []
    # Iterate over unique activity labels
    for lab in np.sort(np.unique(y)):
        # Filter data for the current activity label
        X_lab = X[y == lab]
        
        # Calculate per-variable statistics aggregated over windows and time:
        # Mean across samples (axis=0) and time steps (axis=2). Result shape: (n_variables,)
        means = X_lab.mean(axis=(0, 2))
        stds = X_lab.std(axis=(0, 2))
        # Energy: mean of squared signal across samples and time steps
        energy = (X_lab ** 2).mean(axis=(0, 2))
        
        # Aggregate the calculated statistics by sensor group
        for g, idxs in group_to_indices.items():
            # Average the statistics across the variable indices within the current group 'g'
            rows.append({
                "Label": int(lab),
                "Activity": label_to_activity.get(int(lab), "Unknown"),
                "Group": g,
                "Mean": float(means[idxs].mean()),
                "Std": float(stds[idxs].mean()),
                "Energy": float(energy[idxs].mean()),
            })
            
    # Create a DataFrame from the results
    df = pd.DataFrame(rows).sort_values(["Label", "Group"], kind="stable")

    # Create pivot tables for each statistic
    mean_pivot = df.pivot(index="Activity", columns="Group", values="Mean")
    std_pivot = df.pivot(index="Activity", columns="Group", values="Std")
    energy_pivot = df.pivot(index="Activity", columns="Group", values="Energy")

    return mean_pivot, std_pivot, energy_pivot