import numpy as np
import pandas as pd

def subsample_scores_and_labels(x_scores, y_labels, n=10_000, random_state=42):
    """
    Subsample scores and corresponding labels for visualization.

    Parameters:
    - x_scores (np.ndarray): Transformed scores from PLS/OPLS model (e.g., model.x_scores_).
    - y_labels (array-like): Corresponding class labels or targets.
    - n (int): Number of points to sample. Default is 10,000.
    - random_state (int): Seed for reproducibility.

    Returns:
    - x_scores_sub (np.ndarray): Subsampled scores.
    - y_labels_sub (same type as input): Subsampled labels.
    """
    total = len(y_labels)
    if n >= total:
        return x_scores, y_labels

    rng = np.random.RandomState(random_state)
    idx = rng.choice(total, size=n, replace=False)

    x_sub = x_scores[idx]
    y_sub = y_labels.iloc[idx] if isinstance(y_labels, pd.Series) else np.array(y_labels)[idx]

    return x_sub, y_sub
