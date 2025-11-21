import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.linear_model import LassoCV
from scipy.stats import zscore
from pyopls import OPLS

from typing import Tuple, Union, Optional

def preprocess_data(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    scale: bool = True,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]:
    """
    Fill NA values with 0, drop constant (zero-variance) columns, and optionally scale features.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series or np.ndarray): Target values.
    - scale (bool): Whether to apply z-score normalization. Defaults to True.
    - verbose (bool): If True, prints details about preprocessing steps. Defaults to False.

    Returns:
    - Tuple[pd.DataFrame, pd.Series or np.ndarray]: Cleaned and optionally scaled X, and unchanged y.
    """
    X = X.fillna(0)
    constant_cols = X.columns[X.std(ddof=1) == 0]
    X = X.loc[:, X.std(ddof=1) != 0]

    if scale:
        X = pd.DataFrame(zscore(X, ddof=1), index=X.index, columns=X.columns)

    if verbose:
        print(f"[Preprocessing] Dropped {len(constant_cols)} constant columns.")
        if scale:
            print("[Preprocessing] Applied z-score normalization.")

    return X, y

def select_lasso_features(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    alphas: Optional[list] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Selects features using LASSO regression with cross-validation.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series or np.ndarray): Target values.
    - alphas (list, optional): List of alpha values to try for LassoCV. Defaults to [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1].
    - verbose (bool): If True, prints number of selected features and optimal alpha.

    Returns:
    - pd.DataFrame: Subset of X with selected features.
    """
    if alphas is None:
        alphas = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1]
    lassocv = LassoCV(alphas=alphas, cv=10)
    lassocv.fit(X, y)
    selected = X.columns[lassocv.coef_ != 0]
    if verbose:
        print(f"[LASSO] Selected {len(selected)} features, alpha = {lassocv.alpha_:.1e}")
    return X[selected]

def fit_pls_model(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    n_components: int = 2,
    orthogonalize: bool = True,
    verbose: bool = False
) -> Tuple[PLSRegression, np.ndarray, Optional[OPLS]]:
    """
    Fit a PLS or OPLS model to the data.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series or np.ndarray): Target values.
    - n_components (int): Number of components to use in the model. Defaults to 2.
    - orthogonalize (bool): If True, applies orthogonalization via OPLS. Defaults to True.
    - verbose (bool): If True, prints model fitting status. Defaults to False.

    Returns:
    - Tuple[PLSRegression, np.ndarray, Optional[OPLS]]: Trained PLS model, transformed X (Z), and optional OPLS object.
    """
    if orthogonalize:
        opls = OPLS(n_components)
        Z = opls.fit_transform(X, y)
        if verbose:
            print("[OPLS] Applied orthogonalization.")
    else:
        Z = X.copy()
        opls = None
        if verbose:
            print("[PLS] Running without orthogonalization.")

    pls = PLSRegression(n_components=n_components)
    pls.fit(Z, y)
    if verbose:
        print(f"[PLS] Fitted model with {n_components} components.")
    return pls, Z, opls

def evaluate_model(
    model: PLSRegression,
    X_transformed: np.ndarray,
    y_true: Union[pd.Series, np.ndarray],
    mode: str = 'regression',
    verbose: bool = False
) -> dict:
    """
    Evaluate a PLS/OPLS model using cross-validation.

    Parameters:
    - model (PLSRegression): Trained PLS model.
    - X_transformed (np.ndarray): Transformed feature matrix.
    - y_true (pd.Series or np.ndarray): Ground truth values.
    - mode (str): 'regression' or 'classification'.
    - verbose (bool): If True, prints evaluation metrics.

    Returns:
    - dict: Dictionary with predictions and evaluation metrics.
    """
    y_pred = cross_val_predict(model, X_transformed, y_true, cv=5)
    if mode == 'regression':
        q2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        if verbose:
            print(f"[Evaluation] Q²: {q2:.3f}, MSE: {mse:.3f}")
        return {'y_pred': y_pred, 'Q2': q2, 'MSE': mse}
    elif mode == 'classification':
        acc = accuracy_score(y_true, np.sign(y_pred))
        if verbose:
            print(f"[Evaluation] Accuracy: {acc:.3f}")
        return {'y_pred': y_pred, 'Accuracy': acc}
    else:
        raise ValueError("Mode must be 'regression' or 'classification'")

def permutation_test(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    mode: str,
    n_components: int,
    n_permutations: int,
    orthogonalize: bool,
    true_score: float,
    score_metric: str = 'Q2',  # 'Q2' or 'MSE' for regression, 'accuracy' for classification
    random_state: Optional[int] = 42,
    verbose: bool = False,
    return_scores: bool = False
) -> Union[float, Tuple[float, list]]:
    """
    Perform permutation testing to assess statistical significance of the model.

    ...

    - score_metric (str): Which metric to use for scoring ('Q2', 'MSE', or 'accuracy').
    """
    np.random.seed(random_state)
    scores = []
    for i in range(n_permutations):
        y_perm = np.random.permutation(y)
        pls, Z, _ = fit_pls_model(X, y_perm, n_components, orthogonalize)
        y_pred = cross_val_predict(pls, Z, y_perm, cv=5)

        if mode == 'regression':
            if score_metric.lower() == 'q2':
                score = r2_score(y_perm, y_pred)
            elif score_metric.lower() == 'mse':
                score = mean_squared_error(y_perm, y_pred)
            else:
                raise ValueError("Invalid score_metric for regression: choose 'Q2' or 'MSE'")
        elif mode == 'classification':
            score = accuracy_score(y_perm, np.sign(y_pred))
        else:
            raise ValueError("Mode must be 'regression' or 'classification'")

        scores.append(score)
        if verbose and (i + 1) % 10 == 0:
            print(f"[Permutation] Completed {i + 1}/{n_permutations} permutations")

    scores = np.sort(scores)
    if score_metric.lower() == 'mse':
        p = (np.sum(scores <= true_score) + 1) / (n_permutations + 1)  # lower is better
    else:
        p = (np.sum(scores >= true_score) + 1) / (n_permutations + 1)  # higher is better

    if verbose:
        print(f"[Permutation] Empirical p-value: {p:.4f}")
    
    return (p, scores) if return_scores else p

def calculate_vip(model: PLSRegression, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """
    Calculate VIP scores from a fitted PLS model.

    Parameters:
    - model (PLSRegression): Fitted PLS model.
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series or np.ndarray): Target values.

    Returns:
    - np.ndarray: VIP scores for each feature.
    """
    T = model.x_scores_
    W = model.x_weights_
    Q = model.y_loadings_

    p, h = W.shape
    vip = np.zeros((p,))
    s = np.diag(T.T @ T @ Q.T @ Q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([(W[i, j] ** 2) * s[j] for j in range(h)])
        vip[i] = np.sqrt(p * np.sum(weight) / total_s)

    return vip
