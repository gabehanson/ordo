import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

def plot_scores_scatter(
    x_scores,
    y_labels,
    components=(0, 1),
    class_labels=None,
    colors=None,
    alpha=0.1,
    title="X Scores",
    save_path=None,
    cmap='viridis',
    max_points=100000,
    random_state=42,
    xlabel=None,
    ylabel=None
):
    """
    Scatter plot of PLS/OPLS X scores, with optional downsampling.
    Supports custom x/y axis labels (e.g., including mixOmics variance explained).
    """

    comp_x, comp_y = components
    x_scores = np.asarray(x_scores)
    y_labels = np.asarray(y_labels)

    # Downsample if needed
    if len(y_labels) > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(y_labels), size=max_points, replace=False)
        x_scores = x_scores[idx]
        y_labels = y_labels[idx]

    df = pd.DataFrame(x_scores[:, [comp_x, comp_y]], columns=['lv1', 'lv2'])
    df['label'] = y_labels

    plt.figure(figsize=(8, 6))

    # Classification mode if <=10 classes
    if pd.Series(y_labels).nunique() <= 10:
        colors = colors or {-1: 'tab:blue', 1: 'tab:orange', 0: 'tab:green'}
        class_labels = class_labels or {-1: 'Negative', 1: 'Positive', 0: 'Neutral'}
        unique_labels = sorted(df['label'].unique())

        for label in unique_labels:
            subset = df[df['label'] == label]
            plt.scatter(
                subset['lv1'],
                subset['lv2'],
                c=colors.get(label, 'gray'),
                label=class_labels.get(label, str(label)),
                alpha=alpha
            )
        plt.legend(loc='best')

    else:
        # Regression mode
        sc = plt.scatter(df['lv1'], df['lv2'], c=df['label'], cmap=cmap, alpha=alpha)
        plt.colorbar(sc, label='Target')

    # Title and axis labels
    plt.title(title)
    plt.xlabel(xlabel if xlabel else f"LV{comp_x+1}")
    plt.ylabel(ylabel if ylabel else f"LV{comp_y+1}")

    # Saving
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def plot_vip_barplot(vips, loadings, feature_names, title="VIP Scores", threshold=1.0, save_path=None, colors=None):
    """
    Barplot of VIP scores with color indicating direction of loading.

    Parameters:
    - vips (np.ndarray): VIP scores from model.
    - loadings (np.ndarray): Loadings for LV1.
    - feature_names (list): Names of the features.
    - title (str): Title of the plot.
    - threshold (float): Threshold lines for highlighting VIP scores.
    - save_path (str): If provided, saves the figure to this path.
    - colors (dict): Optional color map. Accepts either:
        - {'positive': 'color', 'negative': 'color'}
        - {-1: 'color', 1: 'color'}, which will be mapped to 'negative' and 'positive'
    """
    # Standardize color input
    if colors and (-1 in colors and 1 in colors):
        colors = {'positive': colors[1], 'negative': colors[-1]}
    else:
        colors = colors or {'positive': 'tab:orange', 'negative': 'tab:blue'}

    df = pd.DataFrame({'Feature': feature_names, 'VIP': vips, 'Loading': loadings})
    df = df.sort_values(by='VIP', ascending=False)
    df['Direction'] = df['Loading'].apply(lambda x: 'positive' if x > 0 else 'negative')
    df['Signed_VIP'] = df.apply(lambda row: row['VIP'] if row['Loading'] > 0 else -row['VIP'], axis=1)

    plt.figure(figsize=(4, len(feature_names) * 0.25 + 4))
    sns.barplot(data=df, x='Signed_VIP', y='Feature', hue='Direction', dodge=False, palette=colors)

    plt.axvline(x=threshold, color='red', linestyle='--')
    plt.axvline(x=-threshold, color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('VIP Score (signed by loading)')
    plt.ylabel('')
    plt.legend().set_visible(False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()



def plot_roc_curve(y_true, y_pred, label='Model', save_path=None):
    """
    Plot ROC curve from true and predicted labels.

    Parameters:
    - y_true (array-like): Ground truth binary labels.
    - y_pred (array-like): Predicted continuous scores.
    - label (str): Label for the ROC line.
    - save_path (str): If provided, saves the figure to this path.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)

    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC={auc_score:.4f})', color='red')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_permutation_histogram(scores, true_score, bins=50, title="Permutation Accuracy Distribution", save_path=None):
    """
    Plot histogram of permutation test scores with a vertical line for the model accuracy.

    Parameters:
    - scores (array-like): Distribution of permuted scores.
    - true_score (float): Accuracy or Q² of the original model.
    - bins (int): Number of bins in the histogram.
    - title (str): Plot title.
    - save_path (str): If provided, saves the figure to this path.
    """
    plt.hist(scores, bins=bins, edgecolor='black')
    plt.axvline(x=true_score, color='red', linestyle='--', linewidth=2, label='Model Score')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_model_scores(
    model,
    y_labels,
    components=(0, 1),
    class_labels=None,
    colors=None,
    alpha=0.1,
    title="X Scores",
    save_path=None
):
    """
    Wrapper to plot X scores directly from a fitted PLS model.

    Parameters:
    - model: Fitted PLSRegression model with x_scores_ attribute.
    - y_labels (array-like): Target values or class labels.
    - components (tuple): Latent variables to plot. Defaults to (0, 1).
    - class_labels (dict): Custom class labels for legend.
    - colors (dict): Custom colors for classes.
    - alpha (float): Transparency of scatter points.
    - title (str): Title of the plot.
    - save_path (str): Optional path to save the plot.
    """
    x_scores = model.x_scores_
    plot_scores_scatter(
        x_scores=x_scores,
        y_labels=y_labels,
        components=components,
        class_labels=class_labels,
        colors=colors,
        alpha=alpha,
        title=title,
        save_path=save_path
    )

def plot_vip_barplot_from_model(model, X, title="VIP Scores", threshold=1.0, save_path=None, colors=None):
    """
    Wrapper to compute and plot VIP scores for a fitted PLS model.

    Parameters:
    - model (PLSRegression): Trained PLS model.
    - X (pd.DataFrame): Original feature matrix used in the model.
    - title (str): Title of the plot.
    - threshold (float): VIP threshold to highlight.
    - save_path (str): Optional path to save the figure.
    - colors (dict): Optional color map {'positive': 'color', 'negative': 'color'}.
    """
    T = model.x_scores_
    W = model.x_weights_
    Q = model.y_loadings_

    # Calculate VIP scores
    p, h = W.shape
    vip = np.zeros((p,))
    s = np.diag(T.T @ T @ Q.T @ Q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(W[i, j] ** 2) * s[j] for j in range(h)])
        vip[i] = np.sqrt(p * np.sum(weight) / total_s)

    # Use loadings from first component to assign sign
    loadings = model.x_loadings_[:, 0]
    feature_names = X.columns

    plot_vip_barplot(vips=vip, loadings=loadings, feature_names=feature_names, title=title, threshold=threshold, save_path=save_path, colors=colors)

    
def plot_roc_curve_from_model(model, X, y, label='Model', save_path=None):
    """
    Compute predictions and plot ROC curve from a fitted PLS/OPLS model.

    Parameters:
    - model: Fitted PLSRegression model.
    - X (pd.DataFrame or np.ndarray): Input features.
    - y (array-like): Ground truth binary labels.
    - label (str): Label for the ROC line.
    - save_path (str): If provided, saves the figure to this path.
    """
    y_pred = model.predict(X).ravel()
    plot_roc_curve(y_true=y, y_pred=y_pred, label=label, save_path=save_path)
    
    
def plot_vip_barplot_with_error(vip_matrix: pd.DataFrame, 
                                 loadings: np.ndarray,
                                 threshold: float = 1.0, 
                                 title: str = "VIP Scores", 
                                 save_path: str = None, 
                                 colors: dict = None):
    """
    Plot mean VIP scores with standard deviation error bars and loading-signed coloring.

    Parameters:
    - vip_matrix (pd.DataFrame): Rows = bootstrap iterations, columns = feature names.
    - loadings (np.ndarray): x_loadings_ from fitted model (to sign VIPs).
    - threshold (float): Red line threshold to highlight important VIPs.
    - title (str): Plot title.
    - save_path (str): Path to save figure if desired.
    - colors (dict): {'positive': 'color', 'negative': 'color'} or {-1: 'color', 1: 'color'}
    """
    vip_mean = vip_matrix.mean()
    vip_std = vip_matrix.std()
    features = vip_matrix.columns
    load_sign = pd.Series(loadings[:len(features)], index=features)
    
    # Match and sort
    df = pd.DataFrame({
        'Feature': features,
        'VIP': vip_mean,
        'SD': vip_std,
        'Loading': load_sign
    })
    df['Direction'] = df['Loading'].apply(lambda x: 'positive' if x > 0 else 'negative')
    df['Signed_VIP'] = df.apply(lambda row: row['VIP'] if row['Loading'] > 0 else -row['VIP'], axis=1)
    df = df.sort_values('VIP', ascending=False)

    # Colors
    if colors and (-1 in colors and 1 in colors):
        colors = {'positive': colors[1], 'negative': colors[-1]}
    else:
        colors = colors or {'positive': 'tab:orange', 'negative': 'tab:blue'}
    
    palette = df['Direction'].map(colors)

    plt.figure(figsize=(6, 0.25 * len(df) + 4))
    plt.barh(df['Feature'], df['Signed_VIP'], xerr=df['SD'], color=palette, edgecolor='black', capsize=4)
    plt.axvline(x=threshold, color='red', linestyle='--')
    plt.axvline(x=-threshold, color='red', linestyle='--')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("VIP Score (Mean ± SD, signed by loading)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
