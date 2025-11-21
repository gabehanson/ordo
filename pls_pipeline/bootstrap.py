from .core import fit_pls_model, evaluate_model, calculate_vip


def bootstrap_pls_analysis(X, y, samples_per_group=None, desired_coverage=10, 
                           n_components=2, orthogonalize=True, 
                           random_state=42, verbose=True):
    """
    Perform bootstrap-based PLS/OPLS modeling with balanced sampling.

    Returns:
    - dict with keys: 'cv_accuracy', 'vip_matrix', 'model', 'y_labels'
    """
    np.random.seed(random_state)

    # Count group sizes
    group_labels, group_counts = np.unique(y, return_counts=True)
    group_sizes = dict(zip(group_labels, group_counts))

    # Default: 50% of smaller group
    if samples_per_group is None:
        samples_per_group = int(0.5 * min(group_sizes.values()))
        if verbose:
            print(f"[Bootstrap] Defaulting to {samples_per_group} samples per group (50% of smaller group).")

    # Determine number of iterations
    n_iterations = estimate_bootstrap_iterations(group_sizes, samples_per_group, desired_coverage)
    if verbose:
        print(f"[Bootstrap] Running {n_iterations} iterations with {samples_per_group} samples per group.")

    # Storage
    vip_list = []
    acc_list = []
    final_model = None
    final_y = None

    for i in range(n_iterations):
        # Sample balanced subset
        idxs = []
        for label in group_labels:
            label_idxs = np.where(y == label)[0]
            sampled = np.random.choice(label_idxs, size=samples_per_group, replace=False)
            idxs.append(sampled)
        idx_total = np.concatenate(idxs)
        X_boot = X.iloc[idx_total]
        y_boot = y.iloc[idx_total]

        # Fit model
        pls, Z, _ = fit_pls_model(X_boot, y_boot, n_components=n_components, orthogonalize=orthogonalize)

        # Evaluate model
        metrics = evaluate_model(pls, Z, y_boot, mode='classification')
        acc_list.append(metrics['Accuracy'])

        # VIP scores
        vip = calculate_vip(pls, X_boot, y_boot)
        vip_list.append(vip)

        # Save final model and labels
        final_model = pls
        final_y = y_boot

        if verbose and (i+1) % 5 == 0:
            print(f"[Bootstrap] Completed {i+1}/{n_iterations}")

    vip_matrix = pd.DataFrame(vip_list, columns=X.columns)

    return {
        'cv_accuracy': acc_list,
        'vip_matrix': vip_matrix,
        'model': final_model,
        'y_labels': final_y
    }

def estimate_bootstrap_iterations(group_sizes: dict, 
                                   samples_per_group: int, 
                                   desired_coverage: int = 10) -> int:
    """
    Estimate the number of bootstrap iterations needed to cover each sample approximately `desired_coverage` times.

    Parameters:
    - group_sizes (dict): Dictionary with class labels as keys and group sizes as values.
    - samples_per_group (int): Number of samples drawn per group in each iteration.
    - desired_coverage (int): Desired average number of times each sample appears.

    Returns:
    - int: Number of iterations needed.
    """
    iterations = [
        int(np.ceil(desired_coverage * size / samples_per_group))
        for size in group_sizes.values()
    ]
    return max(iterations)
