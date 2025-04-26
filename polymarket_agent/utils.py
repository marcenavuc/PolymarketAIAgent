import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score


def bootstrap_roc_auc(y_true, y_pred_proba, n_iterations=1000, confidence_level=0.95, random_state=None):
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    rng = np.random.RandomState(random_state)
    aucs = []
    try:
        orig_auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        orig_auc = np.nan
    for _ in range(n_iterations):
        idx = rng.choice(len(y_true), len(y_true), True)
        if len(np.unique(y_true[idx])) < 2:
            aucs.append(np.nan)
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_pred_proba[idx]))
        except ValueError:
            aucs.append(np.nan)
    aucs = np.array(aucs)
    aucs = aucs[~np.isnan(aucs)]
    if len(aucs) == 0:
        return orig_auc, np.nan, np.nan, np.array([])
    alpha = (1 - confidence_level) / 2
    lower = np.percentile(aucs, alpha * 100)
    upper = np.percentile(aucs, (1 - alpha) * 100)
    return orig_auc, lower, upper, aucs


def roc_auc_single(y_true, y_score, color='darkorange'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color=color, lw=2.5, marker='.', markersize=5, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.500)')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(alpha=0.4)
    plt.show()