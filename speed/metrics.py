"""
Evaluation metrics for downstream EEG classification and regression tasks.

Provides standard metrics used in EEG benchmarking literature:
balanced accuracy, AUROC, F1-score, and Cohen's Kappa.
"""

from typing import List, Optional, Union

import numpy as np


def balanced_accuracy(y_true, y_pred) -> float:
    """
    Compute balanced accuracy (average per-class recall).

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Balanced accuracy score.
    """
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))


def auroc(
    y_true,
    y_score,
    multi_class: str = "ovr",
    average: str = "macro"
) -> float:
    """
    Compute area under the ROC curve.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_score : array-like
        Predicted probabilities or decision scores.
        Shape (n_samples, n_classes) for multiclass.
    multi_class : str
        Multiclass strategy: "ovr" (one-vs-rest) or "ovo" (one-vs-one).
    average : str
        Averaging strategy: "macro", "weighted".

    Returns
    -------
    float
        AUROC score.
    """
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(
        y_true, y_score, multi_class=multi_class, average=average
    ))


def f1_score(y_true, y_pred, average: str = "macro") -> float:
    """
    Compute F1 score.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    average : str
        Averaging strategy: "macro", "micro", "weighted", or "binary".

    Returns
    -------
    float
        F1 score.
    """
    from sklearn.metrics import f1_score as sklearn_f1
    return float(sklearn_f1(y_true, y_pred, average=average))


def cohens_kappa(y_true, y_pred) -> float:
    """
    Compute Cohen's Kappa coefficient.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Cohen's Kappa score.
    """
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_true, y_pred))


def classification_report(
    y_true,
    y_pred,
    label_names: Optional[List[str]] = None,
    y_score=None,
) -> dict:
    """
    Compute a summary of classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    label_names : list of str, optional
        Human-readable names for each class.
    y_score : array-like, optional
        Predicted probabilities for AUROC computation.

    Returns
    -------
    dict
        Dictionary with keys: accuracy, balanced_accuracy, f1_macro,
        f1_weighted, cohens_kappa, and optionally auroc.
    """
    from sklearn.metrics import accuracy_score

    report = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "cohens_kappa": cohens_kappa(y_true, y_pred),
    }

    if y_score is not None:
        try:
            report["auroc"] = auroc(y_true, y_score)
        except ValueError:
            pass  # AUROC undefined for some edge cases (single class)

    return report
