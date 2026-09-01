"""Evaluation. One function, one set of numbers, canonical class order everywhere.

Every array, matrix and report is indexed by the 10 canonical classes in the
order given by configs/data.yaml. The confusion matrix is always 10x10; there is
no union-of-label-sets step, which is what silently inflated the legacy matrices
to 14x14 and deflated every metric with them (MIGRATION_NOTES.md section 3).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import load_data_config


def predict_logits(module, cache, idxs, labels_by_idx, batch_size: int = 64, device=None):
    """Forward pass over a list of idx values. Returns (idx, y_true, logits)."""
    import torch
    from torch.utils.data import DataLoader

    from .train import FoldDataset

    device = device or next(module.parameters()).device
    dataset = FoldDataset(cache, idxs, labels_by_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    module.eval()
    all_logits, all_true, all_idx = [], [], []
    with torch.no_grad():
        for images, targets, image_idx in loader:
            logits = module(images.to(device))
            all_logits.append(logits.float().cpu().numpy())
            all_true.append(np.asarray(targets))
            all_idx.append(np.asarray(image_idx))
    return (
        np.concatenate(all_idx),
        np.concatenate(all_true),
        np.concatenate(all_logits, axis=0),
    )


def metrics_from_predictions(
    y_true, y_pred, data_cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Every reported metric, in canonical class order."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    data_cfg = data_cfg or load_data_config()
    classes = list(data_cfg["classes"])
    labels = list(range(len(classes)))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # the cross-check the legacy code added after the fact -- kept, unconditionally
    diagonal_accuracy = float(matrix.diagonal().sum()) / float(matrix.sum())
    if abs(diagonal_accuracy - accuracy) > 1e-9:
        raise ValueError(
            "accuracy %.8f disagrees with confusion-matrix diagonal %.8f -- label order is wrong"
            % (accuracy, diagonal_accuracy)
        )

    return {
        "accuracy": accuracy,
        "f1_macro": float(f_macro),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_per_class": {c: float(v) for c, v in zip(classes, f1)},
        "recall_per_class": {c: float(v) for c, v in zip(classes, recall)},
        "precision_per_class": {c: float(v) for c, v in zip(classes, precision)},
        "support_per_class": {c: int(v) for c, v in zip(classes, support)},
        "confusion_matrix": matrix.tolist(),
        "class_order": classes,
        "n_images": int(len(y_true)),
    }


def evaluate_fold(
    module, cache, idxs, labels_by_idx, data_cfg: dict[str, Any] | None = None, device=None
) -> dict[str, Any]:
    """Predict and score one partition."""
    image_idx, y_true, logits = predict_logits(
        module, cache, idxs, labels_by_idx, device=device
    )
    y_pred = logits.argmax(axis=1)
    result = metrics_from_predictions(y_true, y_pred, data_cfg)
    result["idx"] = [int(i) for i in image_idx]
    result["y_true"] = [int(v) for v in y_true]
    result["y_pred"] = [int(v) for v in y_pred]
    return result
