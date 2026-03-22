"""
evaluation.py
-------------
Performance evaluation for the federated ISAC Transformer.

Computes:
  - Accuracy
  - Precision (macro)
  - Recall (macro)
  - F1-Score (macro)
  - Cross-Entropy Loss  L = -(1/m) Σ [y log(ŷ) + (1-y) log(1-ŷ)]

All functions accept either raw PyTorch tensors or NumPy arrays.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from dataset import NUM_CLASSES, CLASS_NAMES


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Run inference on the full dataloader and return a metrics dictionary.

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1_score, loss,
        confusion_matrix, classification_report
    """
    model.eval()
    criterion = nn.NLLLoss()   # expects log-probabilities from the model

    all_preds  = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            log_probs = model(X_batch)                   # (batch, num_classes)
            loss = criterion(log_probs, y_batch)

            total_loss    += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

            preds = log_probs.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics = {
        "accuracy"              : float(accuracy_score(y_true, y_pred)),
        "precision"             : float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall"                : float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_score"              : float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "loss"                  : float(avg_loss),
        "confusion_matrix"      : confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist(),
        "classification_report" : classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
        ),
    }
    return metrics


# ---------------------------------------------------------------------------
# Lightweight in-loop evaluation (used during local training)
# ---------------------------------------------------------------------------
def quick_accuracy(log_probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Fast batch accuracy — does NOT require a full pass over the dataset."""
    preds = log_probs.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Cross-Entropy Loss (manual implementation for transparency)
# ---------------------------------------------------------------------------
def cross_entropy_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    """
    Binary / multi-label cross-entropy (for documentation purposes).

    L = -(1/m) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]

    In practice the model uses nn.NLLLoss with log_softmax outputs.
    """
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------
def print_metrics(metrics: dict, round_num: int = None):
    prefix = f"[Round {round_num}]" if round_num is not None else "[Eval]"
    print(
        f"{prefix}  Loss: {metrics['loss']:.4f}  "
        f"Acc: {metrics['accuracy']:.4f}  "
        f"P: {metrics['precision']:.4f}  "
        f"R: {metrics['recall']:.4f}  "
        f"F1: {metrics['f1_score']:.4f}"
    )


if __name__ == "__main__":
    # Smoke-test with a random model
    from model import ISACTransformer
    from dataset import get_test_dataloader

    model = ISACTransformer()
    dl = get_test_dataloader(num_samples=200)
    m = evaluate_model(model, dl)
    print_metrics(m)
    print("\nClassification Report:\n", m["classification_report"])
