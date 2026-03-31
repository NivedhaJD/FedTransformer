# federated_learning/explainability/xai.py
# SHAP-style feature attribution for 6G-ISAC Transformer predictions.

import numpy as np
import torch
from typing import Dict, List

from network.dataset import FEATURE_NAMES


class LocalExplainer:
    """
    Produces SHAP-style feature importance attributions for the
    ISACTransformer model predictions using a kernel SHAP approximation.
    """

    def __init__(self, model, background_data: np.ndarray):
        self.model = model
        # Use a small background reference set
        self.background = background_data[:20]
        self.feature_names = FEATURE_NAMES

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Run inference and return class probabilities."""
        self.model.eval()
        with torch.no_grad():
            inp = torch.tensor(X, dtype=torch.float32)
            log_probs = self.model(inp)
            return torch.exp(log_probs).numpy()

    def _kernel_shap(self, X: np.ndarray, n_samples: int = 50) -> np.ndarray:
        """
        Lightweight Kernel SHAP approximation.
        Returns shape (len(X), seq_len, n_features) averaged to (len(X), n_features).
        """
        # Average across sequence dimension for attribution
        X_avg = X.mean(axis=1) if X.ndim == 3 else X  # (N, features)
        bg_avg = self.background.mean(axis=1) if self.background.ndim == 3 else self.background

        n_features = X_avg.shape[1]
        shap_values = np.zeros((len(X_avg), n_features))
        baseline = self._predict_proba(self.background).mean(axis=0)

        rng = np.random.default_rng(42)
        for i, x in enumerate(X_avg):
            for _ in range(n_samples):
                mask = rng.integers(0, 2, size=n_features).astype(bool)
                # Create masked samples by blending with background
                masked_batch = bg_avg.copy()
                masked_batch[:, mask] = x[mask]
                # Reconstruct sequences (repeat the feature vector)
                if self.background.ndim == 3:
                    seq_len = self.background.shape[1]
                    masked_seq = np.repeat(masked_batch[:, np.newaxis, :], seq_len, axis=1)
                else:
                    masked_seq = masked_batch
                pred_with = self._predict_proba(masked_seq).mean(axis=0)
                # Contribution to the dominant (non-normal) class
                dom_class = np.argmax(pred_with[1:]) + 1
                shap_values[i] += (pred_with[dom_class] - baseline[dom_class]) * mask
            shap_values[i] /= n_samples

        return shap_values

    def explain(self, X: np.ndarray, n_samples: int = 5) -> Dict:
        """
        Generate SHAP-style explanations for a batch of 6G-ISAC samples.

        Returns:
          {
            "shap_values":    np.ndarray (n_samples × n_features),
            "mean_abs_shap":  dict {feature: mean |SHAP|},
            "top_features":   list of (feature, importance) sorted desc,
            "predictions":    np.ndarray of class predictions,
            "reason":         human-readable explanation string,
          }
        """
        # Use a small subset for speed
        X_sub = X[:n_samples]
        shap_vals = self._kernel_shap(X_sub, n_samples=30)
        proba = self._predict_proba(X_sub)
        preds = proba.argmax(axis=1)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        importance = dict(zip(self.feature_names, mean_abs.tolist()))
        top = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # Build human-readable reason
        top_feature, top_val = top[0]
        if np.unique(preds).size > 1 or preds[0] != 0:
            reason = (
                f"Network anomaly detected. Primary driver: '{top_feature}' "
                f"(importance={top_val:.3f}). "
                f"Secondary: '{top[1][0]}' ({top[1][1]:.3f})."
            )
        else:
            reason = "Network classified as normal across sampled window."

        print(f"[XAI] Top feature: {top[0][0]} ({top[0][1]:.3f})")
        return {
            "shap_values":   shap_vals,
            "mean_abs_shap": importance,
            "top_features":  top[:5],
            "predictions":   preds,
            "reason":        reason,
        }

    def validate_explanation(self, explanation: Dict) -> bool:
        """
        Peer validation check — explanation must be non-trivial
        (at least one feature must have |SHAP| > 0.001).
        """
        max_importance = max(explanation["mean_abs_shap"].values())
        valid = max_importance > 0.001
        if not valid:
            print("[XAI] Explanation rejected — no feature exceeds threshold.")
        return valid
