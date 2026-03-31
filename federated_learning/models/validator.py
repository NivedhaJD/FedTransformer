# federated_learning/models/validator.py
# Peer validation — verify that each update improves the model
# and that the XAI explanation is coherent.

import torch
import numpy as np
from typing import Dict, Tuple
from torch.utils.data import DataLoader

from utils.metrics import evaluate_model


class UpdateValidator:
    """
    Each incoming client update passes two checks:
      1. Performance check — does the local model perform acceptably?
      2. XAI coherence check — does the explanation make statistical sense?
    """

    def __init__(self, test_loader: DataLoader, device: torch.device = torch.device("cpu")):
        self.test_loader = test_loader
        self.device = device

    def validate(
        self,
        client_id: int,
        global_model: torch.nn.Module,
        local_state_dict: Dict,
        explanation: Dict,
    ) -> Tuple[bool, Dict]:
        """Returns (is_valid, report_dict)."""
        from federated_learning.models.transformer import ISACTransformer

        # Evaluate global model
        global_metrics = evaluate_model(global_model, self.test_loader, self.device)
        global_acc = global_metrics["accuracy"]

        # Evaluate local model
        local_model = ISACTransformer().to(self.device)
        local_model.load_state_dict(local_state_dict)
        local_metrics = evaluate_model(local_model, self.test_loader, self.device)
        local_acc = local_metrics["accuracy"]

        perf_ok = local_acc >= global_acc - 0.05  # allow small tolerance
        xai_ok = self._check_xai(explanation)
        is_valid = perf_ok and xai_ok

        report = {
            "client_id":  client_id,
            "global_acc": round(global_acc, 4),
            "local_acc":  round(local_acc, 4),
            "perf_ok":    perf_ok,
            "xai_ok":     xai_ok,
            "accepted":   is_valid,
        }
        status = "✓ accepted" if is_valid else "✗ rejected"
        print(f"[Validator] Client {client_id}: acc {global_acc:.3f}→{local_acc:.3f} "
              f"| xai={xai_ok} → {status}")
        return is_valid, report

    def _check_xai(self, explanation: Dict) -> bool:
        """XAI coherence: at least one feature importance must be meaningful."""
        if not explanation or "mean_abs_shap" not in explanation:
            return False
        max_imp = max(explanation["mean_abs_shap"].values(), default=0)
        return max_imp > 0.001
