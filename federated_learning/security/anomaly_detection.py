# federated_learning/security/anomaly_detection.py
# Detect and reject malicious / poisoned model updates via cosine similarity.

import torch
from typing import Dict, List, Tuple


def cosine_similarity(
    global_weights: Dict[str, torch.Tensor],
    local_weights: Dict[str, torch.Tensor],
) -> float:
    """
    Measure how aligned a local update is with the global model.
    Returns a value in [-1, 1]; below the threshold → likely poisoned.
    """
    flat_global = torch.cat([v.float().flatten() for v in global_weights.values()])
    flat_local  = torch.cat([v.float().flatten() for v in local_weights.values()])
    dot  = torch.dot(flat_global, flat_local)
    norm = flat_global.norm() * flat_local.norm() + 1e-8
    return float(dot / norm)


class PoisoningDetector:
    """
    Compare each client's update against the global model.
    Updates that are too dissimilar are flagged as potential poisoning attacks.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self.quarantined: List[int] = []

    def check_update(
        self,
        client_id: int,
        global_weights: Dict[str, torch.Tensor],
        local_weights:  Dict[str, torch.Tensor],
    ) -> Tuple[bool, float]:
        """Returns (is_clean, similarity_score)."""
        sim = cosine_similarity(global_weights, local_weights)
        is_clean = sim >= self.threshold

        status = "CLEAN" if is_clean else "POISONED ⚠"
        print(f"[AnomalyDetector] Client {client_id}: similarity={sim:.4f} → {status}")

        if not is_clean:
            self.quarantined.append(client_id)

        return is_clean, sim

    def filter_updates(
        self,
        global_weights: Dict[str, torch.Tensor],
        client_results: List[dict],
    ) -> Tuple[List[dict], Dict[int, float]]:
        """
        Run all client results through the similarity check.
        Returns:
          - clean_updates: list safe to pass to FedAvg
          - similarity_report: {client_id: similarity_score}
        """
        self.quarantined = []
        clean_updates = []
        report = {}

        for result in client_results:
            cid = result["client_id"]
            is_clean, sim = self.check_update(cid, global_weights, result["state_dict"])
            report[cid] = round(sim, 4)
            if is_clean:
                clean_updates.append(result)

        rejected = [c for c, s in report.items() if s < self.threshold]
        print(f"[AnomalyDetector] {len(clean_updates)}/{len(client_results)} "
              f"updates accepted. Rejected: {rejected}")
        return clean_updates, report
