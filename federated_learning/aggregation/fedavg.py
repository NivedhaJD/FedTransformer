import copy
import torch
from typing import List

def federated_average(client_results: List[dict]) -> dict:
    """
    Federated Averaging:
        w_{t+1} = Σ_i (n_i / N) · w_i^t
    """
    total_samples = sum(r["num_samples"] for r in client_results)

    # Start with a zero-filled copy of the global state dict
    aggregated = copy.deepcopy(client_results[0]["state_dict"])
    for key in aggregated:
        aggregated[key] = torch.zeros_like(aggregated[key], dtype=torch.float32)

    # Weighted sum: w_agg += (n_i / N) * w_i
    for result in client_results:
        weight = result["num_samples"] / total_samples
        for key in aggregated:
            aggregated[key] += weight * result["state_dict"][key].float()

    return aggregated
