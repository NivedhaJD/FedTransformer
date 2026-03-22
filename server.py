"""
server.py
---------
Federated Learning server for 6G-ISAC simulation.

Responsibilities:
  1. Maintain and distribute the global Transformer model.
  2. Select a subset of clients to participate in each round.
  3. Aggregate client updates using Federated Averaging (FedAvg):

        w_{t+1} = Σ_i (n_i / N) · w_i^t

     where n_i is the number of local samples on client i and N = Σ n_i.

  4. Evaluate the global model on a held-out test set after each round.
  5. Log all metrics via metrics_logger.
"""

import copy
import random
import time
from typing import List, Dict

import torch
import torch.nn as nn

from model import ISACTransformer
from client import ISACClient
from dataset import get_test_dataloader
from evaluation import evaluate_model, print_metrics
from metrics_logger import log_global_metrics, log_round_summary


# ---------------------------------------------------------------------------
# FederatedServer
# ---------------------------------------------------------------------------
class FederatedServer:
    """
    Central coordinator for the federated learning process.

    Parameters
    ----------
    num_clients        : total number of edge nodes in the system
    clients_per_round  : how many clients participate in each round
    num_rounds         : total number of communication rounds T
    model_kwargs       : keyword arguments forwarded to ISACTransformer
    test_samples       : size of the global evaluation dataset
    device             : torch device
    seed               : random seed for reproducibility
    """

    def __init__(
        self,
        num_clients: int = 10,
        clients_per_round: int = 5,
        num_rounds: int = 20,
        model_kwargs: dict = None,
        test_samples: int = 300,
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
    ):
        self.num_clients       = num_clients
        self.clients_per_round = min(clients_per_round, num_clients)
        self.num_rounds        = num_rounds
        self.device            = device

        random.seed(seed)
        torch.manual_seed(seed)

        # ── Global model ────────────────────────────────────────────────────
        if model_kwargs is None:
            model_kwargs = {}
        self.global_model = ISACTransformer(**model_kwargs).to(device)

        # ── Held-out evaluation set ─────────────────────────────────────────
        self.test_loader = get_test_dataloader(num_samples=test_samples)

        # ── History (for plotting) ──────────────────────────────────────────
        self.history: Dict[str, list] = {
            "round"          : [],
            "accuracy"       : [],
            "loss"           : [],
            "precision"      : [],
            "recall"         : [],
            "f1_score"       : [],
            "clients_used"   : [],
        }

    # -----------------------------------------------------------------------
    # FedAvg aggregation
    # -----------------------------------------------------------------------
    def _federated_average(self, client_results: List[dict]) -> dict:
        """
        Federated Averaging:

            w_{t+1} = Σ_i (n_i / N) · w_i^t

        client_results: list of dicts, each containing
            'state_dict'  — model weights from client i
            'num_samples' — n_i (dataset size at client i)
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

    # -----------------------------------------------------------------------
    # Client selection
    # -----------------------------------------------------------------------
    def _select_clients(self, all_clients: List[ISACClient]) -> List[ISACClient]:
        """
        Randomly select clients_per_round clients from the pool.
        In a real 6G system this would factor in channel quality and energy.
        """
        return random.sample(all_clients, self.clients_per_round)

    # -----------------------------------------------------------------------
    # Run one communication round
    # -----------------------------------------------------------------------
    def run_round(self, round_num: int, all_clients: List[ISACClient]) -> dict:
        """
        Execute one full federated communication round:
          1. Select clients
          2. Broadcast global weights
          3. Local training on each client
          4. Aggregate via FedAvg
          5. Evaluate global model
        """
        round_start = time.time()
        print(f"\n{'='*60}")
        print(f" ROUND {round_num}/{self.num_rounds}")
        print(f"{'='*60}")

        # ── Step 1: Select participating clients ───────────────────────────
        selected = self._select_clients(all_clients)
        print(f"  Selected clients: {[c.client_id for c in selected]}")

        # ── Step 2 & 3: Distribute global model; clients train locally ─────
        current_weights = self.global_model.state_dict()
        client_results  = []

        for client in selected:
            client.set_model(current_weights)        # broadcast
            result = client.train(round_num)         # local training
            client_results.append(result)

        # ── Step 4: FedAvg aggregation ─────────────────────────────────────
        aggregated_weights = self._federated_average(client_results)
        self.global_model.load_state_dict(aggregated_weights)

        # ── Step 5: Global evaluation ──────────────────────────────────────
        metrics = evaluate_model(self.global_model, self.test_loader, self.device)
        round_duration = time.time() - round_start

        print_metrics(metrics, round_num)

        # ── Logging ────────────────────────────────────────────────────────
        log_global_metrics(
            round_num=round_num,
            metrics=metrics,
            num_clients=len(selected),
        )
        log_round_summary(
            round_num=round_num,
            duration_s=round_duration,
            clients_selected=len(selected),
            clients_completed=len(client_results),
        )

        # ── Update history ─────────────────────────────────────────────────
        self.history["round"].append(round_num)
        self.history["accuracy"].append(metrics["accuracy"])
        self.history["loss"].append(metrics["loss"])
        self.history["precision"].append(metrics["precision"])
        self.history["recall"].append(metrics["recall"])
        self.history["f1_score"].append(metrics["f1_score"])
        self.history["clients_used"].append(len(selected))

        return metrics

    # -----------------------------------------------------------------------
    # Full training loop
    # -----------------------------------------------------------------------
    def run(self, all_clients: List[ISACClient]) -> Dict[str, list]:
        """
        Execute all num_rounds communication rounds.

        Returns
        -------
        history dict with per-round metrics (for plotting).
        """
        print("\n" + "★" * 60)
        print("  Federated Transformer Learning — 6G-ISAC Simulation")
        print(f"  Clients: {self.num_clients}  |  "
              f"Per round: {self.clients_per_round}  |  "
              f"Rounds: {self.num_rounds}")
        print("★" * 60)

        for r in range(1, self.num_rounds + 1):
            self.run_round(r, all_clients)

        print("\n✓ Training complete.")
        return self.history

    # -----------------------------------------------------------------------
    # Save / load helpers
    # -----------------------------------------------------------------------
    def save_model(self, path: str = "global_model.pt"):
        torch.save(self.global_model.state_dict(), path)
        print(f"[Server] Global model saved → {path}")

    def load_model(self, path: str = "global_model.pt"):
        state = torch.load(path, map_location=self.device)
        self.global_model.load_state_dict(state)
        print(f"[Server] Global model loaded ← {path}")


if __name__ == "__main__":
    from client import build_clients

    device  = torch.device("cpu")
    clients = build_clients(num_clients=5, samples_per_client=200,
                            local_epochs=2, device=device)
    server  = FederatedServer(num_clients=5, clients_per_round=3,
                              num_rounds=3, device=device)
    history = server.run(clients)
    print("\nFinal accuracy:", history["accuracy"][-1])
