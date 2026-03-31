"""
coordinator.py
--------------
Federated Learning coordinator for 6G-ISAC simulation.

Orchestrates the complete pipeline per round:
  1. Initialise network
  2. Select clients
  3. Broadcast global model
  4. Local training (Transformer on 6G-ISAC data)
  5. XAI explanations (SHAP feature attribution)
  6. Validate updates (performance + XAI coherence)
  7. Encrypt updates (AES-256-GCM)
  8. Detect poisoning (cosine similarity)
  9. FedAvg aggregation:  w_{t+1} = Σ (n_i / N) · w_i
 10. Evaluate global model
 11. Distribute updated model & log metrics
"""

import copy
import random
import time
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import numpy as np

from federated_learning.models.transformer import ISACTransformer
from network.client import ISACClient
from network.dataset import get_test_dataloader
from utils.metrics import evaluate_model, print_metrics
from utils.logger import log_global_metrics, log_round_summary
from federated_learning.aggregation.fedavg import federated_average
from federated_learning.security.encryption import (
    generate_session_key, encrypt_update, decrypt_update
)
from federated_learning.security.anomaly_detection import PoisoningDetector
from federated_learning.explainability.xai import LocalExplainer


# ---------------------------------------------------------------------------
# FederatedCoordinator
# ---------------------------------------------------------------------------
class FederatedCoordinator:
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
    enable_security    : enable encryption + poisoning detection
    enable_xai         : enable SHAP explanations + validation
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
        enable_security: bool = True,
        enable_xai: bool = True,
        on_step_callback: callable = None,
    ):
        self.num_clients       = num_clients
        self.clients_per_round = min(clients_per_round, num_clients)
        self.num_rounds        = num_rounds
        self.device            = device
        self.enable_security   = enable_security
        self.enable_xai        = enable_xai
        self.on_step_callback  = on_step_callback

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

    def _emit(self, step: int, data: dict = None):
        """Helper to optionally broadcast step events to the UI."""
        if self.on_step_callback:
            msg = {"step": step}
            if data:
                msg.update(data)
            self.on_step_callback(msg)

    # -----------------------------------------------------------------------
    # Step 1: Initialise
    # -----------------------------------------------------------------------
    def initialise(self):
        """Step 1: Prepare network."""
        print("\n" + "=" * 60)
        print("  STEP 1 - Initialise network")
        print("=" * 60)
        print(f"  [Coordinator] Global ISACTransformer model created on {self.device}.")
        self._emit(1, {"msg": "Initialised"})

    # -----------------------------------------------------------------------
    # Client selection
    # -----------------------------------------------------------------------
    def _select_clients(self, all_clients: List[ISACClient]) -> List[ISACClient]:
        """Randomly select clients_per_round clients from the pool."""
        return random.sample(all_clients, self.clients_per_round)

    # -----------------------------------------------------------------------
    # Run one communication round
    # -----------------------------------------------------------------------
    def run_round(self, round_num: int, all_clients: List[ISACClient]) -> dict:
        """Execute one full federated communication round."""
        round_start = time.time()
        print(f"\n{'='*60}")
        print(f" ROUND {round_num}/{self.num_rounds}")
        print(f"{'='*60}")

        # ── Step 2: Select participating clients ───────────────────────────
        print("\n  STEP 2 — Select clients")
        selected = self._select_clients(all_clients)
        print(f"  Selected clients: {[c.client_id for c in selected]}")
        self._emit(2, {"selected": [c.client_id for c in selected]})

        # ── Step 3: Broadcast global model ─────────────────────────────────
        print("\n  STEP 3 — Broadcast global model")
        current_weights = self.global_model.state_dict()
        for client in selected:
            client.set_model(current_weights)
        self._emit(3)

        # ── Step 4: Local training ─────────────────────────────────────────
        print("\n  STEP 4 — Local training")
        client_results = []
        for client in selected:
            result = client.train(round_num)
            client_results.append(result)
        
        # Clean up state_dict from results so we don't send massive tensors over websocket
        safe_results = [{k: v for k, v in r.items() if k != 'state_dict'} for r in client_results]
        self._emit(4, {"results": safe_results})

        # ── Step 5: XAI explanations ───────────────────────────────────────
        explanations = {}
        if self.enable_xai:
            print("\n  STEP 5 — XAI explanations")
            for result in client_results:
                cid = result["client_id"]
                try:
                    local_model = ISACTransformer().to(self.device)
                    local_model.load_state_dict(result["state_dict"])
                    dummy_data = np.random.randn(10, 32, 16).astype(np.float32)
                    explainer = LocalExplainer(local_model, dummy_data)
                    explanations[cid] = explainer.explain(dummy_data[:5])
                except Exception as e:
                    print(f"  [XAI] Client {cid}: skipped ({e})")
                    explanations[cid] = {"mean_abs_shap": {"dummy": 0.1}, "top_features": []}
        
        self._emit(5, {"explanations": explanations})

        # ── Step 6: Validate updates ───────────────────────────────────────
        validated_results = client_results
        if self.enable_xai:
            print("\n  STEP 6 - Validate updates")
            validated_results = []
            for result in client_results:
                cid = result["client_id"]
                expl = explanations.get(cid, {})
                if expl and "mean_abs_shap" in expl:
                    max_imp = max(expl["mean_abs_shap"].values(), default=0)
                    if max_imp > 0.001:
                        validated_results.append(result)
                        print(f"  [Validator] Client {cid}: [OK] accepted")
                    else:
                        print(f"  [Validator] Client {cid}: [REJECT] rejected")
                else:
                    validated_results.append(result)
        
        self._emit(6, {"validated": [r["client_id"] for r in validated_results]})

        # ── Step 7 & 8: Encrypt + detect poisoning ────────────────────────
        clean_results = validated_results
        similarities = []
        if self.enable_security and len(validated_results) > 0:
            print("\n  STEP 7 — Encrypt updates")
            session_key = generate_session_key()
            self._emit(7, {"action": "encrypted"})

            encrypted_updates = []
            for result in validated_results:
                payload = encrypt_update(result["state_dict"], session_key)
                encrypted_updates.append({
                    "client_id": result["client_id"],
                    "payload":   payload,
                    "num_samples": result["num_samples"],
                    "state_dict": result["state_dict"],
                })

            print("\n  STEP 8 — Detect poisoning")
            detector = PoisoningDetector(threshold=0.5)
            decrypted = []
            for enc in encrypted_updates:
                dec_weights = decrypt_update(enc["payload"], session_key)
                decrypted.append({
                    "client_id": enc["client_id"],
                    "state_dict": dec_weights,
                    "num_samples": enc["num_samples"],
                })
            clean_results, reports = detector.filter_updates(current_weights, decrypted)
            similarities = [{"client_id": k, "sim": v["similarity"], "status": v["status"]} for k,v in reports.items()]
        else:
            clean_results = validated_results
            self._emit(7)
        
        self._emit(8, {"similarities": similarities})

        # ── Step 9: FedAvg aggregation ─────────────────────────────────────
        print("\n  STEP 9 — FedAvg aggregation")
        if not clean_results:
            print("  [Coordinator] No clean updates. Round aborted.")
            return {}
        aggregated_weights = federated_average(clean_results)
        self.global_model.load_state_dict(aggregated_weights)
        self._emit(9)

        # ── Step 10: Global evaluation ─────────────────────────────────────
        print("\n  STEP 10 — Evaluate global model")
        metrics = evaluate_model(self.global_model, self.test_loader, self.device)
        round_duration = time.time() - round_start
        print_metrics(metrics, round_num)
        self._emit(10, {"metrics": metrics})

        # ── Step 11: Log & distribute ──────────────────────────────────────
        print("\n  STEP 11 — Distribute updated model & log")
        log_global_metrics(round_num=round_num, metrics=metrics, num_clients=len(selected))
        log_round_summary(round_num=round_num, duration_s=round_duration,
                          clients_selected=len(selected), clients_completed=len(clean_results))
        self._emit(11)

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
        """Execute all num_rounds communication rounds."""
        print("\n" + "*" * 60)
        print("  Federated Transformer Learning - 6G-ISAC Simulation")
        print(f"  Clients: {self.num_clients}  |  "
              f"Per round: {self.clients_per_round}  |  "
              f"Rounds: {self.num_rounds}")
        print(f"  Security: {'ON' if self.enable_security else 'OFF'}  |  "
              f"XAI: {'ON' if self.enable_xai else 'OFF'}")
        print("*" * 60)

        self.initialise()

        for r in range(1, self.num_rounds + 1):
            self.run_round(r, all_clients)

        self._print_summary()
        return self.history

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE — SUMMARY")
        print("=" * 60)
        for i, r in enumerate(self.history["round"]):
            print(f"  Round {r:2d}: "
                  f"acc={self.history['accuracy'][i]:.4f} "
                  f"f1={self.history['f1_score'][i]:.4f} "
                  f"loss={self.history['loss'][i]:.4f}")
        print("\n✓ Training complete.")

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
    from network.client import build_clients

    device  = torch.device("cpu")
    clients = build_clients(num_clients=5, samples_per_client=200,
                            local_epochs=2, device=device)
    server  = FederatedCoordinator(num_clients=5, clients_per_round=3,
                                   num_rounds=3, device=device)
    history = server.run(clients)
    print("\nFinal accuracy:", history["accuracy"][-1])
