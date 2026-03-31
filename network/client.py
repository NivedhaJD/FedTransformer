"""
client.py
---------
Federated Learning edge-node client for 6G-ISAC simulation.

Each client:
  1. Receives the current global model weights from the server.
  2. Trains locally on its private ISAC dataset for E local epochs.
  3. Returns the updated model weights (and number of local samples) to the server.

Local training implements:
  - Forward pass through the ISAC Transformer
  - Cross-Entropy Loss  L = -(1/m) Σ [y log(ŷ) + (1-y) log(1-ŷ)]
  - Gradient Descent update  w_{t+1} = w_t - η ∇L(w_t)
"""

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from federated_learning.models.transformer import ISACTransformer
from network.dataset import ISACDataset, get_dataloader
from utils.metrics import quick_accuracy
from utils.logger import log_client_metrics


# ---------------------------------------------------------------------------
# ISACClient
# ---------------------------------------------------------------------------
class ISACClient:
    """
    Simulates a single 6G edge-node client in the federated system.

    Parameters
    ----------
    client_id      : unique integer identifier for this node
    num_samples    : size of the node's local dataset
    seq_len        : sequence length for time-series samples
    local_epochs   : number of training epochs per communication round
    batch_size     : mini-batch size
    learning_rate  : η — step size for gradient descent
    device         : torch device (cpu / cuda)
    """

    def __init__(
        self,
        client_id: int,
        num_samples: int = 500,
        seq_len: int = 32,
        local_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: torch.device = torch.device("cpu"),
    ):
        self.client_id     = client_id
        self.num_samples   = num_samples
        self.local_epochs  = local_epochs
        self.learning_rate = learning_rate
        self.device        = device

        # Load this node's private dataset
        self.dataloader: DataLoader = get_dataloader(
            node_id=client_id,
            num_samples=num_samples,
            seq_len=seq_len,
            batch_size=batch_size,
            shuffle=True,
        )

        # The local model is a copy of the global model; initialised later
        self.model: ISACTransformer = None

    # -----------------------------------------------------------------------
    def set_model(self, global_state_dict: dict):
        """Load global model weights into the local model."""
        self.model = ISACTransformer().to(self.device)
        # Deep-copy weights so the local model is fully independent
        self.model.load_state_dict(copy.deepcopy(global_state_dict))

    # -----------------------------------------------------------------------
    def train(self, round_num: int) -> dict:
        """
        Execute local training for one communication round.

        Returns
        -------
        dict with:
            state_dict      — updated model weights
            num_samples     — dataset size (used for FedAvg weighting)
            local_accuracy  — final epoch accuracy on local data
            local_loss      — final epoch loss
            training_time_s — wall-clock training time
        """
        assert self.model is not None, "Call set_model() before train()."

        self.model.train()

        # NLLLoss expects log-probabilities (our model outputs log_softmax)
        criterion = nn.NLLLoss()

        # Gradient descent optimiser: w_{t+1} = w_t - η ∇L(w_t)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        start_time = time.time()
        final_loss, final_acc = 0.0, 0.0

        for epoch in range(self.local_epochs):
            epoch_loss, epoch_acc, batches = 0.0, 0.0, 0

            for X_batch, y_batch in self.dataloader:
                X_batch = X_batch.to(self.device)   # (B, seq_len, features)
                y_batch = y_batch.to(self.device)   # (B,)

                # ── Forward pass ──────────────────────────────────────────
                optimizer.zero_grad()
                log_probs = self.model(X_batch)     # (B, num_classes)

                # Cross-Entropy Loss (via NLLLoss + log_softmax)
                # Equivalent to: L = -(1/m) Σ y log(ŷ)
                loss = criterion(log_probs, y_batch)

                # ── Backward pass & weight update ─────────────────────────
                # Computes ∇L(w_t) and applies w_{t+1} = w_t - η ∇L(w_t)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc  += quick_accuracy(log_probs.detach(), y_batch)
                batches    += 1

            final_loss = epoch_loss / max(batches, 1)
            final_acc  = epoch_acc  / max(batches, 1)

        training_time = time.time() - start_time

        # Log client-level metrics
        log_client_metrics(
            round_num=round_num,
            client_id=self.client_id,
            num_samples=self.num_samples,
            local_accuracy=final_acc,
            local_loss=final_loss,
            training_time_s=training_time,
        )

        print(
            f"  [Client {self.client_id:02d}] "
            f"Round {round_num} | "
            f"Loss: {final_loss:.4f} | "
            f"Acc: {final_acc:.4f} | "
            f"Time: {training_time:.2f}s"
        )

        return {
            "state_dict"     : copy.deepcopy(self.model.state_dict()),
            "num_samples"    : self.num_samples,
            "local_accuracy" : final_acc,
            "local_loss"     : final_loss,
            "training_time_s": training_time,
        }


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------
def build_clients(
    num_clients: int,
    samples_per_client: int = 500,
    seq_len: int = 32,
    local_epochs: int = 3,
    learning_rate: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> list:
    """Create a list of ISACClient instances."""
    return [
        ISACClient(
            client_id=i,
            num_samples=samples_per_client,
            seq_len=seq_len,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            device=device,
        )
        for i in range(num_clients)
    ]


if __name__ == "__main__":
    # Quick single-client test
    from federated_learning.models.transformer import ISACTransformer

    device = torch.device("cpu")
    global_model = ISACTransformer().to(device)

    client = ISACClient(client_id=0, num_samples=200, local_epochs=2, device=device)
    client.set_model(global_model.state_dict())
    result = client.train(round_num=1)
    print("Returned keys:", list(result.keys()))
    print(f"Num samples: {result['num_samples']}")
