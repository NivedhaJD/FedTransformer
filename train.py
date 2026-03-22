"""
train.py
--------
Entry point for the Federated Transformer 6G-ISAC simulation.

Usage:
    python train.py
    python train.py --clients 10 --rounds 20 --clients_per_round 5

All configurable via CLI arguments or the CONFIG dict at the top of the file.
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")          # headless backend for server environments
import matplotlib.pyplot as plt

from client import build_clients
from server import FederatedServer
from metrics_logger import clear_logs


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
CONFIG = {
    # Federated setup
    "num_clients"        : 10,    # Total edge nodes in the 6G network
    "clients_per_round"  : 5,     # Nodes participating per communication round
    "num_rounds"         : 20,    # Total training rounds (T)

    # Local training
    "local_epochs"       : 3,     # Epochs each client trains locally (E)
    "samples_per_client" : 500,   # Dataset size per node
    "seq_len"            : 32,    # Time-series length per sample
    "batch_size"         : 32,
    "learning_rate"      : 1e-3,  # η — gradient descent step size

    # Model architecture
    "d_model"     : 64,
    "num_heads"   : 4,
    "num_layers"  : 2,
    "d_ff"        : 128,
    "num_classes" : 4,
    "dropout"     : 0.1,

    # Misc
    "seed"            : 42,
    "output_dir"      : "outputs",
    "save_model"      : True,
}


# ---------------------------------------------------------------------------
# CLI override
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Federated ISAC Transformer Training")
    parser.add_argument("--clients",           type=int, default=CONFIG["num_clients"])
    parser.add_argument("--rounds",            type=int, default=CONFIG["num_rounds"])
    parser.add_argument("--clients_per_round", type=int, default=CONFIG["clients_per_round"])
    parser.add_argument("--local_epochs",      type=int, default=CONFIG["local_epochs"])
    parser.add_argument("--lr",                type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--seed",              type=int, default=CONFIG["seed"])
    parser.add_argument("--no_save",           action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def save_plots(history: dict, out_dir: str):
    """Save training-curve PNG charts to out_dir."""
    Path(out_dir).mkdir(exist_ok=True)

    rounds = history["round"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Federated Transformer — 6G-ISAC Training Curves", fontsize=13, fontweight="bold")

    # ── Accuracy ─────────────────────────────────────────────────────────
    axes[0].plot(rounds, history["accuracy"], marker="o", color="#2196F3", linewidth=2)
    axes[0].set_title("Global Accuracy")
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # ── Loss ──────────────────────────────────────────────────────────────
    axes[1].plot(rounds, history["loss"], marker="s", color="#F44336", linewidth=2)
    axes[1].set_title("Global Loss")
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("NLL Loss")
    axes[1].grid(True, alpha=0.3)

    # ── F1 Score ──────────────────────────────────────────────────────────
    axes[2].plot(rounds, history["f1_score"], marker="^", color="#4CAF50", linewidth=2, label="F1")
    axes[2].plot(rounds, history["precision"], linestyle="--", color="#FF9800", linewidth=1.5, label="Precision")
    axes[2].plot(rounds, history["recall"],    linestyle=":",  color="#9C27B0", linewidth=1.5, label="Recall")
    axes[2].set_title("F1 / Precision / Recall")
    axes[2].set_xlabel("Communication Round")
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[train.py] Training curves saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Apply CLI overrides
    CONFIG["num_clients"]       = args.clients
    CONFIG["num_rounds"]        = args.rounds
    CONFIG["clients_per_round"] = args.clients_per_round
    CONFIG["local_epochs"]      = args.local_epochs
    CONFIG["learning_rate"]     = args.lr
    CONFIG["seed"]              = args.seed
    if args.no_save:
        CONFIG["save_model"] = False

    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train.py] Device: {device}")
    print(f"[train.py] Config: {json.dumps({k: v for k, v in CONFIG.items() if k != 'output_dir'}, indent=2)}")

    # ── Clear previous logs ─────────────────────────────────────────────────
    clear_logs()

    # ── Build clients ───────────────────────────────────────────────────────
    clients = build_clients(
        num_clients=CONFIG["num_clients"],
        samples_per_client=CONFIG["samples_per_client"],
        seq_len=CONFIG["seq_len"],
        local_epochs=CONFIG["local_epochs"],
        learning_rate=CONFIG["learning_rate"],
        device=device,
    )

    # ── Build server ────────────────────────────────────────────────────────
    model_kwargs = {
        "d_model"    : CONFIG["d_model"],
        "num_heads"  : CONFIG["num_heads"],
        "num_layers" : CONFIG["num_layers"],
        "d_ff"       : CONFIG["d_ff"],
        "num_classes": CONFIG["num_classes"],
        "dropout"    : CONFIG["dropout"],
    }

    server = FederatedServer(
        num_clients=CONFIG["num_clients"],
        clients_per_round=CONFIG["clients_per_round"],
        num_rounds=CONFIG["num_rounds"],
        model_kwargs=model_kwargs,
        device=device,
        seed=CONFIG["seed"],
    )

    # ── Run federated training ──────────────────────────────────────────────
    t0 = time.time()
    history = server.run(clients)
    elapsed = time.time() - t0

    # ── Post-training summary ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Training time   : {elapsed:.1f}s")
    print(f"  Final accuracy  : {history['accuracy'][-1]:.4f}")
    print(f"  Final loss      : {history['loss'][-1]:.4f}")
    print(f"  Final F1 score  : {history['f1_score'][-1]:.4f}")
    print(f"{'─'*60}")

    # ── Save artifacts ──────────────────────────────────────────────────────
    Path(CONFIG["output_dir"]).mkdir(exist_ok=True)

    if CONFIG["save_model"]:
        server.save_model(os.path.join(CONFIG["output_dir"], "global_model.pt"))

    save_plots(history, CONFIG["output_dir"])

    # Save history as JSON for the dashboard
    history_path = os.path.join(CONFIG["output_dir"], "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train.py] History saved → {history_path}")

    print("\n✓  All done.  Run  `streamlit run dashboard.py`  to open the dashboard.\n")


if __name__ == "__main__":
    main()
