"""
metrics_logger.py
-----------------
Logs per-round federated training metrics to CSV files and provides
helper functions for reading them back (used by the dashboard).
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Default output directory for all log files
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

GLOBAL_LOG   = LOG_DIR / "global_metrics.csv"
CLIENT_LOG   = LOG_DIR / "client_metrics.csv"
ROUND_LOG    = LOG_DIR / "round_summary.csv"

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def _write_row(filepath: Path, row: dict, fieldnames: List[str]):
    """Append one row to a CSV, creating the file with headers if needed."""
    file_exists = filepath.exists()
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
GLOBAL_FIELDS = ["timestamp", "round", "accuracy", "precision", "recall",
                 "f1_score", "loss", "num_clients"]

CLIENT_FIELDS = ["timestamp", "round", "client_id", "num_samples",
                 "local_accuracy", "local_loss", "training_time_s"]

ROUND_FIELDS  = ["timestamp", "round", "duration_s", "clients_selected",
                 "clients_completed", "aggregation_method"]


def log_global_metrics(round_num: int, metrics: Dict, num_clients: int):
    """
    Log aggregated (global) metrics after each federated round.

    metrics dict keys: accuracy, precision, recall, f1_score, loss
    """
    row = {
        "timestamp"  : datetime.utcnow().isoformat(),
        "round"      : round_num,
        "accuracy"   : round(metrics.get("accuracy", 0), 4),
        "precision"  : round(metrics.get("precision", 0), 4),
        "recall"     : round(metrics.get("recall", 0), 4),
        "f1_score"   : round(metrics.get("f1_score", 0), 4),
        "loss"       : round(metrics.get("loss", 0), 4),
        "num_clients": num_clients,
    }
    _write_row(GLOBAL_LOG, row, GLOBAL_FIELDS)


def log_client_metrics(round_num: int, client_id: int, num_samples: int,
                       local_accuracy: float, local_loss: float,
                       training_time_s: float):
    """Log metrics for one client in one round."""
    row = {
        "timestamp"      : datetime.utcnow().isoformat(),
        "round"          : round_num,
        "client_id"      : client_id,
        "num_samples"    : num_samples,
        "local_accuracy" : round(local_accuracy, 4),
        "local_loss"     : round(local_loss, 4),
        "training_time_s": round(training_time_s, 3),
    }
    _write_row(CLIENT_LOG, row, CLIENT_FIELDS)


def log_round_summary(round_num: int, duration_s: float,
                      clients_selected: int, clients_completed: int,
                      aggregation_method: str = "FedAvg"):
    """Log a high-level summary for each communication round."""
    row = {
        "timestamp"          : datetime.utcnow().isoformat(),
        "round"              : round_num,
        "duration_s"         : round(duration_s, 3),
        "clients_selected"   : clients_selected,
        "clients_completed"  : clients_completed,
        "aggregation_method" : aggregation_method,
    }
    _write_row(ROUND_LOG, row, ROUND_FIELDS)


# ---------------------------------------------------------------------------
# Read-back helpers (used by dashboard.py)
# ---------------------------------------------------------------------------
def read_global_metrics() -> List[Dict]:
    """Return list of dicts from global_metrics.csv, or [] if missing."""
    if not GLOBAL_LOG.exists():
        return []
    with open(GLOBAL_LOG, newline="") as f:
        return list(csv.DictReader(f))


def read_client_metrics() -> List[Dict]:
    if not CLIENT_LOG.exists():
        return []
    with open(CLIENT_LOG, newline="") as f:
        return list(csv.DictReader(f))


def read_round_summary() -> List[Dict]:
    if not ROUND_LOG.exists():
        return []
    with open(ROUND_LOG, newline="") as f:
        return list(csv.DictReader(f))


def clear_logs():
    """Delete existing log files (call before a fresh training run)."""
    for p in [GLOBAL_LOG, CLIENT_LOG, ROUND_LOG]:
        if p.exists():
            p.unlink()
    print("[MetricsLogger] All log files cleared.")


if __name__ == "__main__":
    # Quick smoke-test
    clear_logs()
    log_global_metrics(1, {"accuracy": 0.72, "precision": 0.70,
                            "recall": 0.68, "f1_score": 0.69, "loss": 0.85}, 5)
    log_client_metrics(1, 0, 500, 0.70, 0.90, 2.3)
    log_round_summary(1, 15.2, 5, 5)
    print("Global metrics:", read_global_metrics())
    print("Client metrics:", read_client_metrics())
    print("Round summary :", read_round_summary())
