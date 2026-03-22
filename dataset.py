"""
dataset.py
----------
Synthetic 6G-ISAC dataset generator.

Each sample represents a snapshot of a 6G base-station node and includes:
  - Channel State Information (CSI) features
  - Signal strength / RSSI
  - Latency / Round-Trip-Time
  - Sensing signals (radar-like)
  - Network traffic features

Label classes (4 network states):
  0 — Normal operation
  1 — High interference
  2 — Mobile target detected (sensing event)
  3 — Congestion / overload
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    # Channel State Information (4 features)
    "csi_magnitude",
    "csi_phase",
    "doppler_shift",
    "multipath_delay",
    # Signal quality (3 features)
    "rssi_dbm",
    "snr_db",
    "ber",
    # Latency / timing (2 features)
    "rtt_ms",
    "jitter_ms",
    # Sensing / radar (4 features)
    "radar_range_m",
    "radar_velocity",
    "radar_rcs",
    "angle_of_arrival",
    # Network traffic (3 features)
    "throughput_mbps",
    "packet_loss_rate",
    "active_users",
]

NUM_FEATURES = len(FEATURE_NAMES)   # 16
NUM_CLASSES = 4
CLASS_NAMES = ["Normal", "High Interference", "Target Detected", "Congestion"]


# ---------------------------------------------------------------------------
# Per-class distribution parameters
# ---------------------------------------------------------------------------
_CLASS_PARAMS = {
    # (feature_mean_offsets, noise_scale)
    0: {"rssi": -60,  "snr": 25, "rtt": 10,  "throughput": 800, "interference": 0.0},
    1: {"rssi": -85,  "snr": 8,  "rtt": 30,  "throughput": 200, "interference": 1.5},
    2: {"rssi": -70,  "snr": 18, "rtt": 12,  "throughput": 600, "interference": 0.3},
    3: {"rssi": -75,  "snr": 12, "rtt": 60,  "throughput": 50,  "interference": 0.8},
}


def _generate_sample(label: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a single feature vector that statistically reflects
    the given network-state class.
    """
    p = _CLASS_PARAMS[label]

    csi_magnitude   = rng.normal(0.7 - label * 0.1, 0.05)
    csi_phase       = rng.uniform(-np.pi, np.pi)
    doppler_shift   = rng.normal(0.5 * (label == 2), 0.1)     # elevated for target
    multipath_delay = rng.exponential(0.05 + 0.02 * label)

    rssi_dbm        = rng.normal(p["rssi"], 5)
    snr_db          = rng.normal(p["snr"], 3)
    ber             = np.clip(rng.normal(0.01 * (label + 1), 0.005), 0, 1)

    rtt_ms          = rng.normal(p["rtt"], 5)
    jitter_ms       = rng.exponential(2 + label * 2)

    radar_range_m   = rng.normal(150 + 50 * (label == 2), 30)
    radar_velocity  = rng.normal(20 * (label == 2), 5)
    radar_rcs       = rng.normal(10 + 5 * (label == 2), 2)
    aoa             = rng.uniform(0, 360)

    throughput      = rng.normal(p["throughput"], 50)
    packet_loss     = np.clip(rng.normal(0.01 * (label + 1), 0.005), 0, 1)
    active_users    = rng.poisson(50 + 20 * (label == 3))

    features = np.array([
        csi_magnitude, csi_phase, doppler_shift, multipath_delay,
        rssi_dbm, snr_db, ber,
        rtt_ms, jitter_ms,
        radar_range_m, radar_velocity, radar_rcs, aoa,
        throughput, packet_loss, active_users,
    ], dtype=np.float32)

    return features


# ---------------------------------------------------------------------------
# ISACDataset — time-series version (seq_len steps per sample)
# ---------------------------------------------------------------------------
class ISACDataset(Dataset):
    """
    PyTorch Dataset of (sequence, label) pairs.

    Each sample is a tensor of shape (seq_len, NUM_FEATURES) representing
    a short time-series of 6G-ISAC measurements at one edge node.

    Parameters
    ----------
    num_samples : total number of samples
    seq_len     : number of time-steps per sample
    node_id     : used to set a reproducible RNG seed per node
    class_probs : class sampling probabilities (default: uniform)
    noise_std   : additive Gaussian noise level for node-specific variation
    """

    def __init__(
        self,
        num_samples: int = 500,
        seq_len: int = 32,
        node_id: int = 0,
        class_probs: list = None,
        noise_std: float = 0.05,
    ):
        rng = np.random.default_rng(seed=42 + node_id)

        if class_probs is None:
            # Slightly imbalanced classes to mimic real deployments
            class_probs = [0.4, 0.25, 0.2, 0.15]

        self.seq_len = seq_len
        self.num_features = NUM_FEATURES

        X_list, y_list = [], []

        for _ in range(num_samples):
            label = rng.choice(NUM_CLASSES, p=class_probs)
            # Build a short time-series for this label
            sequence = np.stack(
                [_generate_sample(label, rng) for _ in range(seq_len)], axis=0
            )
            # Add node-specific noise to simulate heterogeneous data
            sequence += rng.normal(0, noise_std, sequence.shape).astype(np.float32)
            X_list.append(sequence)
            y_list.append(label)

        self.X = np.array(X_list, dtype=np.float32)   # (N, seq_len, features)
        self.y = np.array(y_list, dtype=np.int64)      # (N,)

        # Z-score normalise each feature across the local dataset
        mean = self.X.mean(axis=(0, 1), keepdims=True)
        std  = self.X.std(axis=(0, 1), keepdims=True) + 1e-8
        self.X = (self.X - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
def get_dataloader(
    node_id: int,
    num_samples: int = 500,
    seq_len: int = 32,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Return a DataLoader for a single edge node."""
    ds = ISACDataset(num_samples=num_samples, seq_len=seq_len, node_id=node_id)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def get_test_dataloader(num_samples: int = 300, seq_len: int = 32, batch_size: int = 64) -> DataLoader:
    """A shared held-out test set (node_id=999 for unique seed)."""
    ds = ISACDataset(num_samples=num_samples, seq_len=seq_len, node_id=999, noise_std=0.01)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    dl = get_dataloader(node_id=0, num_samples=100)
    X, y = next(iter(dl))
    print(f"Batch X shape : {X.shape}")   # (32, 32, 16)
    print(f"Batch y shape : {y.shape}")   # (32,)
    print(f"Classes present: {y.unique().tolist()}")
