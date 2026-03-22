# 📡 Federated Transformer for 6G-ISAC Networks

A complete research simulation of **Federated Learning** applied to **6G Integrated Sensing and Communication (ISAC)** using a **Transformer neural network**.

---

## 🔬 What This Simulates

Multiple 6G edge nodes (base stations / devices) collaboratively train a Transformer model
**without sharing raw data** — only model weights are exchanged with a central server.

### Network States Classified
| Label | Class | Description |
|-------|-------|-------------|
| 0 | Normal | Typical operation |
| 1 | High Interference | Elevated noise/jamming |
| 2 | Target Detected | Radar/sensing event |
| 3 | Congestion | Overload condition |

---

## 🧮 Key Equations Implemented

### Federated Averaging (FedAvg)
```
w_{t+1} = Σ_i (n_i / N) · w_i^t
```

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) · V
```

### Softmax
```
softmax(x_i) = e^{x_i} / Σ e^{x_j}
```

### Cross-Entropy Loss
```
L = -(1/m) Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]
```

### Gradient Descent
```
w_{t+1} = w_t - η · ∇L(w_t)
```

---

## 📁 File Structure

```
federated_isac/
├── model.py            ← Transformer (Positional Enc, MHA, FFN)
├── dataset.py          ← Synthetic 6G-ISAC data generator
├── client.py           ← Edge-node client (local training)
├── server.py           ← FL server (FedAvg aggregation)
├── train.py            ← Main simulation entry point
├── evaluation.py       ← Accuracy / Precision / Recall / F1 / Loss
├── metrics_logger.py   ← CSV logging utilities
├── dashboard.py        ← Streamlit real-time dashboard
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the federated training simulation
```bash
python train.py
```

Optional arguments:
```bash
python train.py --clients 10 --rounds 20 --clients_per_round 5 --local_epochs 3 --lr 0.001
```

### 3. Launch the dashboard
```bash
streamlit run dashboard.py
```
Open your browser at **http://localhost:8501**

---

## ⚙️ Configuration

Edit `CONFIG` dict at the top of `train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_clients` | 10 | Total 6G edge nodes |
| `clients_per_round` | 5 | Nodes per communication round |
| `num_rounds` | 20 | Total federated rounds (T) |
| `local_epochs` | 3 | Local training epochs (E) |
| `samples_per_client` | 500 | Dataset size per node |
| `learning_rate` | 1e-3 | Gradient descent step (η) |
| `d_model` | 64 | Transformer embedding size |
| `num_heads` | 4 | Attention heads |
| `num_layers` | 2 | Encoder layers |

---

## 📊 Dashboard Panels

1. **System Overview** — KPI cards (rounds, clients, accuracy, F1)
2. **Accuracy vs Round** — global model accuracy curve
3. **Loss vs Round** — NLL loss curve
4. **Precision / Recall / F1** — multi-metric chart
5. **Client Participation** — bar chart per round
6. **Per-client Heatmap** — local accuracy matrix (client × round)
7. **Raw Tables** — full CSV data

---

## 🗂️ Output Files

After training, the `outputs/` and `logs/` directories contain:

```
outputs/
├── global_model.pt          ← Final trained transformer weights
├── training_curves.png      ← Static training plots
└── training_history.json    ← Round-by-round metrics

logs/
├── global_metrics.csv       ← Per-round global evaluation
├── client_metrics.csv       ← Per-client per-round results
└── round_summary.csv        ← Round timing and participation
```

---

## 🧪 Individual Module Tests

```bash
python model.py       # Check transformer architecture
python dataset.py     # Verify data generation
python evaluation.py  # Smoke-test metric computation
```

---

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- Streamlit 1.28+
- See `requirements.txt` for full list

---

*Research simulation for 6G / cybersecurity educational purposes.*
