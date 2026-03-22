"""
dashboard.py
------------
Streamlit real-time monitoring dashboard for the Federated 6G-ISAC simulation.

Launch with:
    streamlit run dashboard.py

The dashboard auto-refreshes every N seconds and reads the CSV logs written
by metrics_logger.py during (or after) a training run.

Panels:
  ① System overview cards  (rounds, clients, best accuracy, best F1)
  ② Accuracy vs Round      (line chart)
  ③ Loss vs Round          (line chart)
  ④ Precision / Recall / F1 (multi-line chart)
  ⑤ Client participation   (bar chart — unique clients per round)
  ⑥ Per-client heatmap     (training time and local accuracy)
  ⑦ Raw metrics table
"""

import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from metrics_logger import read_global_metrics, read_client_metrics, read_round_summary

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="6G-ISAC Federated Learning Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark tech aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

    /* Main background */
    .stApp { background-color: #0a0e1a; color: #e0e6f0; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0d1220; border-right: 1px solid #1e2d4a; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #0d1f35 0%, #0a2040 100%);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 0 12px rgba(0, 120, 255, 0.08);
    }
    div[data-testid="metric-container"] label { color: #7aa3cc !important; font-size: 0.75rem; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #4fc3f7 !important; font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem;
    }

    /* Section headers */
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        color: #4fc3f7;
        font-size: 0.85rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 6px;
        margin: 24px 0 12px 0;
    }

    /* Status badge */
    .badge-live   { background:#0f3 ; color:#000; padding:2px 10px; border-radius:12px; font-size:.75rem; font-weight:700; }
    .badge-done   { background:#48e ; color:#fff; padding:2px 10px; border-radius:12px; font-size:.75rem; font-weight:700; }
    .badge-wait   { background:#fa0 ; color:#000; padding:2px 10px; border-radius:12px; font-size:.75rem; font-weight:700; }

    /* Plotly container */
    .js-plotly-plot { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Plotly theme helper
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(10,14,26,0)",
    plot_bgcolor ="rgba(13,28,50,0.6)",
    font=dict(family="IBM Plex Mono", color="#c0cfe0", size=11),
    xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    margin=dict(l=40, r=20, t=40, b=40),
    hovermode="x unified",
)

COLORS = {
    "accuracy" : "#4fc3f7",
    "loss"     : "#ef5350",
    "precision": "#ffb74d",
    "recall"   : "#aed581",
    "f1"       : "#ce93d8",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3)
def load_global() -> pd.DataFrame:
    rows = read_global_metrics()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ["accuracy", "precision", "recall", "f1_score", "loss"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    return df.dropna(subset=["round"]).sort_values("round")


@st.cache_data(ttl=3)
def load_clients() -> pd.DataFrame:
    rows = read_client_metrics()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ["local_accuracy", "local_loss", "training_time_s", "num_samples"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df["client_id"] = pd.to_numeric(df["client_id"], errors="coerce")
    return df.dropna(subset=["round"])


@st.cache_data(ttl=3)
def load_rounds() -> pd.DataFrame:
    rows = read_round_summary()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ["duration_s", "clients_selected", "clients_completed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    return df.dropna(subset=["round"]).sort_values("round")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📡 6G-ISAC FL Monitor")
    st.markdown("---")

    auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**Algorithm**")
    st.markdown(
        """
        `FedAvg`:
        ```
        w_{t+1} = Σ (n_i/N) w_i^t
        ```
        `Attention`:
        ```
        softmax(QKᵀ/√d_k)·V
        ```
        `Loss (CE)`:
        ```
        -1/m Σ y·log(ŷ)
        ```
        `Gradient Descent`:
        ```
        w = w - η·∇L(w)
        ```
        """
    )
    st.markdown("---")
    st.caption("Federated Transformer · 6G ISAC")

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='color:#4fc3f7; font-family:IBM Plex Mono; font-size:1.6rem; margin-bottom:0;'>"
    "📡 Federated Transformer · 6G-ISAC Dashboard"
    "</h1>",
    unsafe_allow_html=True,
)
st.caption("Real-time monitoring of federated edge-node training across 6G ISAC network nodes")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
gdf = load_global()
cdf = load_clients()
rdf = load_rounds()

has_data = not gdf.empty

# Status badge
if has_data:
    completed = int(gdf["round"].max())
    st.markdown(f"<span class='badge-done'>✓ {completed} rounds logged</span>", unsafe_allow_html=True)
else:
    st.markdown("<span class='badge-wait'>⏳ Awaiting training data…  Run `python train.py` first.</span>",
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ① Overview KPI cards
# ---------------------------------------------------------------------------
st.markdown("<div class='section-header'>① System Overview</div>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)

if has_data:
    latest = gdf.iloc[-1]
    best_acc = gdf["accuracy"].max()
    best_f1  = gdf["f1_score"].max()
    total_rounds  = int(gdf["round"].max())
    active_clients = int(latest["num_clients"]) if "num_clients" in latest else 0
    c1.metric("Training Rounds",  f"{total_rounds}")
    c2.metric("Active Clients",   f"{active_clients}")
    c3.metric("Latest Accuracy",  f"{latest['accuracy']:.3f}")
    c4.metric("Best Accuracy",    f"{best_acc:.3f}")
    c5.metric("Best F1 Score",    f"{best_f1:.3f}")
else:
    for col, lbl in zip([c1,c2,c3,c4,c5], ["Rounds","Clients","Accuracy","Best Acc","Best F1"]):
        col.metric(lbl, "—")

# ---------------------------------------------------------------------------
# ② & ③ Accuracy + Loss charts
# ---------------------------------------------------------------------------
st.markdown("<div class='section-header'>② Accuracy & Loss vs Training Round</div>",
            unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    if has_data:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gdf["round"], y=gdf["accuracy"],
            name="Accuracy", mode="lines+markers",
            line=dict(color=COLORS["accuracy"], width=2.5),
            marker=dict(size=6),
        ))
        fig.update_layout(
            title="Global Model Accuracy",
            yaxis_range=[0, 1],
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No accuracy data yet.")

with col_b:
    if has_data:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gdf["round"], y=gdf["loss"],
            name="Loss", mode="lines+markers",
            line=dict(color=COLORS["loss"], width=2.5),
            marker=dict(size=6),
        ))
        fig.update_layout(title="Global Model Loss (NLL)", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No loss data yet.")

# ---------------------------------------------------------------------------
# ④ Precision / Recall / F1
# ---------------------------------------------------------------------------
st.markdown("<div class='section-header'>③ Precision · Recall · F1-Score</div>",
            unsafe_allow_html=True)

if has_data:
    fig = go.Figure()
    for metric, color in [("f1_score", COLORS["f1"]),
                           ("precision", COLORS["precision"]),
                           ("recall", COLORS["recall"])]:
        fig.add_trace(go.Scatter(
            x=gdf["round"], y=gdf[metric],
            name=metric.replace("_", " ").title(),
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5),
        ))
    fig.update_layout(
        title="F1 / Precision / Recall per Round",
        yaxis_range=[0, 1],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No metric data yet.")

# ---------------------------------------------------------------------------
# ⑤ Client participation
# ---------------------------------------------------------------------------
st.markdown("<div class='section-header'>④ Client Participation per Round</div>",
            unsafe_allow_html=True)

col_p, col_t = st.columns(2)

with col_p:
    if not rdf.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=rdf["round"], y=rdf["clients_completed"],
            marker_color="#4fc3f7",
            name="Clients completed",
        ))
        fig.add_trace(go.Bar(
            x=rdf["round"], y=rdf["clients_selected"] - rdf["clients_completed"],
            marker_color="#263d5e",
            name="Selected but dropped",
        ))
        fig.update_layout(
            title="Client Participation",
            barmode="stack",
            xaxis_title="Round",
            yaxis_title="Clients",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No round summary data yet.")

with col_t:
    if not rdf.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=rdf["round"], y=rdf["duration_s"],
            marker_color="#ce93d8",
            name="Round duration (s)",
        ))
        fig.update_layout(
            title="Round Duration (seconds)",
            xaxis_title="Round",
            yaxis_title="Seconds",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timing data yet.")

# ---------------------------------------------------------------------------
# ⑥ Per-client heatmap (accuracy)
# ---------------------------------------------------------------------------
st.markdown("<div class='section-header'>⑤ Per-Client Local Accuracy Heatmap</div>",
            unsafe_allow_html=True)

if not cdf.empty:
    pivot = (
        cdf.groupby(["round", "client_id"])["local_accuracy"]
        .mean()
        .unstack(fill_value=0)
    )
    fig = px.imshow(
        pivot.T,
        labels=dict(x="Round", y="Client ID", color="Local Acc"),
        color_continuous_scale="Blues",
        aspect="auto",
        title="Local Accuracy — Client × Round",
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No client data yet.")

# ---------------------------------------------------------------------------
# ⑦ Raw tables (expandable)
# ---------------------------------------------------------------------------
st.markdown("<div class='section-header'>⑥ Raw Metrics Tables</div>", unsafe_allow_html=True)

with st.expander("Global Metrics Table"):
    if has_data:
        st.dataframe(gdf.style.format({
            "accuracy": "{:.4f}", "precision": "{:.4f}",
            "recall": "{:.4f}", "f1_score": "{:.4f}", "loss": "{:.4f}",
        }), use_container_width=True)
    else:
        st.write("No data.")

with st.expander("Client Metrics Table"):
    if not cdf.empty:
        st.dataframe(cdf.style.format({
            "local_accuracy": "{:.4f}", "local_loss": "{:.4f}",
            "training_time_s": "{:.3f}",
        }), use_container_width=True)
    else:
        st.write("No data.")

# ---------------------------------------------------------------------------
# Footer + auto-refresh
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("6G-ISAC Federated Transformer Simulation  ·  Powered by PyTorch & Streamlit")

if auto_refresh:
    time.sleep(5)
    st.cache_data.clear()
    st.rerun()
