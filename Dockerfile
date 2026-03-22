# ============================================================
# Dockerfile — Federated Transformer for 6G-ISAC
# ============================================================
# Multi-stage build:
#   builder  — installs all dependencies into /install/deps
#   runtime  — lean final image with only what's needed
# ============================================================

# ── Stage 1: builder ────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /install

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install ALL deps into the same prefix — torch CPU-only first, then the rest
RUN pip install --no-cache-dir --prefix=/install/deps \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --prefix=/install/deps -r requirements.txt


# ── Stage 2: runtime ────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="FedTransformer 6G-ISAC" \
      org.opencontainers.image.description="Federated Learning with Transformers for 6G-ISAC simulation"

RUN useradd --create-home --shell /bin/bash feduser

WORKDIR /app

# Copy ALL installed packages from builder in one shot
COPY --from=builder /install/deps /usr/local

# Copy project source
COPY --chown=feduser:feduser . .

# Create runtime directories
RUN mkdir -p logs outputs && chown -R feduser:feduser logs outputs

USER feduser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

EXPOSE 8501
EXPOSE 8000

CMD ["python", "train.py"]

# ── Alternative targets ─────────────────────────────────────
# Launch the Streamlit dashboard:
#   docker run -p 8501:8501 fedtransformer streamlit run dashboard.py --server.address 0.0.0.0
#
# Override training args:
#   docker run fedtransformer python train.py --clients 10 --rounds 20 --clients_per_round 5
