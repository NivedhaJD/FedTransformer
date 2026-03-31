"""
model.py
--------
Transformer Neural Network for 6G-ISAC Federated Learning.

Implements:
  - Positional Encoding
  - Scaled Dot-Product Attention: Attention(Q,K,V) = softmax(QKᵀ / √d_k) V
  - Multi-Head Attention
  - Feed-Forward Network
  - Full Transformer Encoder + Classification Head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Injects sequence-position information into the token embeddings.
    Uses fixed sine / cosine encoding (Vaswani et al., 2017).
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the (max_len, d_model) encoding table once
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)           # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )                                                                               # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                                           # (1, L, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention
# ---------------------------------------------------------------------------
class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) · V

    Q, K, V — query, key, value matrices
    d_k      — dimension of key vectors (used for scaling)
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        d_k = Q.size(-1)                                  # scaling factor √d_k

        # Raw attention scores: (batch, heads, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax: softmax(x_i) = e^x_i / Σ e^x_j
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum over values
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    Projects Q, K, V into h parallel heads, applies scaled dot-product
    attention in each head, then concatenates and projects back.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        batch, seq, _ = x.size()
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None
    ) -> torch.Tensor:
        batch = Q.size(0)

        # Project → split into heads
        Q = self._split_heads(self.W_q(Q))
        K = self._split_heads(self.W_k(K))
        V = self._split_heads(self.W_v(V))

        # Attention in parallel across all heads
        out, _ = self.attention(Q, K, V, mask)

        # Concatenate heads: (batch, heads, seq, d_k) → (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.num_heads * self.d_k)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# Position-wise Feed-Forward Network
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """Two-layer FFN with ReLU activation and dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ---------------------------------------------------------------------------
# Transformer Encoder Layer
# ---------------------------------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    """
    Single encoder block:
      x → Multi-Head Attention → Add & Norm → FFN → Add & Norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # Self-attention sub-layer with residual connection
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward sub-layer with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ---------------------------------------------------------------------------
# ISACTransformer — Full Model
# ---------------------------------------------------------------------------
class ISACTransformer(nn.Module):
    """
    Transformer-based classifier for 6G-ISAC network state prediction.

    Architecture:
      Input embedding → Positional Encoding
        → N × TransformerEncoderLayer
        → Global average pooling over sequence
        → Classification head (Linear + Softmax)

    Parameters
    ----------
    input_dim   : number of raw input features per time-step
    d_model     : model dimension (embedding size)
    num_heads   : number of attention heads
    num_layers  : number of stacked encoder layers
    d_ff        : feed-forward hidden dimension
    num_classes : number of output classes
    dropout     : dropout probability
    """

    def __init__(
        self,
        input_dim: int = 16,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 128,
        num_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project raw features to d_model dimensional space
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Stack of encoder layers
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, input_dim)
        Returns log-probabilities of shape (batch, num_classes).
        """
        # Embed + positional encoding
        x = self.input_projection(x)          # (batch, seq, d_model)
        x = self.pos_encoding(x)

        # Pass through encoder stack
        for layer in self.encoder_layers:
            x = layer(x)

        # Aggregate sequence via mean pooling → (batch, d_model)
        x = x.mean(dim=1)

        # Classification: raw logits
        logits = self.classifier(x)            # (batch, num_classes)

        # Return log-softmax for NLLLoss compatibility
        return F.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Helper: count trainable parameters
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ISACTransformer()
    print(f"ISACTransformer — trainable parameters: {count_parameters(model):,}")
    dummy = torch.randn(8, 32, 16)          # batch=8, seq=32, features=16
    out = model(dummy)
    print(f"Output shape: {out.shape}")     # (8, 4)
