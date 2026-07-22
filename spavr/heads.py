import torch
import torch.nn.functional as F
from torch import nn


class QuantileHead(nn.Module):
    """Full-trace Transformer reading ``rollout_trace (B, R, P, S+2, D)``.

    Object + predicted proprio tokens from every rollout round are flattened
    into one sequence, tagged with slot-type / within-round time / rollout-round
    embeddings, encoded jointly, then read by three quantile queries that feed a
    median-anchored monotonic readout for ``q10 <= q50 <= q90``.
    """

    def __init__(
        self,
        slot_dim,
        num_objects,
        predicted_size,
        hidden_dim=256,
        taus=(0.1, 0.5, 0.9),
        max_rounds=16,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()
        self.num_objects = num_objects
        self.register_buffer("taus", torch.tensor(taus), persistent=True)
        self.register_buffer(
            "type_ids", torch.tensor([0] * num_objects + [1]), persistent=False
        )

        self.type_emb = nn.Embedding(2, slot_dim)
        self.time_emb = nn.Embedding(predicted_size, slot_dim)
        self.round_emb = nn.Embedding(max_rounds, slot_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            slot_dim, num_heads, dim_feedforward=hidden_dim,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.queries = nn.Parameter(torch.randn(3, slot_dim) * slot_dim ** -0.5)
        self.cross_attn = nn.MultiheadAttention(
            slot_dim, num_heads, dropout=0.0, batch_first=True
        )

        self.trunk = nn.Sequential(
            nn.LayerNorm(slot_dim), nn.Linear(slot_dim, hidden_dim), nn.GELU()
        )
        self.median_head = nn.Linear(hidden_dim, 1)
        self.lower_gap_head = nn.Linear(hidden_dim, 1)
        self.upper_gap_head = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        """Map ``rollout_trace (B, R, P, S+2, D)`` to quantiles ``(B, 3)``."""
        b, r, p = z.shape[:3]
        tokens = z[:, :, :, : self.num_objects + 1]      # object + proprio, (B,R,P,S+1,D)

        rounds = torch.arange(r, device=z.device)
        steps = torch.arange(p, device=z.device)
        tokens = (
            tokens
            + self.type_emb(self.type_ids).view(1, 1, 1, -1, tokens.shape[-1])
            + self.time_emb(steps).view(1, 1, p, 1, -1)
            + self.round_emb(rounds).view(1, r, 1, 1, -1)
        )

        encoded = self.encoder(tokens.reshape(b, -1, tokens.shape[-1]))
        attended, _ = self.cross_attn(self.queries.expand(b, -1, -1), encoded, encoded)

        feats = self.trunk(attended)                     # (B, 3, hidden)
        q50 = self.median_head(feats[:, 1]).squeeze(-1)
        q10 = q50 - F.softplus(self.lower_gap_head(feats[:, 0]).squeeze(-1))
        q90 = q50 + F.softplus(self.upper_gap_head(feats[:, 2]).squeeze(-1))
        return torch.stack([q10, q50, q90], dim=1)       # (B, 3), non-decreasing


def pinball_loss(q_pred, target, taus):
    """Quantile regression loss summed over quantiles, averaged over the batch.

    ``q_pred (B, K)``, ``target (B,)``, ``taus (K,)``.
    """
    error = target.unsqueeze(1) - q_pred                          # (B, K)
    loss = torch.maximum(taus * error, (taus - 1.0) * error)
    return loss.sum(dim=1).mean()
