import torch
import torch.nn.functional as F
from torch import nn


class QuantileHead(nn.Module):
    def __init__(self, slot_dim, num_objects, hidden_dim=256, taus=(0.1, 0.5, 0.9)):
        super().__init__()
        self.num_objects = num_objects
        self.register_buffer("taus", torch.tensor(taus), persistent=True)
        self.query = nn.Parameter(torch.randn(slot_dim) * slot_dim ** -0.5)
        self.mlp = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(taus)),
        )

    def forward(self, z):
        """``z (B, S+2, D)`` -> monotone quantiles ``(B, K)``."""
        objects = z[:, :self.num_objects]                          # (B, S, D)
        scores = objects @ self.query * self.query.shape[0] ** -0.5
        weights = F.softmax(scores, dim=1).unsqueeze(-1)           # (B, S, 1)
        pooled = (weights * objects).sum(dim=1)                    # (B, D)

        raw = self.mlp(pooled)                                     # (B, K)
        increments = torch.cat([raw[:, :1], F.softplus(raw[:, 1:])], dim=1)
        return increments.cumsum(dim=1)                            # non-decreasing


def pinball_loss(q_pred, target, taus):
    """Quantile regression loss summed over quantiles, averaged over the batch.

    ``q_pred (B, K)``, ``target (B,)``, ``taus (K,)``.
    """
    error = target.unsqueeze(1) - q_pred                          # (B, K)
    loss = torch.maximum(taus * error, (taus - 1.0) * error)
    return loss.sum(dim=1).mean()
