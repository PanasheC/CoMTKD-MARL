"""Centralized coherence critic for cooperative teacher agents."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class CriticOutput:
    team_value: Tensor
    agent_values: Tensor


class CentralizedCoherenceCritic(nn.Module):
    """Permutation-aware critic over the full teacher coalition.

    Teacher tokens combine local observations, raw actions, and participation.
    Self-attention captures interactions among teachers. The critic returns a
    team value and one counterfactual baseline value per teacher.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        teacher_count: int,
        hidden_dim: int = 128,
        attention_heads: int = 4,
        cardinality_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.teacher_count = teacher_count
        token_input = observation_dim + action_dim + 1
        self.token_encoder = nn.Sequential(
            nn.Linear(token_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, batch_first=True, dropout=0.0
        )
        cardinality_dim = teacher_count if cardinality_dim is None else cardinality_dim
        self.team_head = nn.Sequential(
            nn.Linear(hidden_dim + cardinality_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.agent_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        observations: Tensor,
        raw_actions: Tensor,
        active_mask: Tensor,
        cardinality_probabilities: Tensor,
    ) -> CriticOutput:
        if observations.ndim != 3:
            raise ValueError("observations must have shape [batch, teachers, features]")
        token = torch.cat(
            [observations, raw_actions, active_mask.to(observations.dtype).unsqueeze(-1)],
            dim=-1,
        )
        token = self.token_encoder(token)
        attended, _ = self.attention(token, token, token, need_weights=False)
        token = token + attended
        mask = active_mask.to(token.dtype).unsqueeze(-1)
        pooled = (token * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        team_value = self.team_head(torch.cat([pooled, cardinality_probabilities], dim=-1)).squeeze(-1)
        pooled_agents = pooled.unsqueeze(1).expand(-1, self.teacher_count, -1)
        agent_values = self.agent_head(torch.cat([token, pooled_agents], dim=-1)).squeeze(-1)
        return CriticOutput(team_value=team_value, agent_values=agent_values)
