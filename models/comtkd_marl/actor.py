"""Teacher actors and the teacher-cardinality policy."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal


@dataclass
class TeacherActionBatch:
    raw: Tensor
    gate: Tensor
    importance: Tensor
    temperature: Tensor
    channels: Tensor
    log_prob: Tensor
    entropy: Tensor


class TeacherActor(nn.Module):
    """Continuous teacher policy used by the cooperative MAPPO learner.

    The policy samples an unconstrained action and transforms it into a soft
    participation gate, an importance score, a temperature, and a simplex over
    knowledge channels. PPO probabilities are evaluated in the unconstrained
    action space, which avoids unstable Jacobian corrections.
    """

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 128,
        channel_count: int = 4,
        min_temperature: float = 1.0,
        max_temperature: float = 8.0,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.channel_count = channel_count
        self.action_dim = 3 + channel_count
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -0.7))

    def distribution(self, observation: Tensor) -> Normal:
        encoded = self.encoder(observation)
        mean = self.mean_head(encoded)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    def transform(self, raw_action: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        gate = torch.sigmoid(raw_action[..., 0])
        importance = raw_action[..., 1]
        temperature_fraction = torch.sigmoid(raw_action[..., 2])
        temperature = self.min_temperature + (
            self.max_temperature - self.min_temperature
        ) * temperature_fraction
        channels = torch.softmax(raw_action[..., 3:], dim=-1)
        return gate, importance, temperature, channels

    def sample(self, observation: Tensor, deterministic: bool = False) -> TeacherActionBatch:
        dist = self.distribution(observation)
        raw = dist.mean if deterministic else dist.rsample()
        gate, importance, temperature, channels = self.transform(raw)
        return TeacherActionBatch(
            raw=raw,
            gate=gate,
            importance=importance,
            temperature=temperature,
            channels=channels,
            log_prob=dist.log_prob(raw).sum(dim=-1),
            entropy=dist.entropy().sum(dim=-1),
        )

    def evaluate_raw(self, observation: Tensor, raw_action: Tensor) -> TeacherActionBatch:
        dist = self.distribution(observation)
        gate, importance, temperature, channels = self.transform(raw_action)
        return TeacherActionBatch(
            raw=raw_action,
            gate=gate,
            importance=importance,
            temperature=temperature,
            channels=channels,
            log_prob=dist.log_prob(raw_action).sum(dim=-1),
            entropy=dist.entropy().sum(dim=-1),
        )


@dataclass
class CardinalityActionBatch:
    index: Tensor
    count: Tensor
    log_prob: Tensor
    entropy: Tensor
    probabilities: Tensor


class CardinalityPolicy(nn.Module):
    """Categorical policy over the active teacher count from 1 through M."""

    def __init__(self, global_observation_dim: int, teacher_count: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if teacher_count < 1:
            raise ValueError("teacher_count must be positive")
        self.teacher_count = teacher_count
        self.network = nn.Sequential(
            nn.Linear(global_observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, teacher_count),
        )

    def distribution(self, global_observation: Tensor) -> Categorical:
        return Categorical(logits=self.network(global_observation))

    def sample(self, global_observation: Tensor, deterministic: bool = False) -> CardinalityActionBatch:
        dist = self.distribution(global_observation)
        index = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return CardinalityActionBatch(
            index=index,
            count=index + 1,
            log_prob=dist.log_prob(index),
            entropy=dist.entropy(),
            probabilities=dist.probs,
        )

    def evaluate(self, global_observation: Tensor, index: Tensor) -> CardinalityActionBatch:
        dist = self.distribution(global_observation)
        return CardinalityActionBatch(
            index=index,
            count=index + 1,
            log_prob=dist.log_prob(index),
            entropy=dist.entropy(),
            probabilities=dist.probs,
        )
