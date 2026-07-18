"""Complete CoMTKD-MARL policy, coalition selector, critic, and oracle."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .actor import CardinalityActionBatch, CardinalityPolicy, TeacherActor
from .cardinality import normalized_active_weights, select_topk_mask
from .critic import CentralizedCoherenceCritic, CriticOutput
from .synchronization import KnowledgeSynchronizationOracle


@dataclass
class JointPolicyOutput:
    raw_actions: Tensor
    gates: Tensor
    importance: Tensor
    temperatures: Tensor
    channels: Tensor
    actor_log_probs: Tensor
    actor_entropies: Tensor
    cardinality: CardinalityActionBatch
    active_mask: Tensor
    teacher_weights: Tensor
    global_observation: Tensor
    critic: CriticOutput


class CoMTKDMARL(nn.Module):
    """Coherent multi-teacher policy with centralized training."""

    def __init__(
        self,
        teacher_count: int,
        observation_dim: int,
        channel_count: int = 4,
        hidden_dim: int = 128,
        attention_heads: int = 4,
        sync_rounds: int = 3,
        min_temperature: float = 1.0,
        max_temperature: float = 8.0,
    ) -> None:
        super().__init__()
        self.teacher_count = teacher_count
        self.observation_dim = observation_dim
        self.channel_count = channel_count
        self.actors = nn.ModuleList(
            TeacherActor(
                observation_dim,
                hidden_dim=hidden_dim,
                channel_count=channel_count,
                min_temperature=min_temperature,
                max_temperature=max_temperature,
            )
            for _ in range(teacher_count)
        )
        self.global_observation_dim = 2 * observation_dim + 2
        self.cardinality_policy = CardinalityPolicy(
            self.global_observation_dim, teacher_count, hidden_dim=hidden_dim
        )
        self.critic = CentralizedCoherenceCritic(
            observation_dim=observation_dim,
            action_dim=self.actors[0].action_dim,
            teacher_count=teacher_count,
            hidden_dim=hidden_dim,
            attention_heads=attention_heads,
        )
        self.synchronization_oracle = KnowledgeSynchronizationOracle(rounds=sync_rounds)

    def build_global_observation(
        self,
        observations: Tensor,
        previous_coherence: Tensor | None = None,
        remaining_capacity: Tensor | None = None,
    ) -> Tensor:
        batch = observations.shape[0]
        mean = observations.mean(dim=1)
        std = observations.std(dim=1, unbiased=False)
        if previous_coherence is None:
            previous_coherence = observations.new_zeros(batch)
        if remaining_capacity is None:
            remaining_capacity = observations.new_ones(batch)
        return torch.cat(
            [mean, std, previous_coherence.unsqueeze(-1), remaining_capacity.unsqueeze(-1)],
            dim=-1,
        )

    def act(
        self,
        observations: Tensor,
        previous_coherence: Tensor | None = None,
        remaining_capacity: Tensor | None = None,
        deterministic: bool = False,
        forced_cardinality: int | None = None,
    ) -> JointPolicyOutput:
        global_observation = self.build_global_observation(
            observations, previous_coherence, remaining_capacity
        )
        teacher_actions = [
            actor.sample(observations[:, index], deterministic=deterministic)
            for index, actor in enumerate(self.actors)
        ]
        raw = torch.stack([action.raw for action in teacher_actions], dim=1)
        gates = torch.stack([action.gate for action in teacher_actions], dim=1)
        importance = torch.stack([action.importance for action in teacher_actions], dim=1)
        temperatures = torch.stack([action.temperature for action in teacher_actions], dim=1)
        channels = torch.stack([action.channels for action in teacher_actions], dim=1)
        log_probs = torch.stack([action.log_prob for action in teacher_actions], dim=1)
        entropies = torch.stack([action.entropy for action in teacher_actions], dim=1)
        cardinality = self.cardinality_policy.sample(global_observation, deterministic=deterministic)
        if forced_cardinality is not None:
            forced = max(1, min(int(forced_cardinality), self.teacher_count))
            forced_index = torch.full_like(cardinality.index, forced - 1)
            cardinality = self.cardinality_policy.evaluate(global_observation, forced_index)
        selection_scores = gates * torch.softmax(importance, dim=1)
        active_mask = select_topk_mask(selection_scores, cardinality.count)
        weights = normalized_active_weights(gates, importance, active_mask)
        critic = self.critic(
            observations, raw, active_mask, cardinality.probabilities
        )
        return JointPolicyOutput(
            raw_actions=raw,
            gates=gates,
            importance=importance,
            temperatures=temperatures,
            channels=channels,
            actor_log_probs=log_probs,
            actor_entropies=entropies,
            cardinality=cardinality,
            active_mask=active_mask,
            teacher_weights=weights,
            global_observation=global_observation,
            critic=critic,
        )

    def evaluate_actions(
        self,
        observations: Tensor,
        global_observation: Tensor,
        raw_actions: Tensor,
        cardinality_index: Tensor,
    ) -> JointPolicyOutput:
        teacher_actions = [
            actor.evaluate_raw(observations[:, index], raw_actions[:, index])
            for index, actor in enumerate(self.actors)
        ]
        gates = torch.stack([action.gate for action in teacher_actions], dim=1)
        importance = torch.stack([action.importance for action in teacher_actions], dim=1)
        temperatures = torch.stack([action.temperature for action in teacher_actions], dim=1)
        channels = torch.stack([action.channels for action in teacher_actions], dim=1)
        log_probs = torch.stack([action.log_prob for action in teacher_actions], dim=1)
        entropies = torch.stack([action.entropy for action in teacher_actions], dim=1)
        cardinality = self.cardinality_policy.evaluate(global_observation, cardinality_index)
        active_mask = select_topk_mask(
            gates * torch.softmax(importance, dim=1), cardinality.count
        )
        weights = normalized_active_weights(gates, importance, active_mask)
        critic = self.critic(
            observations, raw_actions, active_mask, cardinality.probabilities
        )
        return JointPolicyOutput(
            raw_actions=raw_actions,
            gates=gates,
            importance=importance,
            temperatures=temperatures,
            channels=channels,
            actor_log_probs=log_probs,
            actor_entropies=entropies,
            cardinality=cardinality,
            active_mask=active_mask,
            teacher_weights=weights,
            global_observation=global_observation,
            critic=critic,
        )

    def actor_parameters(self):
        yield from self.actors.parameters()
        yield from self.cardinality_policy.parameters()
