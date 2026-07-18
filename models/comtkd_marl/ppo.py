"""Contextual MAPPO optimization for teacher agents and cardinality policy."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .controller import CoMTKDMARL


@dataclass
class RolloutBatch:
    observations: Tensor
    global_observations: Tensor
    raw_actions: Tensor
    cardinality_indices: Tensor
    old_actor_log_probs: Tensor
    old_cardinality_log_probs: Tensor
    team_rewards: Tensor
    agent_rewards: Tensor
    old_team_values: Tensor
    old_agent_values: Tensor


@dataclass
class RolloutBuffer:
    observations: list[Tensor] = field(default_factory=list)
    global_observations: list[Tensor] = field(default_factory=list)
    raw_actions: list[Tensor] = field(default_factory=list)
    cardinality_indices: list[Tensor] = field(default_factory=list)
    old_actor_log_probs: list[Tensor] = field(default_factory=list)
    old_cardinality_log_probs: list[Tensor] = field(default_factory=list)
    team_rewards: list[Tensor] = field(default_factory=list)
    agent_rewards: list[Tensor] = field(default_factory=list)
    old_team_values: list[Tensor] = field(default_factory=list)
    old_agent_values: list[Tensor] = field(default_factory=list)

    def __len__(self) -> int:
        return sum(tensor.shape[0] for tensor in self.team_rewards)

    def add(
        self,
        observations: Tensor,
        global_observations: Tensor,
        raw_actions: Tensor,
        cardinality_indices: Tensor,
        old_actor_log_probs: Tensor,
        old_cardinality_log_probs: Tensor,
        team_rewards: Tensor,
        agent_rewards: Tensor,
        old_team_values: Tensor,
        old_agent_values: Tensor,
    ) -> None:
        values = (
            observations,
            global_observations,
            raw_actions,
            cardinality_indices,
            old_actor_log_probs,
            old_cardinality_log_probs,
            team_rewards,
            agent_rewards,
            old_team_values,
            old_agent_values,
        )
        detached = [value.detach().cpu() for value in values]
        (
            observations,
            global_observations,
            raw_actions,
            cardinality_indices,
            old_actor_log_probs,
            old_cardinality_log_probs,
            team_rewards,
            agent_rewards,
            old_team_values,
            old_agent_values,
        ) = detached
        self.observations.append(observations)
        self.global_observations.append(global_observations)
        self.raw_actions.append(raw_actions)
        self.cardinality_indices.append(cardinality_indices)
        self.old_actor_log_probs.append(old_actor_log_probs)
        self.old_cardinality_log_probs.append(old_cardinality_log_probs)
        self.team_rewards.append(team_rewards)
        self.agent_rewards.append(agent_rewards)
        self.old_team_values.append(old_team_values)
        self.old_agent_values.append(old_agent_values)

    def as_batch(self, device: torch.device) -> RolloutBatch:
        if not self.team_rewards:
            raise RuntimeError("Rollout buffer is empty")
        def cat(items: list[Tensor]) -> Tensor:
            return torch.cat(items, dim=0).to(device)
        return RolloutBatch(
            observations=cat(self.observations),
            global_observations=cat(self.global_observations),
            raw_actions=cat(self.raw_actions),
            cardinality_indices=cat(self.cardinality_indices),
            old_actor_log_probs=cat(self.old_actor_log_probs),
            old_cardinality_log_probs=cat(self.old_cardinality_log_probs),
            team_rewards=cat(self.team_rewards),
            agent_rewards=cat(self.agent_rewards),
            old_team_values=cat(self.old_team_values),
            old_agent_values=cat(self.old_agent_values),
        )

    def clear(self) -> None:
        for value in self.__dict__.values():
            value.clear()


@dataclass(frozen=True)
class PPOStats:
    actor_loss: float
    cardinality_loss: float
    critic_loss: float
    entropy: float
    approximate_kl: float


class MAPPOTrainer:
    def __init__(
        self,
        policy: CoMTKDMARL,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        clip_ratio: float = 0.2,
        entropy_coefficient: float = 0.01,
        value_coefficient: float = 0.5,
        max_grad_norm: float = 1.0,
        epochs: int = 4,
        minibatch_size: int = 256,
    ) -> None:
        self.policy = policy
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.clip_ratio = clip_ratio
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.minibatch_size = minibatch_size

    @staticmethod
    def _normalize(advantage: Tensor) -> Tensor:
        return (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)

    def update(self, buffer: RolloutBuffer, device: torch.device) -> PPOStats:
        batch = buffer.as_batch(device)
        team_advantage = self._normalize(batch.team_rewards - batch.old_team_values)
        agent_advantage = self._normalize(batch.agent_rewards - batch.old_agent_values)
        team_returns = batch.team_rewards
        agent_returns = batch.agent_rewards
        sample_count = batch.team_rewards.shape[0]
        totals = torch.zeros(5, device=device)
        update_count = 0

        for _ in range(self.epochs):
            permutation = torch.randperm(sample_count, device=device)
            for start in range(0, sample_count, self.minibatch_size):
                index = permutation[start : start + self.minibatch_size]
                output = self.policy.evaluate_actions(
                    batch.observations[index],
                    batch.global_observations[index],
                    batch.raw_actions[index],
                    batch.cardinality_indices[index],
                )
                actor_ratio = torch.exp(
                    output.actor_log_probs - batch.old_actor_log_probs[index]
                )
                agent_adv = agent_advantage[index]
                unclipped = actor_ratio * agent_adv
                clipped = actor_ratio.clamp(
                    1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                ) * agent_adv
                actor_loss = -torch.minimum(unclipped, clipped).mean()

                cardinality_ratio = torch.exp(
                    output.cardinality.log_prob - batch.old_cardinality_log_probs[index]
                )
                team_adv = team_advantage[index]
                cardinality_loss = -torch.minimum(
                    cardinality_ratio * team_adv,
                    cardinality_ratio.clamp(
                        1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                    ) * team_adv,
                ).mean()
                entropy = output.actor_entropies.mean() + output.cardinality.entropy.mean()
                total_actor_loss = actor_loss + cardinality_loss - self.entropy_coefficient * entropy
                self.actor_optimizer.zero_grad(set_to_none=True)
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.actor_parameters()), self.max_grad_norm
                )
                self.actor_optimizer.step()

                # Recompute critic values after the actor update. Raw actions remain fixed.
                with torch.no_grad():
                    refreshed = self.policy.evaluate_actions(
                        batch.observations[index],
                        batch.global_observations[index],
                        batch.raw_actions[index],
                        batch.cardinality_indices[index],
                    )
                    active_mask = refreshed.active_mask
                    card_probs = refreshed.cardinality.probabilities
                critic_output = self.policy.critic(
                    batch.observations[index],
                    batch.raw_actions[index],
                    active_mask,
                    card_probs,
                )
                critic_loss = (
                    torch.mean((critic_output.team_value - team_returns[index]) ** 2)
                    + torch.mean((critic_output.agent_values - agent_returns[index]) ** 2)
                )
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.value_coefficient * critic_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(), self.max_grad_norm
                )
                self.critic_optimizer.step()

                approximate_kl = (
                    batch.old_actor_log_probs[index] - output.actor_log_probs
                ).mean().abs()
                totals += torch.stack(
                    [
                        actor_loss.detach(),
                        cardinality_loss.detach(),
                        critic_loss.detach(),
                        entropy.detach(),
                        approximate_kl.detach(),
                    ]
                )
                update_count += 1

        buffer.clear()
        means = (totals / max(update_count, 1)).tolist()
        return PPOStats(*[float(value) for value in means])
