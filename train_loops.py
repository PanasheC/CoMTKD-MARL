"""Training loops for teachers, equal-weight KD, and CoMTKD-MARL."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from distiller_zoo import CoherentDistillationObjective, DistillKL
from helper.metrics import (
    AverageMeter,
    EvaluationMetrics,
    brier_score,
    expected_calibration_error,
    topk_correct,
)
from models.comtkd_marl import (
    CoMTKDMARL,
    MAPPOTrainer,
    RolloutBuffer,
    build_teacher_observations,
)
from models.util import FeatureAdapterBank
from utils import adjust_lr


@dataclass
class CoMTKDTrainState:
    previous_weights: Tensor | None = None
    previous_coherence: Tensor | None = None
    policy_updates: int = 0


@dataclass(frozen=True)
class EpochResult:
    loss: float
    top1: float
    top5: float
    metrics: dict[str, float]


def _teacher_forward(
    teachers: Sequence[nn.Module], inputs: Tensor
) -> tuple[Tensor, list[Tensor]]:
    logits: list[Tensor] = []
    features: list[Tensor] = []
    with torch.no_grad():
        for teacher in teachers:
            teacher_features, teacher_logits = teacher(inputs, is_feat=True)
            logits.append(teacher_logits)
            features.append(teacher_features[-2])
    return torch.stack(logits, dim=1), features


def _student_forward(model: nn.Module, inputs: Tensor) -> tuple[list[Tensor], Tensor]:
    features, logits = model(inputs, is_feat=True)
    return features, logits


def _amp_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def train_baseline_epoch(
    loader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    scaler: torch.amp.GradScaler | None = None,
) -> EpochResult:
    model.train()
    lr = adjust_lr(optimizer, epoch, args)
    loss_meter = AverageMeter("loss")
    top1_total = torch.zeros((), device=device)
    top5_total = torch.zeros((), device=device)
    sample_total = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, getattr(args, "amp", False)):
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        top1, top5 = topk_correct(logits, targets, (1, 5))
        sample_total += targets.shape[0]
        top1_total += top1
        top5_total += top5
        loss_meter.update(loss.item(), targets.shape[0])
        if batch_index % getattr(args, "print_freq", 50) == 0:
            args.logger.info(
                "teacher epoch=%d batch=%d/%d lr=%.5f loss=%.4f top1=%.2f",
                epoch,
                batch_index,
                len(loader),
                lr,
                loss_meter.avg,
                float(top1_total / sample_total * 100.0),
            )
    return EpochResult(
        loss_meter.avg,
        float(top1_total / sample_total * 100.0),
        float(top5_total / sample_total * 100.0),
        {"lr": lr},
    )


def train_avg_epoch(
    loader,
    student: nn.Module,
    teachers: Sequence[nn.Module],
    adapters: FeatureAdapterBank,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    scaler: torch.amp.GradScaler | None = None,
) -> EpochResult:
    student.train()
    adapters.train()
    for teacher in teachers:
        teacher.eval()
    lr = adjust_lr(optimizer, epoch, args)
    kd = DistillKL(getattr(args, "kd_T", 4.0))
    loss_meter = AverageMeter("loss")
    top1_total = torch.zeros((), device=device)
    top5_total = torch.zeros((), device=device)
    sample_total = 0
    for batch_index, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        teacher_logits, teacher_features_raw = _teacher_forward(teachers, inputs)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, getattr(args, "amp", False)):
            student_features, student_logits = _student_forward(student, inputs)
            teacher_features = adapters.teachers(teacher_features_raw)
            student_feature = adapters.student(student_features[-2])
            average_logits = teacher_logits.mean(dim=1)
            average_feature = teacher_features.mean(dim=1)
            supervised = F.cross_entropy(student_logits, targets)
            logit = kd(student_logits, average_logits)
            feature = F.mse_loss(student_feature, average_feature)
            loss = (
                getattr(args, "ce_weight", 1.0) * supervised
                + getattr(args, "kd_weight", 1.0) * logit
                + getattr(args, "feat_weight", 5.0) * feature
            )
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        top1, top5 = topk_correct(student_logits, targets, (1, 5))
        sample_total += targets.shape[0]
        top1_total += top1
        top5_total += top5
        loss_meter.update(loss.item(), targets.shape[0])
        if batch_index % getattr(args, "print_freq", 50) == 0:
            args.logger.info(
                "average-kd epoch=%d batch=%d/%d lr=%.5f loss=%.4f top1=%.2f",
                epoch,
                batch_index,
                len(loader),
                lr,
                loss_meter.avg,
                float(top1_total / sample_total * 100.0),
            )
    return EpochResult(
        loss_meter.avg,
        float(top1_total / sample_total * 100.0),
        float(top5_total / sample_total * 100.0),
        {"lr": lr},
    )


def _normalize_reward(reward: Tensor) -> Tensor:
    return (reward - reward.mean()) / reward.std(unbiased=False).clamp_min(1e-5)


def train_comtkd_epoch(
    loader,
    student: nn.Module,
    teachers: Sequence[nn.Module],
    adapters: FeatureAdapterBank,
    policy: CoMTKDMARL,
    objective: CoherentDistillationObjective,
    optimizer: torch.optim.Optimizer,
    ppo_trainer: MAPPOTrainer,
    rollout_buffer: RolloutBuffer,
    device: torch.device,
    epoch: int,
    args,
    state: CoMTKDTrainState,
    teacher_costs: Tensor | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[EpochResult, CoMTKDTrainState]:
    student.train()
    adapters.train()
    policy.train()
    for teacher in teachers:
        teacher.eval()
    lr = adjust_lr(optimizer, epoch, args)
    meters = {name: AverageMeter(name) for name in (
        "loss", "supervised", "logit", "feature", "coherence", "active", "reward", "rho"
    )}
    top1_total = torch.zeros((), device=device)
    top5_total = torch.zeros((), device=device)
    sample_total = 0

    for batch_index, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        teacher_logits, teacher_features_raw = _teacher_forward(teachers, inputs)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, getattr(args, "amp", False)):
            student_features, student_logits = _student_forward(student, inputs)
            teacher_features = adapters.teachers(teacher_features_raw)
            student_feature = adapters.student(student_features[-2])
            observation_output = build_teacher_observations(
                teacher_logits,
                teacher_features,
                student_logits,
                student_feature,
                targets,
                teacher_costs=teacher_costs,
                base_temperature=getattr(args, "kd_T", 4.0),
            )
            previous_coherence = state.previous_coherence
            if previous_coherence is not None and previous_coherence.shape[0] != inputs.shape[0]:
                previous_coherence = None
            policy_output = policy.act(
                observation_output.observations,
                previous_coherence=previous_coherence,
                deterministic=False,
                forced_cardinality=getattr(args, "forced_cardinality", None),
            )
            teacher_probabilities = torch.softmax(
                teacher_logits / policy_output.temperatures.detach().unsqueeze(-1), dim=-1
            )
            sync = policy.synchronization_oracle(
                teacher_probabilities,
                teacher_features,
                policy_output.teacher_weights.detach(),
                policy_output.active_mask,
            )
            losses = objective(
                student_logits=student_logits,
                student_feature=student_feature,
                targets=targets,
                aggregate_probability=sync.aggregate_probability,
                aggregate_feature=sync.aggregate_feature,
                teacher_weights=policy_output.teacher_weights.detach(),
                channel_allocations=policy_output.channels.detach(),
            )
            total_loss = losses.total
        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(adapters.parameters()),
                getattr(args, "student_grad_clip", 5.0),
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(adapters.parameters()),
                getattr(args, "student_grad_clip", 5.0),
            )
            optimizer.step()

        with torch.no_grad():
            reward_mode = getattr(args, "reward_mode", "loss_delta")
            if reward_mode == "loss_delta":
                was_training = student.training
                student.eval()
                post_logits = student(inputs)
                if was_training:
                    student.train()
                pre_ce = F.cross_entropy(student_logits.detach(), targets, reduction="none")
                post_ce = F.cross_entropy(post_logits, targets, reduction="none")
                progress = (pre_ce - post_ce) / pre_ce.abs().clamp_min(1e-4)
            elif reward_mode == "negative_loss":
                progress = -losses.per_sample_total.detach()
            else:
                raise ValueError(f"Unknown reward_mode: {reward_mode}")

            weighted_novelty = (
                policy_output.teacher_weights * observation_output.novelty
            ).sum(dim=1)
            weighted_redundancy = (
                policy_output.teacher_weights * observation_output.redundancy
            ).sum(dim=1)
            weighted_conflict = (
                policy_output.teacher_weights * observation_output.conflict
            ).sum(dim=1)
            active_cost = policy_output.active_mask.float().mean(dim=1)
            if state.previous_weights is not None and state.previous_weights.shape == policy_output.teacher_weights.shape:
                stability = ((policy_output.teacher_weights - state.previous_weights) ** 2).sum(dim=1)
            else:
                stability = torch.zeros_like(progress)
            team_reward = (
                progress
                + getattr(args, "reward_novelty", 0.05) * weighted_novelty
                - getattr(args, "reward_coherence", 0.10) * sync.final_coherence
                - getattr(args, "reward_redundancy", 0.05) * weighted_redundancy
                - getattr(args, "reward_conflict", 0.05) * weighted_conflict
                - getattr(args, "reward_cost", 0.02) * active_cost
                - getattr(args, "reward_stability", 0.01) * stability
            )
            team_reward = _normalize_reward(team_reward)
            agent_reward = (
                team_reward.unsqueeze(1)
                + getattr(args, "reward_novelty", 0.05) * observation_output.novelty
                + 0.05 * observation_output.gradient_alignment
                - getattr(args, "reward_redundancy", 0.05) * observation_output.redundancy
                - getattr(args, "reward_conflict", 0.05) * observation_output.conflict
                - getattr(args, "reward_cost", 0.02) * policy_output.active_mask.float()
            )
            rollout_buffer.add(
                observations=observation_output.observations,
                global_observations=policy_output.global_observation,
                raw_actions=policy_output.raw_actions,
                cardinality_indices=policy_output.cardinality.index,
                old_actor_log_probs=policy_output.actor_log_probs,
                old_cardinality_log_probs=policy_output.cardinality.log_prob,
                team_rewards=team_reward,
                agent_rewards=agent_reward,
                old_team_values=policy_output.critic.team_value,
                old_agent_values=policy_output.critic.agent_values,
            )
            state.previous_weights = policy_output.teacher_weights.detach()
            state.previous_coherence = sync.final_coherence.detach()

        if len(rollout_buffer) >= getattr(args, "rollout_size", 1024):
            stats = ppo_trainer.update(rollout_buffer, device)
            state.policy_updates += 1
            args.logger.info(
                "policy update=%d actor=%.4f card=%.4f critic=%.4f entropy=%.4f kl=%.5f",
                state.policy_updates,
                stats.actor_loss,
                stats.cardinality_loss,
                stats.critic_loss,
                stats.entropy,
                stats.approximate_kl,
            )

        top1, top5 = topk_correct(student_logits.detach(), targets, (1, 5))
        sample_total += targets.shape[0]
        top1_total += top1
        top5_total += top5
        meters["loss"].update(total_loss.item(), targets.shape[0])
        meters["supervised"].update(losses.supervised.item(), targets.shape[0])
        meters["logit"].update(losses.logit.item(), targets.shape[0])
        meters["feature"].update(losses.feature.item(), targets.shape[0])
        meters["coherence"].update(sync.final_coherence.mean().item(), targets.shape[0])
        meters["active"].update(policy_output.active_mask.float().sum(dim=1).mean().item(), targets.shape[0])
        meters["reward"].update(team_reward.mean().item(), targets.shape[0])
        meters["rho"].update(sync.spectral_rho.mean().item(), targets.shape[0])

        if batch_index % getattr(args, "print_freq", 50) == 0:
            args.logger.info(
                "comtkd epoch=%d batch=%d/%d lr=%.5f loss=%.4f top1=%.2f active=%.2f ci=%.5f rho=%.4f",
                epoch,
                batch_index,
                len(loader),
                lr,
                meters["loss"].avg,
                float(top1_total / sample_total * 100.0),
                meters["active"].avg,
                meters["coherence"].avg,
                meters["rho"].avg,
            )

    if len(rollout_buffer) > 0:
        stats = ppo_trainer.update(rollout_buffer, device)
        state.policy_updates += 1
        args.logger.info(
            "end-epoch policy update=%d actor=%.4f card=%.4f critic=%.4f",
            state.policy_updates,
            stats.actor_loss,
            stats.cardinality_loss,
            stats.critic_loss,
        )

    metrics = {name: meter.avg for name, meter in meters.items()}
    metrics["lr"] = lr
    return (
        EpochResult(
            loss=meters["loss"].avg,
            top1=float(top1_total / sample_total * 100.0),
            top5=float(top5_total / sample_total * 100.0),
            metrics=metrics,
        ),
        state,
    )


@torch.no_grad()
def evaluate(
    loader,
    model: nn.Module,
    device: torch.device,
) -> EvaluationMetrics:
    model.eval()
    loss_meter = AverageMeter("loss")
    top1_total = torch.zeros((), device=device)
    top5_total = torch.zeros((), device=device)
    sample_total = 0
    logits_all: list[Tensor] = []
    targets_all: list[Tensor] = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        top1, top5 = topk_correct(logits, targets, (1, 5))
        top1_total += top1
        top5_total += top5
        sample_total += targets.shape[0]
        loss_meter.update(loss.item(), targets.shape[0])
        logits_all.append(logits.cpu())
        targets_all.append(targets.cpu())
    logits_tensor = torch.cat(logits_all)
    targets_tensor = torch.cat(targets_all)
    return EvaluationMetrics(
        loss=loss_meter.avg,
        top1=float(top1_total / sample_total * 100.0),
        top5=float(top5_total / sample_total * 100.0),
        ece=float(expected_calibration_error(logits_tensor, targets_tensor)),
        brier=float(brier_score(logits_tensor, targets_tensor)),
    )


# Compatibility aliases from the original repository.
def test(epoch, net, device, val_loader, criterion_ce, args, verbose=True):
    del epoch, criterion_ce
    metrics = evaluate(val_loader, net, device)
    if verbose:
        args.logger.info(
            "evaluation loss=%.4f top1=%.2f top5=%.2f ece=%.4f brier=%.4f",
            metrics.loss,
            metrics.top1,
            metrics.top5,
            metrics.ece,
            metrics.brier,
        )
    return metrics.top1
