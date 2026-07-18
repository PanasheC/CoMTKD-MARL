"""Graph synchronization for teacher probabilities and knowledge features."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _safe_probabilities(probabilities: Tensor, eps: float = 1e-8) -> Tensor:
    probabilities = probabilities.clamp_min(eps)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(eps)


def pairwise_js_divergence(probabilities: Tensor) -> Tensor:
    """Pairwise Jensen-Shannon divergence for [B, M, C] probabilities."""
    p = _safe_probabilities(probabilities)
    p_i = p.unsqueeze(2)
    p_j = p.unsqueeze(1)
    midpoint = 0.5 * (p_i + p_j)
    kl_i = (p_i * (p_i.log() - midpoint.log())).sum(dim=-1)
    kl_j = (p_j * (p_j.log() - midpoint.log())).sum(dim=-1)
    return (0.5 * (kl_i + kl_j)).clamp_min(0.0)


def consensus_disagreement(knowledge: Tensor, active_mask: Tensor) -> Tensor:
    """Mean squared distance from the active coalition mean."""
    mask = active_mask.to(knowledge.dtype).unsqueeze(-1)
    mean = (knowledge * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    squared = ((knowledge - mean) ** 2).sum(dim=-1) * active_mask.to(knowledge.dtype)
    return squared.sum(dim=1) / active_mask.sum(dim=1).clamp_min(1)


def probability_coherence_index(probabilities: Tensor, active_mask: Tensor) -> Tensor:
    weights = active_mask.to(probabilities.dtype)
    mean = (probabilities * weights.unsqueeze(-1)).sum(dim=1)
    mean = mean / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    p = _safe_probabilities(probabilities)
    q = _safe_probabilities(mean).unsqueeze(1)
    midpoint = 0.5 * (p + q)
    js = 0.5 * (
        (p * (p.log() - midpoint.log())).sum(dim=-1)
        + (q * (q.log() - midpoint.log())).sum(dim=-1)
    )
    js = js.clamp_min(0.0)
    return (js * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def metropolis_mixing(similarity: Tensor, active_mask: Tensor) -> Tensor:
    """Build a symmetric, row-stochastic, and therefore doubly stochastic matrix."""
    if similarity.ndim != 3:
        raise ValueError("similarity must have shape [batch, teachers, teachers]")
    batch, teachers, _ = similarity.shape
    active_pair = active_mask.unsqueeze(1) & active_mask.unsqueeze(2)
    eye = torch.eye(teachers, device=similarity.device, dtype=torch.bool).unsqueeze(0)
    adjacency = similarity * active_pair.to(similarity.dtype) * (~eye).to(similarity.dtype)
    degree = adjacency.sum(dim=-1)
    denominator = 1.0 + torch.maximum(degree.unsqueeze(2), degree.unsqueeze(1))
    off_diagonal = adjacency / denominator.clamp_min(1e-8)
    diagonal = 1.0 - off_diagonal.sum(dim=-1)
    mixing = off_diagonal + torch.diag_embed(diagonal)
    inactive = ~active_mask
    mixing = mixing.masked_fill(inactive.unsqueeze(2), 0.0)
    mixing = mixing.masked_fill(inactive.unsqueeze(1), 0.0)
    mixing = mixing + torch.diag_embed(inactive.to(mixing.dtype))
    return mixing


def spectral_contraction(mixing: Tensor, active_mask: Tensor) -> Tensor:
    """Compute ||W - J||_2 on each active teacher subgraph."""
    values: list[Tensor] = []
    for batch_idx in range(mixing.shape[0]):
        indices = torch.nonzero(active_mask[batch_idx], as_tuple=False).flatten()
        if indices.numel() <= 1:
            values.append(mixing.new_zeros(()))
            continue
        w = mixing[batch_idx].index_select(0, indices).index_select(1, indices)
        count = indices.numel()
        j = torch.full_like(w, 1.0 / float(count))
        values.append(torch.linalg.matrix_norm(w - j, ord=2))
    return torch.stack(values)


@dataclass
class SynchronizationOutput:
    synchronized_probabilities: Tensor
    synchronized_features: Tensor
    aggregate_probability: Tensor
    aggregate_feature: Tensor
    mixing_matrix: Tensor
    initial_coherence: Tensor
    final_coherence: Tensor
    feature_disagreement: Tensor
    spectral_rho: Tensor
    pairwise_js: Tensor


class KnowledgeSynchronizationOracle(nn.Module):
    """Training-time coherence oracle based on graph consensus."""

    def __init__(
        self,
        rounds: int = 3,
        similarity_temperature: float = 0.25,
        perturbation_std: float = 0.0,
    ) -> None:
        super().__init__()
        if rounds < 0:
            raise ValueError("rounds must be nonnegative")
        self.rounds = rounds
        self.similarity_temperature = similarity_temperature
        self.perturbation_std = perturbation_std

    def forward(
        self,
        probabilities: Tensor,
        features: Tensor,
        teacher_weights: Tensor,
        active_mask: Tensor,
    ) -> SynchronizationOutput:
        probabilities = _safe_probabilities(probabilities)
        pairwise_js = pairwise_js_divergence(probabilities)
        similarity = torch.exp(-pairwise_js / max(self.similarity_temperature, 1e-6))
        mixing = metropolis_mixing(similarity, active_mask)
        initial = probability_coherence_index(probabilities, active_mask)
        synced_p = probabilities
        synced_f = features
        for _ in range(self.rounds):
            synced_p = torch.bmm(mixing, synced_p)
            synced_f = torch.bmm(mixing, synced_f)
            if self.training and self.perturbation_std > 0.0:
                synced_f = synced_f + torch.randn_like(synced_f) * self.perturbation_std
                noisy_p = synced_p + torch.randn_like(synced_p) * self.perturbation_std
                synced_p = _safe_probabilities(noisy_p.clamp_min(1e-8))
        final = probability_coherence_index(synced_p, active_mask)
        normalized_weights = teacher_weights * active_mask.to(teacher_weights.dtype)
        normalized_weights = normalized_weights / normalized_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        aggregate_probability = (synced_p * normalized_weights.unsqueeze(-1)).sum(dim=1)
        aggregate_probability = _safe_probabilities(aggregate_probability)
        aggregate_feature = (synced_f * normalized_weights.unsqueeze(-1)).sum(dim=1)
        return SynchronizationOutput(
            synchronized_probabilities=synced_p,
            synchronized_features=synced_f,
            aggregate_probability=aggregate_probability,
            aggregate_feature=aggregate_feature,
            mixing_matrix=mixing,
            initial_coherence=initial,
            final_coherence=final,
            feature_disagreement=consensus_disagreement(synced_f, active_mask),
            spectral_rho=spectral_contraction(mixing, active_mask),
            pairwise_js=pairwise_js,
        )
