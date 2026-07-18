import torch

from models.comtkd_marl.synchronization import (
    KnowledgeSynchronizationOracle,
    consensus_disagreement,
    metropolis_mixing,
    pairwise_js_divergence,
)


def test_mixing_is_doubly_stochastic_on_active_graph():
    similarity = torch.tensor(
        [[[1.0, 0.8, 0.5], [0.8, 1.0, 0.4], [0.5, 0.4, 1.0]]]
    )
    active = torch.ones(1, 3, dtype=torch.bool)
    mixing = metropolis_mixing(similarity, active)
    assert torch.allclose(mixing.sum(dim=-1), torch.ones(1, 3), atol=1e-6)
    assert torch.allclose(mixing.sum(dim=-2), torch.ones(1, 3), atol=1e-6)
    assert torch.allclose(mixing, mixing.transpose(1, 2), atol=1e-6)


def test_consensus_contracts_disagreement():
    torch.manual_seed(3)
    probabilities = torch.softmax(torch.randn(4, 3, 10), dim=-1)
    features = torch.randn(4, 3, 8)
    active = torch.ones(4, 3, dtype=torch.bool)
    weights = torch.full((4, 3), 1 / 3)
    initial = consensus_disagreement(features, active)
    oracle = KnowledgeSynchronizationOracle(rounds=5)
    output = oracle(probabilities, features, weights, active)
    final = consensus_disagreement(output.synchronized_features, active)
    assert torch.all(final <= initial + 1e-6)
    assert torch.all(output.final_coherence <= output.initial_coherence + 1e-6)
    assert torch.allclose(output.aggregate_probability.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_pairwise_js_is_symmetric_and_nonnegative():
    p = torch.softmax(torch.randn(2, 4, 7), dim=-1)
    js = pairwise_js_divergence(p)
    assert torch.all(js >= 0)
    assert torch.allclose(js, js.transpose(1, 2), atol=1e-6)
