import torch

from models.comtkd_marl.cardinality import (
    estimate_optimal_cardinality,
    normalized_active_weights,
    select_topk_mask,
)


def test_select_topk_variable_counts():
    scores = torch.tensor([[1.0, 4.0, 2.0], [3.0, 2.0, 1.0]])
    counts = torch.tensor([1, 2])
    mask = select_topk_mask(scores, counts)
    assert mask.tolist() == [[False, True, False], [True, True, False]]


def test_active_weights_form_simplex():
    gates = torch.tensor([[0.8, 0.2, 0.9]])
    importance = torch.tensor([[0.0, 2.0, -1.0]])
    active = torch.tensor([[True, True, False]])
    weights = normalized_active_weights(gates, importance, active)
    assert torch.allclose(weights.sum(dim=1), torch.ones(1))
    assert weights[0, 2] == 0


def test_saturation_stops_after_nonpositive_increment():
    estimate = estimate_optimal_cardinality(
        conditional_gains=[0.5, 0.2, 0.05, 0.01],
        incremental_costs=[0.05, 0.08, 0.08, 0.08],
        student_capacity=1.0,
    )
    assert estimate.optimal_cardinality == 2
    assert estimate.increments[2] <= 0
