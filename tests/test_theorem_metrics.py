import numpy as np
import torch

from helper.theorem_metrics import estimate_multi_teacher_advantage, optimal_simplex_weights


def test_independent_teachers_have_positive_aggregation_margin():
    covariance = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
    estimate = estimate_multi_teacher_advantage(covariance)
    assert estimate.aggregate_risk < estimate.best_single_risk
    expected = np.array([1.0, 0.5, 1 / 3])
    expected = expected / expected.sum()
    assert np.allclose(estimate.optimal_weights, expected, atol=1e-4)


def test_simplex_optimizer_handles_correlated_matrix():
    covariance = np.array([[1.0, 0.9], [0.9, 1.2]])
    weights = optimal_simplex_weights(covariance)
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)
