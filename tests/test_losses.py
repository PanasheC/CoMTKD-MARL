import torch

from distiller_zoo import CoherentDistillationObjective


def test_coherent_objective_is_finite_and_differentiable():
    torch.manual_seed(1)
    batch, classes, teachers, feature_dim = 5, 10, 3, 8
    student_logits = torch.randn(batch, classes, requires_grad=True)
    student_feature = torch.randn(batch, feature_dim, requires_grad=True)
    targets = torch.randint(classes, (batch,))
    aggregate_probability = torch.softmax(torch.randn(batch, classes), dim=-1)
    aggregate_feature = torch.randn(batch, feature_dim)
    teacher_weights = torch.softmax(torch.randn(batch, teachers), dim=-1)
    channels = torch.softmax(torch.randn(batch, teachers, 4), dim=-1)
    objective = CoherentDistillationObjective()
    output = objective(
        student_logits,
        student_feature,
        targets,
        aggregate_probability,
        aggregate_feature,
        teacher_weights,
        channels,
    )
    assert torch.isfinite(output.total)
    output.total.backward()
    assert student_logits.grad is not None
    assert student_feature.grad is not None
