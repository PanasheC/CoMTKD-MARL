import torch

from models import build_model


def test_cifar_resnet_feature_interface():
    model = build_model("resnet20", num_classes=100).eval()
    with torch.no_grad():
        features, logits = model(torch.randn(2, 3, 32, 32), is_feat=True)
    assert logits.shape == (2, 100)
    assert features[-1].ndim == 2


def test_shufflenet_feature_interface():
    model = build_model("ShuffleV2", num_classes=100).eval()
    with torch.no_grad():
        features, logits = model(torch.randn(2, 3, 32, 32), is_feat=True)
    assert logits.shape == (2, 100)
    assert features[-1].ndim == 2
