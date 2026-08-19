"""Unit tests for model_builder.ResNet / ResBlock.

Run with:
    pytest tests/ -v
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_builder import ResBlock, ResNet  # noqa: E402


def test_resnet34_output_shape():
    """ResNet-34 config ([3,4,6,3]) must map a 224x224 RGB batch to (N, num_classes)."""
    model = ResNet(in_channels=3, out_channels=101, blocks=[3, 4, 6, 3])
    x = torch.randn(2, 3, 224, 224)

    out = model(x)

    assert out.shape == (2, 101)


def test_resnet_arbitrary_input_size():
    """Global average pooling means the model shouldn't be hardcoded to 224x224."""
    model = ResNet(in_channels=3, out_channels=10, blocks=[2, 2, 2, 2])
    x = torch.randn(1, 3, 96, 96)

    out = model(x)

    assert out.shape == (1, 10)


def test_resnet_param_count_is_resnet34_scale():
    """Sanity check: ResNet-34 should be ~21-22M parameters (not, say, 2M or 200M)."""
    model = ResNet(in_channels=3, out_channels=1000, blocks=[3, 4, 6, 3])
    num_params = sum(p.numel() for p in model.parameters())

    assert 20_000_000 < num_params < 23_000_000, f"got {num_params:,} params"


def test_resblock_identity_skip_when_shapes_match():
    """No projection conv should be added when in/out channels and stride both allow identity."""
    block = ResBlock(in_channels=64, out_channels=64, stride=1)

    assert isinstance(block.skip, torch.nn.Sequential) and len(block.skip) == 0


def test_resblock_projection_skip_on_downsample():
    """A projection (1x1 conv + BN) must be added whenever stride != 1 or channels change."""
    block = ResBlock(in_channels=64, out_channels=128, stride=2)

    assert len(block.skip) == 2  # Conv2d + BatchNorm2d

    x = torch.randn(1, 64, 56, 56)
    out = block(x)
    assert out.shape == (1, 128, 28, 28)


def test_gradients_flow_through_full_model():
    """Guards against silently-frozen layers (e.g. a stray requires_grad=False)."""
    model = ResNet(in_channels=3, out_channels=10, blocks=[1, 1, 1, 1])
    x = torch.randn(1, 3, 64, 64)

    loss = model(x).sum()
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.requires_grad]
    assert all(g == g for g in grad_norms)  # no NaNs
    assert any(g > 0 for g in grad_norms)   # not all-zero
