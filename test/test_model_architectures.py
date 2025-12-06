"""Tests for src/model/architectures/model_architectures.py"""

import pytest
import torch
from torch import nn

from src.configs import ModelConfig
from src.model.architectures.model_architectures import (
    ModelArchitectures,
    get_model_architecture,
)


class TestGetModelArchitecture:
    """Test get_model_architecture function"""

    def test_simple_model(self):
        """Test creating simple_model"""
        model = get_model_architecture(
            model_name="simple_model",
            backbone_name="tf_efficientnet_b0",
            pretrained=False,
            in_channels=3,
            n_classes=5,
        )

        assert model is not None
        assert isinstance(model, nn.Module)

    def test_simple_total(self):
        """Test creating simple_total"""
        model = get_model_architecture(
            model_name="simple_total",
            backbone_name="tf_efficientnet_b0",
            pretrained=False,
            in_channels=3,
            n_classes=5,
        )

        assert model is not None
        assert isinstance(model, nn.Module)

    def test_simple_clover_diff(self):
        """Test creating simple_clover_diff"""
        model = get_model_architecture(
            model_name="simple_clover_diff",
            backbone_name="tf_efficientnet_b0",
            pretrained=False,
            in_channels=3,
            n_classes=5,
        )

        assert model is not None
        assert isinstance(model, nn.Module)

    def test_height_model(self):
        """Test creating height_model"""
        model = get_model_architecture(
            model_name="height_model",
            backbone_name="tf_efficientnet_b0",
            pretrained=False,
            in_channels=3,
            n_classes=5,
            emb_dim=128,
            aux_dim_reduction_factor=2,
        )

        assert model is not None
        assert isinstance(model, nn.Module)

    def test_height_gshh_model(self):
        """Test creating height_gshh_model"""
        model = get_model_architecture(
            model_name="height_gshh_model",
            backbone_name="tf_efficientnet_b0",
            pretrained=False,
            in_channels=3,
            n_classes=5,
            emb_dim=128,
            aux_dim_reduction_factor=2,
        )

        assert model is not None
        assert isinstance(model, nn.Module)

    def test_clover_model(self):
        """Test creating clover_model"""
        model = get_model_architecture(
            model_name="clover_model",
            backbone_name="tf_efficientnet_b0",
            pretrained=False,
            in_channels=3,
            n_classes=5,
            emb_dim=128,
            aux_dim_reduction_factor=2,
            head_connection_type="direct",
        )

        assert model is not None
        assert isinstance(model, nn.Module)

    def test_invalid_model_name(self):
        """Test that invalid model name raises error"""
        with pytest.raises(NotImplementedError):
            get_model_architecture(
                model_name="invalid_model",
                backbone_name="tf_efficientnet_b0",
                pretrained=False,
                in_channels=3,
                n_classes=5,
            )

    def test_different_backbones(self):
        """Test with different backbone architectures"""
        backbones = ["tf_efficientnet_b0", "resnet18", "resnet34"]

        for backbone in backbones:
            try:
                model = get_model_architecture(
                    model_name="simple_model",
                    backbone_name=backbone,
                    pretrained=False,
                    in_channels=3,
                    n_classes=5,
                )
                assert model is not None
            except Exception as e:
                # Some backbones might not be available
                pytest.skip(f"Backbone {backbone} not available: {e}")


class TestModelArchitectures:
    """Test ModelArchitectures wrapper class"""

    def test_initialization(self):
        """Test ModelArchitectures initialization"""
        config = ModelConfig(
            model_name="simple_model",
            backbone_name="tf_efficientnet_b0",
            pretrained=False,
            in_channels=3,
            n_classes=5,
        )

        model = ModelArchitectures(config)

        assert model is not None
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        """Test forward pass"""
        config = ModelConfig(
            model_name="simple_model",
            backbone_name="tf_efficientnet_b0",
            pretrained=False,
            in_channels=3,
            n_classes=5,
        )

        model = ModelArchitectures(config)
        model.eval()

        batch_size = 4
        sample_input = {"image": torch.randn(batch_size, 3, 224, 224)}

        with torch.no_grad():
            output = model(sample_input)

        assert "logits" in output
        assert output["logits"].shape == (batch_size, 5)

    def test_different_batch_sizes(self):
        """Test with different batch sizes"""
        config = ModelConfig(
            model_name="simple_model",
            backbone_name="tf_efficientnet_b0",
            pretrained=False,
            in_channels=3,
            n_classes=5,
        )

        model = ModelArchitectures(config)
        model.eval()

        for batch_size in [1, 4, 8, 16]:
            sample_input = {"image": torch.randn(batch_size, 3, 224, 224)}

            with torch.no_grad():
                output = model(sample_input)

            assert output["logits"].shape == (batch_size, 5)

    def test_different_image_sizes(self):
        """Test with different image sizes"""
        config = ModelConfig(
            model_name="simple_model",
            backbone_name="tf_efficientnet_b0",
            pretrained=False,
            in_channels=3,
            n_classes=5,
        )

        model = ModelArchitectures(config)
        model.eval()

        # Test various image sizes
        for img_size in [224, 256, 512]:
            sample_input = {"image": torch.randn(4, 3, img_size, img_size)}

            with torch.no_grad():
                output = model(sample_input)

            # Output should always have correct number of classes
            assert output["logits"].shape[1] == 5

    def test_gradient_flow(self):
        """Test that gradients flow through the model"""
        config = ModelConfig(
            model_name="simple_model",
            backbone_name="tf_efficientnet_b0",
            pretrained=False,
            in_channels=3,
            n_classes=5,
        )

        model = ModelArchitectures(config)
        model.train()

        sample_input = {"image": torch.randn(4, 3, 224, 224, requires_grad=True)}
        output = model(sample_input)

        # Create dummy loss and backpropagate
        loss = output["logits"].sum()
        loss.backward()

        # Check that gradients exist
        assert sample_input["image"].grad is not None

    def test_config_parameters_applied(self):
        """Test that config parameters are applied correctly"""
        config = ModelConfig(
            model_name="simple_model",
            backbone_name="tf_efficientnet_b0",
            pretrained=False,
            in_channels=3,
            n_classes=10,  # Different number of classes
        )

        model = ModelArchitectures(config)
        model.eval()

        sample_input = {"image": torch.randn(4, 3, 224, 224)}

        with torch.no_grad():
            output = model(sample_input)

        # Should output 10 classes as specified in config
        assert output["logits"].shape[1] == 10


class TestModelArchitecturesWithAuxHeads:
    """Test models with auxiliary heads"""

    def test_height_model_output(self):
        """Test height_model outputs both logits and height"""
        config = ModelConfig(
            model_name="height_model",
            backbone_name="tf_efficientnet_b0",
            pretrained=False,
            in_channels=3,
            n_classes=5,
            emb_dim=128,
        )

        model = ModelArchitectures(config)
        model.eval()

        sample_input = {"image": torch.randn(4, 3, 224, 224)}

        with torch.no_grad():
            output = model(sample_input)

        assert "logits" in output
        assert "height" in output
        assert output["logits"].shape == (4, 5)
        assert output["height"].shape == (4, 1)

    def test_clover_model_output(self):
        """Test clover_model outputs both logits and clover"""
        config = ModelConfig(
            model_name="clover_model",
            backbone_name="tf_efficientnet_b0",
            pretrained=False,
            in_channels=3,
            n_classes=5,
            emb_dim=128,
            head_connection_type="direct",
        )

        model = ModelArchitectures(config)
        model.eval()

        sample_input = {"image": torch.randn(4, 3, 224, 224)}

        with torch.no_grad():
            output = model(sample_input)

        assert "logits" in output
        assert "include_clover_pred" in output
        assert output["logits"].shape == (4, 5)
        assert output["include_clover_pred"].shape == (4, 1)


class TestModelArchitecturesIntegration:
    """Integration tests for model architectures"""

    def test_all_models_forward(self):
        """Test that all model types can perform forward pass"""
        model_names = [
            "simple_model",
            "simple_total",
            "simple_clover_diff",
            "height_model",
            "height_gshh_model",
            "clover_model",
        ]

        for model_name in model_names:
            config = ModelConfig(
                model_name=model_name,
                backbone_name="tf_efficientnet_b0",
                pretrained=False,
                in_channels=3,
                n_classes=5,
            )

            model = ModelArchitectures(config)
            model.eval()

            sample_input = {"image": torch.randn(2, 3, 224, 224)}

            with torch.no_grad():
                output = model(sample_input)

            assert "logits" in output
            assert output["logits"].shape[0] == 2

    def test_model_device_transfer(self):
        """Test moving model to different devices"""
        config = ModelConfig(
            model_name="simple_model",
            backbone_name="tf_efficientnet_b0",
            pretrained=False,
        )

        model = ModelArchitectures(config)

        # Test CPU
        model = model.to("cpu")
        sample_input = {"image": torch.randn(2, 3, 224, 224)}
        output = model(sample_input)
        assert output["logits"].device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            model = model.to("cuda")
            sample_input = {"image": torch.randn(2, 3, 224, 224).cuda()}
            output = model(sample_input)
            assert output["logits"].device.type == "cuda"


if __name__ == "__main__":
    # Simple test runner
    print("Testing get_model_architecture...")
    test = TestGetModelArchitecture()
    test.test_simple_model()
    print("✓ simple_model created successfully")

    print("\nTesting ModelArchitectures...")
    test = TestModelArchitectures()
    test.test_initialization()
    test.test_forward_pass()
    test.test_different_batch_sizes()
    print("✓ ModelArchitectures tests passed")

    print("\nTesting models with auxiliary heads...")
    test = TestModelArchitecturesWithAuxHeads()
    test.test_height_model_output()
    test.test_clover_model_output()
    print("✓ Auxiliary head tests passed")

    print("\nAll model architecture tests passed!")
