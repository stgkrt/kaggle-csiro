"""Tests for src/model/losses.py"""

import pytest
import torch

from src.configs import LossConfig
from src.model.losses import (
    CloverLoss,
    HeightLoss,
    LossModule,
    MSELoss,
    SmoothL1Loss,
    WeightedCloverLoss,
    WeightedMSELoss,
    WeightedSmoothL1Loss,
)


class TestWeightedMSELoss:
    """Test WeightedMSELoss class"""

    def test_forward(self):
        """Test forward pass"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])
        loss_fn = WeightedMSELoss(weights=weights, device=device)

        # Create dummy data
        batch_size = 4
        n_classes = 5
        preds = torch.randn(batch_size, n_classes).to(device)
        labels = torch.randn(batch_size, n_classes).to(device)

        inputs = {"logits": preds}
        targets = {"labels": labels}

        loss = loss_fn(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0  # MSE is always non-negative

    def test_weights_applied(self):
        """Test that weights are correctly applied"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        loss_fn = WeightedMSELoss(weights=weights, device=device)

        # Only first column should contribute to loss
        preds = torch.zeros(2, 5).to(device)
        labels = torch.zeros(2, 5).to(device)
        labels[:, 0] = 1.0  # Only first column has error

        inputs = {"logits": preds}
        targets = {"labels": labels}

        loss = loss_fn(inputs, targets)
        assert loss.item() > 0  # Should have non-zero loss


class TestMSELoss:
    """Test MSELoss class"""

    def test_forward(self):
        """Test forward pass"""
        loss_fn = MSELoss()

        batch_size = 4
        n_classes = 5
        preds = torch.randn(batch_size, n_classes)
        labels = torch.randn(batch_size, n_classes)

        inputs = {"logits": preds}
        targets = {"labels": labels}

        loss = loss_fn(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_zero_loss(self):
        """Test that identical predictions and labels give zero loss"""
        loss_fn = MSELoss()

        preds = torch.randn(4, 5)
        inputs = {"logits": preds}
        targets = {"labels": preds.clone()}

        loss = loss_fn(inputs, targets)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


class TestSmoothL1Loss:
    """Test SmoothL1Loss class"""

    def test_forward(self):
        """Test forward pass"""
        loss_fn = SmoothL1Loss()

        batch_size = 4
        n_classes = 5
        preds = torch.randn(batch_size, n_classes)
        labels = torch.randn(batch_size, n_classes)

        inputs = {"logits": preds}
        targets = {"labels": labels}

        loss = loss_fn(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestWeightedSmoothL1Loss:
    """Test WeightedSmoothL1Loss class"""

    def test_forward(self):
        """Test forward pass"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])
        loss_fn = WeightedSmoothL1Loss(weights=weights, device=device)

        batch_size = 4
        n_classes = 5
        preds = torch.randn(batch_size, n_classes).to(device)
        labels = torch.randn(batch_size, n_classes).to(device)

        inputs = {"logits": preds}
        targets = {"labels": labels}

        loss = loss_fn(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestHeightLoss:
    """Test HeightLoss class"""

    def test_forward(self):
        """Test forward pass"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_fn = HeightLoss(device=device, aux_height_weight=0.3)

        batch_size = 4
        n_classes = 5
        preds = torch.randn(batch_size, n_classes).to(device)
        labels = torch.randn(batch_size, n_classes).to(device)
        height_preds = torch.randn(batch_size, 1).to(device)
        height_labels = torch.randn(batch_size, 1).to(device)

        inputs = {"logits": preds, "height": height_preds}
        targets = {"labels": labels, "height": height_labels}

        loss = loss_fn(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_aux_weight_effect(self):
        """Test that aux_weight affects the loss"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create identical main predictions/labels (no loss)
        # But different height predictions/labels (aux loss)
        preds = torch.randn(4, 5).to(device)
        labels = preds.clone()
        height_preds = torch.randn(4, 1).to(device)
        height_labels = torch.randn(4, 1).to(device)

        inputs = {"logits": preds, "height": height_preds}
        targets = {"labels": labels, "height": height_labels}

        # Test with different aux weights
        loss_fn_1 = HeightLoss(device=device, aux_height_weight=0.1)
        loss_fn_2 = HeightLoss(device=device, aux_height_weight=0.5)

        loss_1 = loss_fn_1(inputs, targets)
        loss_2 = loss_fn_2(inputs, targets)

        # Higher aux weight should give higher loss (when aux has error)
        assert loss_2.item() > loss_1.item()


class TestCloverLoss:
    """Test CloverLoss class"""

    def test_forward(self):
        """Test forward pass"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_fn = CloverLoss(device=device, aux_clover_weight=0.3)

        batch_size = 4
        n_classes = 5
        preds = torch.randn(batch_size, n_classes).to(device)
        labels = torch.randn(batch_size, n_classes).to(device)
        clover_preds = torch.randn(batch_size, 1).to(device)
        clover_labels = torch.randint(0, 2, (batch_size, 1)).float().to(device)

        inputs = {"logits": preds, "include_clover_preds": clover_preds}
        targets = {"labels": labels, "include_clover_label": clover_labels}

        loss = loss_fn(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestWeightedCloverLoss:
    """Test WeightedCloverLoss class"""

    def test_forward(self):
        """Test forward pass"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])
        loss_fn = WeightedCloverLoss(
            weights=weights, device=device, aux_clover_weight=0.3
        )

        batch_size = 4
        n_classes = 5
        preds = torch.randn(batch_size, n_classes).to(device)
        labels = torch.randn(batch_size, n_classes).to(device)
        clover_preds = torch.randn(batch_size, 1).to(device)
        clover_labels = torch.randint(0, 2, (batch_size, 1)).float().to(device)

        inputs = {"logits": preds, "include_clover_preds": clover_preds}
        targets = {"labels": labels, "include_clover_label": clover_labels}

        loss = loss_fn(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestLossModule:
    """Test LossModule class"""

    def test_mse_loss(self):
        """Test MSE loss creation"""
        config = LossConfig(loss_name="mse_loss")
        loss_module = LossModule(config)

        assert isinstance(loss_module.loss, MSELoss)

    def test_weighted_mse_loss(self):
        """Test weighted MSE loss creation"""
        config = LossConfig(
            loss_name="weighted_mse_loss",
            target_weights=[0.1, 0.1, 0.1, 0.2, 0.5],
        )
        loss_module = LossModule(config)

        assert isinstance(loss_module.loss, WeightedMSELoss)

    def test_smooth_l1_loss(self):
        """Test Smooth L1 loss creation"""
        config = LossConfig(loss_name="smooth_l1_loss")
        loss_module = LossModule(config)

        assert isinstance(loss_module.loss, SmoothL1Loss)

    def test_weighted_smooth_l1_loss(self):
        """Test weighted Smooth L1 loss creation"""
        config = LossConfig(
            loss_name="weighted_smooth_l1_loss",
            target_weights=[0.1, 0.1, 0.1, 0.2, 0.5],
        )
        loss_module = LossModule(config)

        assert isinstance(loss_module.loss, WeightedSmoothL1Loss)

    def test_height_loss(self):
        """Test height loss creation"""
        config = LossConfig(loss_name="height_loss", aux_clover_weight=0.3)
        loss_module = LossModule(config)

        assert isinstance(loss_module.loss, HeightLoss)

    def test_clover_loss(self):
        """Test clover loss creation"""
        config = LossConfig(loss_name="clover_loss", aux_clover_weight=0.3)
        loss_module = LossModule(config)

        assert isinstance(loss_module.loss, CloverLoss)

    def test_forward(self):
        """Test LossModule forward pass"""
        config = LossConfig(loss_name="mse_loss")
        loss_module = LossModule(config)

        batch_size = 4
        n_classes = 5
        preds = torch.randn(batch_size, n_classes)
        labels = torch.randn(batch_size, n_classes)

        inputs = {"logits": preds}
        targets = {"labels": labels}

        loss = loss_module(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_invalid_loss_name(self):
        """Test that invalid loss name raises error"""
        config = LossConfig(loss_name="invalid_loss")
        with pytest.raises(NotImplementedError):
            LossModule(config)


class TestLossIntegration:
    """Integration tests for loss functions"""

    def test_all_losses_produce_gradients(self):
        """Test that all losses can compute gradients"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loss_configs = [
            LossConfig(loss_name="mse_loss"),
            LossConfig(loss_name="weighted_mse_loss"),
            LossConfig(loss_name="smooth_l1_loss"),
            LossConfig(loss_name="weighted_smooth_l1_loss"),
        ]

        for config in loss_configs:
            loss_module = LossModule(config)

            # Create tensor directly on device to maintain leaf status
            preds = torch.randn(4, 5, requires_grad=True, device=device)
            labels = torch.randn(4, 5, device=device)

            inputs = {"logits": preds}
            targets = {"labels": labels}

            loss = loss_module(inputs, targets)
            loss.backward()

            assert preds.grad is not None, f"No gradient for {config.loss_name}"
            assert preds.grad.shape == preds.shape


if __name__ == "__main__":
    # Simple test runner
    print("Testing WeightedMSELoss...")
    test = TestWeightedMSELoss()
    test.test_forward()
    print("✓ WeightedMSELoss forward test passed")

    print("\nTesting MSELoss...")
    test = TestMSELoss()
    test.test_forward()
    test.test_zero_loss()
    print("✓ MSELoss tests passed")

    print("\nTesting LossModule...")
    test = TestLossModule()
    test.test_mse_loss()
    test.test_weighted_mse_loss()
    test.test_forward()
    print("✓ LossModule tests passed")

    print("\nAll loss tests passed!")
