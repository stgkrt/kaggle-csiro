"""Tests for src/metrics/competition_metrics.py"""

import numpy as np
import pytest
import torch

from src.metrics.competition_metrics import (
    CompetitionMetrics,
    weighted_r2_score,
    weighted_r2_score_torch,
)


class TestWeightedR2Score:
    """Test weighted_r2_score function (numpy version)"""

    def test_perfect_prediction(self):
        """Test with perfect predictions (R² = 1.0)"""
        y_true = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]])
        y_pred = y_true.copy()

        weighted_r2, r2_scores = weighted_r2_score(y_true, y_pred)

        assert np.isclose(weighted_r2, 1.0, atol=1e-6)
        assert np.allclose(r2_scores, np.ones(5), atol=1e-6)

    def test_constant_prediction(self):
        """Test with constant predictions"""
        y_true = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]])
        y_pred = np.ones_like(y_true)

        weighted_r2, r2_scores = weighted_r2_score(y_true, y_pred)

        # R² should be negative for poor predictions
        assert weighted_r2 < 1.0
        assert r2_scores.shape == (5,)

    def test_weights_applied(self):
        """Test that weights are correctly applied"""
        # Create data where only the last column is predicted correctly
        y_true = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 10.0]])
        y_pred = np.array([[0.0, 0.0, 0.0, 0.0, 5.0], [0.0, 0.0, 0.0, 0.0, 10.0]])

        weighted_r2, r2_scores = weighted_r2_score(y_true, y_pred)

        # Last column has weight 0.5, so should dominate the score
        assert r2_scores[4] > r2_scores[0]

    def test_shape_validation(self):
        """Test with correct shapes"""
        y_true = np.random.randn(10, 5)
        y_pred = np.random.randn(10, 5)

        weighted_r2, r2_scores = weighted_r2_score(y_true, y_pred)

        assert isinstance(weighted_r2, (float, np.floating))
        assert r2_scores.shape == (5,)

    def test_single_sample(self):
        """Test with single sample"""
        y_true = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        y_pred = np.array([[1.5, 2.5, 3.5, 4.5, 5.5]])

        weighted_r2, r2_scores = weighted_r2_score(y_true, y_pred)

        # With single sample, R² calculation is edge case
        assert isinstance(weighted_r2, (float, np.floating))
        assert r2_scores.shape == (5,)


class TestWeightedR2ScoreTorch:
    """Test weighted_r2_score_torch function (PyTorch version)"""

    def test_perfect_prediction(self):
        """Test with perfect predictions (R² = 1.0)"""
        y_true = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]])
        y_pred = y_true.clone()

        weighted_r2, r2_scores = weighted_r2_score_torch(y_true, y_pred)

        assert torch.isclose(weighted_r2, torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(r2_scores, torch.ones(5), atol=1e-6)

    def test_constant_prediction(self):
        """Test with constant predictions"""
        y_true = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]])
        y_pred = torch.ones_like(y_true)

        weighted_r2, r2_scores = weighted_r2_score_torch(y_true, y_pred)

        assert weighted_r2.item() < 1.0
        assert r2_scores.shape == (5,)

    def test_cuda_compatibility(self):
        """Test that function works on CUDA if available"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        y_true = torch.randn(10, 5).cuda()
        y_pred = torch.randn(10, 5).cuda()

        weighted_r2, r2_scores = weighted_r2_score_torch(y_true, y_pred)

        assert weighted_r2.device.type == "cuda"
        assert r2_scores.device.type == "cuda"

    def test_gradient_flow(self):
        """Test that gradients can flow through predictions"""
        y_true = torch.randn(10, 5)
        y_pred = torch.randn(10, 5, requires_grad=True)

        weighted_r2, r2_scores = weighted_r2_score_torch(y_true, y_pred)

        # Backward pass should work
        weighted_r2.backward()
        assert y_pred.grad is not None

    def test_numpy_torch_consistency(self):
        """Test that numpy and torch versions give same results"""
        y_true_np = np.random.randn(20, 5)
        y_pred_np = np.random.randn(20, 5)

        y_true_torch = torch.from_numpy(y_true_np).float()
        y_pred_torch = torch.from_numpy(y_pred_np).float()

        weighted_r2_np, r2_scores_np = weighted_r2_score(y_true_np, y_pred_np)
        weighted_r2_torch, r2_scores_torch = weighted_r2_score_torch(
            y_true_torch, y_pred_torch
        )

        assert np.isclose(weighted_r2_np, weighted_r2_torch.item(), atol=1e-5)
        assert np.allclose(r2_scores_np, r2_scores_torch.numpy(), atol=1e-5)


class TestCompetitionMetrics:
    """Test CompetitionMetrics class"""

    def test_initialization(self):
        """Test that class can be initialized"""
        metrics = CompetitionMetrics()
        assert metrics is not None

    def test_call_method(self):
        """Test __call__ method"""
        metrics = CompetitionMetrics()

        y_true = torch.randn(10, 5)
        y_pred = torch.randn(10, 5)

        result = metrics(y_true, y_pred)

        assert isinstance(result, dict)
        assert "weighted_r2" in result
        assert "r2_Dry_Clover_g" in result
        assert "r2_Dry_Dead_g" in result
        assert "r2_Dry_Green_g" in result
        assert "r2_Dry_Total_g" in result
        assert "r2_GDM_g" in result

    def test_metric_values_are_floats(self):
        """Test that all metric values are Python floats"""
        metrics = CompetitionMetrics()

        y_true = torch.randn(10, 5)
        y_pred = torch.randn(10, 5)

        result = metrics(y_true, y_pred)

        for key, value in result.items():
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"

    def test_perfect_prediction_metrics(self):
        """Test metrics with perfect predictions"""
        metrics = CompetitionMetrics()

        y_true = torch.randn(10, 5)
        y_pred = y_true.clone()

        result = metrics(y_true, y_pred)

        # All R² scores should be close to 1.0
        assert np.isclose(result["weighted_r2"], 1.0, atol=1e-5)
        for key in result:
            if key.startswith("r2_"):
                assert np.isclose(result[key], 1.0, atol=1e-5)

    def test_calculate_diff(self):
        """Test calculate_diff method"""
        metrics = CompetitionMetrics()

        y_true = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        y_pred = torch.tensor([[1.5, 2.5, 3.5, 4.5, 5.5]])

        diffs = metrics.calculate_diff(y_true, y_pred)

        expected_diffs = torch.tensor([[-0.5, -0.5, -0.5, -0.5, -0.5]])
        assert torch.allclose(diffs, expected_diffs, atol=1e-6)

    def test_batch_processing(self):
        """Test metrics with different batch sizes"""
        metrics = CompetitionMetrics()

        for batch_size in [1, 4, 16, 32]:
            y_true = torch.randn(batch_size, 5)
            y_pred = torch.randn(batch_size, 5)

            result = metrics(y_true, y_pred)

            assert "weighted_r2" in result
            assert isinstance(result["weighted_r2"], float)


class TestMetricsIntegration:
    """Integration tests for metrics"""

    def test_training_loop_simulation(self):
        """Simulate using metrics in a training loop"""
        metrics = CompetitionMetrics()

        # Simulate multiple batches
        total_r2 = []
        for _ in range(5):
            y_true = torch.randn(16, 5)
            y_pred = torch.randn(16, 5)

            result = metrics(y_true, y_pred)
            total_r2.append(result["weighted_r2"])

        # Check that all metrics were computed
        assert len(total_r2) == 5
        assert all(isinstance(r2, float) for r2 in total_r2)

    def test_edge_cases(self):
        """Test edge cases"""
        metrics = CompetitionMetrics()

        # All zeros
        y_true = torch.zeros(5, 5)
        y_pred = torch.zeros(5, 5)
        result = metrics(y_true, y_pred)
        # R² is undefined for constant true values (0), but function should not crash
        assert "weighted_r2" in result

    def test_negative_predictions(self):
        """Test with negative prediction values"""
        metrics = CompetitionMetrics()

        y_true = torch.randn(10, 5)
        y_pred = -torch.abs(torch.randn(10, 5))  # All negative

        result = metrics(y_true, y_pred)
        assert "weighted_r2" in result

    def test_large_batch(self):
        """Test with large batch size"""
        metrics = CompetitionMetrics()

        y_true = torch.randn(1000, 5)
        y_pred = torch.randn(1000, 5)

        result = metrics(y_true, y_pred)
        assert "weighted_r2" in result
        assert isinstance(result["weighted_r2"], float)


if __name__ == "__main__":
    # Simple test runner
    print("Testing weighted_r2_score...")
    test = TestWeightedR2Score()
    test.test_perfect_prediction()
    test.test_constant_prediction()
    print("✓ weighted_r2_score tests passed")

    print("\nTesting weighted_r2_score_torch...")
    test = TestWeightedR2ScoreTorch()
    test.test_perfect_prediction()
    test.test_numpy_torch_consistency()
    print("✓ weighted_r2_score_torch tests passed")

    print("\nTesting CompetitionMetrics...")
    test = TestCompetitionMetrics()
    test.test_initialization()
    test.test_call_method()
    test.test_perfect_prediction_metrics()
    print("✓ CompetitionMetrics tests passed")

    print("\nAll competition metrics tests passed!")
