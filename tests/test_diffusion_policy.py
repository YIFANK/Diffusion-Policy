"""Basic tests for DiffusionPolicy class."""

import pytest
import torch
import numpy as np
from diffusion_policy.models.diffusion_policy import DiffusionPolicy


class TestDiffusionPolicy:
    """Test cases for DiffusionPolicy."""
    
    def test_init(self):
        """Test DiffusionPolicy initialization."""
        policy = DiffusionPolicy(
            obs_horizon=2,
            pred_horizon=16,
            lowdim_obs_dim=2,
            action_dim=2,
            num_diffusion_iters=100,
            vision=False,
            text=True
        )
        
        assert policy.obs_horizon == 2
        assert policy.pred_horizon == 16
        assert policy.action_dim == 2
        
    def test_forward(self):
        """Test forward pass."""
        policy = DiffusionPolicy(vision=False, text=True)
        
        # Create dummy data
        B = 4
        nimage = torch.randn(B, 2, 3, 96, 96)
        nagent_pos = torch.randn(B, 2, 2)
        naction = torch.randn(B, 16, 2)
        ntext = torch.randn(B, 64)
        
        # Test forward pass
        loss = policy(nimage, nagent_pos, naction, ntext)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative
        
    def test_sample(self):
        """Test sampling from the policy."""
        policy = DiffusionPolicy(vision=False, text=True)
        
        # Create dummy observations
        nimages = torch.randn(2, 3, 96, 96)
        nagent_poses = torch.randn(2, 2)
        ntexts = torch.randn(64)
        
        # Test sampling
        actions = policy.sample(
            nimages=nimages,
            nagent_poses=nagent_poses,
            ntexts=ntexts,
            num_diffusion_iters=10,  # Use fewer iterations for testing
            n_samples=3
        )
        
        assert actions.shape == (3, 16, 2)  # (n_samples, pred_horizon, action_dim)
        assert isinstance(actions, torch.Tensor)
        
    def test_sample_without_text(self):
        """Test sampling without text conditioning."""
        policy = DiffusionPolicy(vision=False, text=False)
        
        nimages = torch.randn(2, 3, 96, 96)
        nagent_poses = torch.randn(2, 2)
        
        actions = policy.sample(
            nimages=nimages,
            nagent_poses=nagent_poses,
            num_diffusion_iters=10,
            n_samples=2
        )
        
        assert actions.shape == (2, 16, 2)


if __name__ == "__main__":
    pytest.main([__file__]) 