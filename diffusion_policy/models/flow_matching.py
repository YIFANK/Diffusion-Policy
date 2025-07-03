#flow matching policy
import math, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18
from .network import ConditionalUnet1D
from .vision_encoder import get_resnet, replace_bn_with_gn
from typing import Optional, Tuple, Dict
from transformers import CLIPTokenizer, CLIPTextModel

class FlowMatchingPolicy(nn.Module):
    def __init__(self, obs_horizon = 2, pred_horizon = 16,
                lowdim_obs_dim = 2, action_dim = 2, num_diffusion_iters=100,
                vision = False, text = True):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.lowdim_obs_dim = lowdim_obs_dim
        self.action_dim = action_dim
        vision_feature_dim = 0
        text_feature_dim = 0
        self.vision, self.text = vision, text
        self.num_diffusion_iters = num_diffusion_iters
        if vision:
            print("Using vision encoder")
            self.vision_encoder = get_resnet('resnet18')
            self.vision_encoder = replace_bn_with_gn(self.vision_encoder)
            vision_feature_dim = 512  # resnet18 output
            
        if text:
            print("Using text encoder")
            # self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
            # for param in self.text_encoder.parameters():
            #     param.requires_grad = False  # freeze
            self.text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            text_feature_dim = 512
        obs_dim = vision_feature_dim + lowdim_obs_dim
        
        # Flow matching predictor (predicts vector field)
        self.flow_predictor = ConditionalUnet1D(
            input_dim = action_dim,
            global_cond_dim = obs_dim * obs_horizon + text_feature_dim,
        )
    
    def forward(self, nimage, nagent_pos, naction, ntext = None, p_uncond = 0.1):
        """
        Flow matching forward pass with classifier-free guidance.
        
        Args:
            nimage: shape (B, obs_horizon, C, H, W)
            nagent_pos: shape (B, obs_horizon, 2)
            naction: shape (B, pred_horizon, action_dim) - target actions
            ntext: shape (B, 64) - text conditioning
            p_uncond: probability of dropping text conditioning for CFG
        
        Returns:
            loss: flow matching loss
        """
        B = nimage.shape[0]

        # Get both conditional and unconditional features
        cond = self.get_cond(nimage, nagent_pos, ntext, uncond=False)
        uncond = self.get_cond(nimage, nagent_pos, ntext, uncond=True)
        
        # Create mask for which samples get unconditional treatment
        uncond_mask = torch.rand(B, device=naction.device) < p_uncond
        
        # Combine conditional and unconditional based on mask
        obs_cond = torch.where(
            uncond_mask.unsqueeze(-1),
            uncond,
            cond
        )

        # Flow matching training
        # Sample random time points t ~ U(0, 1)
        t = torch.rand(B, device=naction.device).unsqueeze(1)  # (B, 1)
        # Sample noise (prior samples, typically Gaussian)
        noise = torch.randn_like(naction)  # (B, pred_horizon, action_dim)
        
        # Create interpolated samples: x_t = (1-t) * noise + t * data
        # This creates a straight path from noise to data
        x_t = (1 - t.unsqueeze(-1)) * noise + t.unsqueeze(-1) * naction
        
        # The true vector field for straight paths is: v_t = data - noise
        true_vector_field = naction - noise
        #scale up to 0-100
        timestep = t.squeeze(-1) * 100
        # Predict vector field using the flow predictor
        pred_vector_field = self.flow_predictor(
            x_t,  # (B, pred_horizon, action_dim)
            timestep=timestep,  # (B,)
            global_cond=obs_cond
        )
        
        # Compute flow matching loss (MSE between predicted and true vector field)
        loss = F.mse_loss(pred_vector_field, true_vector_field)
        
        return loss
    def encode_text(self, text_list):
        tokens = self.tokenizer(text = text_list, padding=True, return_tensors="pt").to(next(self.parameters()).device)
        with torch.no_grad():
            return self.text_encoder(**tokens).last_hidden_state[:, 0, :] # (B, 512)


    def get_cond(self, nimage, nagent_pos, ntext, uncond = False):
        """
        Get conditioning features for inference.
        """
        B = nimage.shape[0]
        if self.vision:
            image_features = self.vision_encoder(nimage.flatten(end_dim=1))
            image_features = image_features.reshape(B, self.obs_horizon, -1)
            obs_features = torch.cat([image_features, nagent_pos], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)
        else:
            obs_features = nagent_pos
            obs_cond = obs_features.flatten(start_dim=1)
        if self.text:
            #tokenize text
            if uncond or ntext is None:
                text_emb = torch.zeros(B, 512, device=nimage.device)
            else:
                text_emb = self.encode_text(ntext)
            obs_cond = torch.cat([obs_cond, text_emb], dim=-1) 
        return obs_cond
    
    @torch.no_grad()
    def sample(self, nimages: torch.Tensor,
            nagent_poses: torch.Tensor, 
            ntexts: Optional[torch.Tensor] = None,
            nsamples: int = 1,
            guidance_scale: float = 1.5) -> torch.Tensor:
        """
        Generate actions using flow matching inference (ODE solving).
        
        Args:
            nimage: shape (B, obs_horizon, C, H, W)
            nagent_pos: shape (B, obs_horizon, 2)
            ntext: shape (B, 64)
            nsamples: number of samples to generate
            guidance_scale: guidance scale for classifier-free guidance
            
        Returns:
            actions: shape (B, pred_horizon, action_dim)
        """
        device = nimages.device
        
        # Get conditioning
        obs_cond = self.get_cond(nimages, nagent_poses, ntexts)
        obs_uncond = self.get_cond(nimages, nagent_poses, ntexts, uncond = True)
        #repeat obs_cond and obs_uncond for nsamples
        obs_cond = obs_cond.repeat(nsamples, 1)
        obs_uncond = obs_uncond.repeat(nsamples, 1)
        # Start from noise
        x = torch.randn(nsamples, self.pred_horizon, self.action_dim, device=device)
        # ODE integration using Euler method
        dt = 1.0 / self.num_diffusion_iters
        
        for i in range(self.num_diffusion_iters):
            t_val = i * dt
            # Predict vector field
            v_cond = self.flow_predictor(x, timestep=t_val * 100, global_cond=obs_cond)
            v_uncond = self.flow_predictor(x, timestep=t_val * 100, global_cond=obs_uncond)
            v_t = guidance_scale * (v_cond - v_uncond) + v_uncond
            
            # Euler step: x_{t+dt} = x_t + dt * v_t
            x = x + dt * v_t
            
        return x


