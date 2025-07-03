# policy.py (replace old DiffusionPolicy definition)
import math, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18
from .network import ConditionalUnet1D
from .vision_encoder import get_resnet, replace_bn_with_gn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from typing import Optional, Tuple, Dict
from transformers import CLIPTokenizer, CLIPTextModel

class SimpleTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, texts):
        """
        texts: list of strings, length B
        returns: tensor of shape (B, 64)
        """
        if torch.is_tensor(texts):
            # If already a tensor, ensure it's float32
            return texts.float()
        else:
            # Convert to tensor and ensure float32
            return torch.tensor(texts, dtype=torch.float32)
    
class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy for visuomotor control.
    
    Implements DDPM-based action generation conditioned on visual observations,
    low-dimensional state, and optional text conditioning.
    
    Args:
        obs_horizon: Number of observation steps to condition on
        pred_horizon: Number of action steps to predict  
        lowdim_obs_dim: Dimension of low-dimensional observations
        action_dim: Dimension of action space
        num_diffusion_iters: Number of diffusion denoising steps
        vision: Whether to use vision encoder
        text: Whether to use text conditioning
    """
    def __init__(self, obs_horizon = 2, pred_horizon = 16,
                lowdim_obs_dim = 2, action_dim = 2,num_diffusion_iters=100,
                vision = False,text = True):
        super().__init__()

        # vision encoder
        vision_feature_dim = 0
        text_feature_dim = 0
        self.vision, self.text = vision, text
        if vision:
            print("Using vision encoder")
            self.vision_encoder = get_resnet('resnet18')
            self.vision_encoder = replace_bn_with_gn(self.vision_encoder)
            vision_feature_dim = 512  # resnet18 output
        if text:
            # self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
            # for param in self.text_encoder.parameters():
            #     param.requires_grad = False  # freeze
            print("Using text encoder")
            self.text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            text_feature_dim = 512
        obs_dim = vision_feature_dim + lowdim_obs_dim
        lowdim_obs_dim = 2
        action_dim = 2

        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # noise prediction network
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim * obs_horizon + text_feature_dim
        )

        # diffusion noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    
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
    # -------------- training -------------------------------------------------
    def forward(self, nimage, nagent_pos, naction, ntext = None, p_uncond = 0.1):
        """
        nimage: shape (B, obs_horizon, C, H, W)
        nagent_pos: shape (B, obs_horizon, 2)
        naction: shape (B, 2)
        ntext: shape (B, 64)
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
        # Sample noise
        noise = torch.randn_like(naction)

        # Sample random diffusion timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=naction.device
        ).long()

        # Forward diffusion (q_sample)
        noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)

        # Predict noise residual
        noise_pred = self.noise_pred_net(
            noisy_actions, timesteps, global_cond=obs_cond
        )

        # Compute loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

    # -------------- inference -----------------------------------------------
    @torch.no_grad()
    def sample(self, 
              nimages: torch.Tensor,
              nagent_poses: torch.Tensor, 
              ntexts: Optional[torch.Tensor] = None,
              num_diffusion_iters: Optional[int] = None,
              n_samples: int = 1,
              guidance_scale: float = 1.5) -> torch.Tensor:
        """Sample actions from the diffusion policy."""
        device = next(self.parameters()).device
        
        # Get conditional and unconditional features using get_cond
        obs_cond_cond = self.get_cond(nimages, nagent_poses, ntexts, uncond=False)
        obs_cond_uncond = self.get_cond(nimages, nagent_poses, ntexts, uncond=True)
        
        # Repeat for n_samples
        obs_cond_cond = obs_cond_cond.repeat(n_samples, 1).to(device)
        obs_cond_uncond = obs_cond_uncond.repeat(n_samples, 1).to(device)

        # ---- Initialize Gaussian noise ----
        naction = torch.randn((n_samples, self.pred_horizon, self.action_dim), device=device)

        # ---- Prepare scheduler ----
        noise_scheduler = self.noise_scheduler
        if num_diffusion_iters is None:
            num_diffusion_iters = noise_scheduler.config.num_train_timesteps
        noise_scheduler.set_timesteps(num_diffusion_iters)

        # ---- DDPM sampling loop with CFG ----
        for k in noise_scheduler.timesteps:
            # conditional prediction
            eps_cond = self.noise_pred_net(
                sample=naction,
                timestep=k,
                global_cond=obs_cond_cond
            )

            # unconditional prediction
            eps_uncond = self.noise_pred_net(
                sample=naction,
                timestep=k,
                global_cond=obs_cond_uncond
            )

            # classifier-free guidance interpolation
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # diffusion step
            naction = noise_scheduler.step(
                model_output=eps,
                timestep=k,
                sample=naction
            ).prev_sample

        return naction  # (n_samples, pred_horizon, action_dim)





