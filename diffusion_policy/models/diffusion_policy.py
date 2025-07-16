# policy.py (replace old DiffusionPolicy definition)
import math, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18
from .network import ConditionalUnet1D, ConditionalTransformer1D
from .vision_encoder import get_resnet, replace_bn_with_gn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from typing import Optional, Tuple, Dict
from transformers import BertTokenizer, VisualBertModel
import pickle
    
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
        cached_labels_path: Path to cached text labels
        noise_pred_net_type: Type of noise prediction network ('unet' or 'transformer')
    """
    def __init__(self, obs_horizon = 2, pred_horizon = 16,
                lowdim_obs_dim = 2, action_dim = 2,num_diffusion_iters=100,
                vision = False,text = True, cached_labels_path = '../output/cached_labels.pkl',
                noise_pred_net_type = 'unet'):
        super().__init__()

        # vision encoder
        vision_feature_dim = 0
        text_feature_dim = 0
        self.vision, self.text = vision, text
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if vision:
            print("Using vision encoder")
            #vision encoder = frozen resnet
            self.vision_encoder = replace_bn_with_gn(get_resnet('resnet18'))
            # self.vision_encoder = nn.Sequential(
            #     replace_bn_with_gn(get_resnet('resnet18')),
            #     nn.Linear(512, 128),
            #     nn.ReLU(),
            #     nn.Linear(128, 128),
            #     nn.ReLU(),
            #     nn.Linear(128, 128),
            # )
            #freeze the resnet
            vision_feature_dim = 512 # resnet18 output
        if text:
            # self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
            # for param in self.text_encoder.parameters():
            #     param.requires_grad = False  # freeze
            print("Using text encoder")
            #use linear projection as text encoder
            self.text_encoder = nn.Sequential(
                nn.Linear(3584,128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128,128),
                nn.ReLU(),
                nn.Linear(128,64)
            )
            text_feature_dim = 64
            self.text_feature_dim = 64
        obs_dim = vision_feature_dim + lowdim_obs_dim
        lowdim_obs_dim = 2
        action_dim = 2

        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cached_labels_path = cached_labels_path
        self.cached_labels = self.load_cached_labels()  
        # noise prediction network
        if noise_pred_net_type == 'unet':
            print("Using UNet-based noise prediction network")
            self.noise_pred_net = ConditionalUnet1D(
                input_dim=action_dim,
                global_cond_dim=obs_dim * obs_horizon + text_feature_dim
            )
        elif noise_pred_net_type == 'transformer':
            print("Using Transformer-based noise prediction network")
            self.noise_pred_net = ConditionalTransformer1D(
                input_dim=action_dim,
                obs_dim=obs_dim,
                global_cond_dim=text_feature_dim,
                max_seq_len=pred_horizon
            )
        else:
            raise ValueError(f"Unknown noise_pred_net_type: {noise_pred_net_type}. Choose 'unet' or 'transformer'.")

        # diffusion noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    
    def encode_text(self, text_list):
        # If text_list contains tensors, use them directly
        if isinstance(text_list[0], torch.Tensor):
            text_emb = text_list
        else:
            # Load cached labels if not already loaded
            if not hasattr(self, '_cached_labels'):
                with open(self.cached_labels_path, 'rb') as f:
                    self._cached_labels = pickle.load(f)
            # Get embeddings for each text
            try:
                text_emb = [self._cached_labels[text] for text in text_list]
            except KeyError as e:
                raise KeyError(f"Text '{e.args[0]}' not found in cached labels")
            text_emb = torch.cat(text_emb, dim=0)
        # print(text_emb.shape)
        # Ensure float type
        text_emb = text_emb.float()
        # Project through linear layer
        text_emb = self.text_encoder(text_emb)
        
        return text_emb
        

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
                text_emb = torch.zeros(B, self.text_feature_dim, device=nimage.device)
            else:
                if isinstance(ntext, list):
                    text_emb = self.encode_text(ntext)
                else:
                    text_emb = ntext
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
        # Ensure cond and uncond have same shape
        assert cond.shape[1] == uncond.shape[1], f"Conditional and unconditional shapes don't match: {cond.shape} vs {uncond.shape}"
        
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

    def load_cached_labels(self):
        """Load cached labels from file."""
        try:
            with open(self.cached_labels_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: {self.cached_labels_path} not found. Creating new cache.")
            return {}
    
    def save_cached_labels(self, labels):
        """Save cached labels to file."""
        with open(self.cached_labels_path, 'wb') as f:
            pickle.dump(labels, f)

    def clear_cached_labels(self):
        """Clear cached labels."""
        self.cached_labels = {}
        self.save_cached_labels(self.cached_labels)
    






