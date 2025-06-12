# policy.py (replace old DiffusionPolicy definition)
import math, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18
from network import ConditionalUnet1D
from vision_encoder import get_resnet, replace_bn_with_gn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class DiffusionPolicy(nn.Module):
    def __init__(self, obs_horizon = 2, pred_horizon = 16,
                lowdim_obs_dim = 2, action_dim = 2,num_diffusion_iters=100):
        super().__init__()

        # vision encoder
        self.vision_encoder = get_resnet('resnet18')
        self.vision_encoder = replace_bn_with_gn(self.vision_encoder)
        vision_feature_dim = 512  # resnet18 output
        lowdim_obs_dim = 2
        obs_dim = vision_feature_dim + lowdim_obs_dim
        action_dim = 2

        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # noise prediction network
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim * obs_horizon
        )

        # diffusion noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    # -------------- training -------------------------------------------------
    def forward(self, nimage, nagent_pos, naction):
        """
        nimage: shape (B, obs_horizon, C, H, W)
        nagent_pos: shape (B, obs_horizon, 2)
        naction: shape (B, 2)
        """
        B = nimage.shape[0]

        # Vision encoding
        image_features = self.vision_encoder(nimage.flatten(end_dim=1))
        image_features = image_features.reshape(B, self.obs_horizon, -1)

        # Concatenate with low-dim observations
        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

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
    def sample(self, nimages, nagent_poses, num_diffusion_iters=None):
        """
        nimages: tensor of shape (obs_horizon, C, H, W)
        nagent_poses: tensor of shape (obs_horizon, 2)
        """
        # get image features
        image_features = self.vision_encoder(nimages)
        # (2,512)

        # concat with low-dim observations
        obs_features = torch.cat([image_features, nagent_poses], dim=-1)
        B = 1
        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, self.pred_horizon, self.action_dim), device=nimages.device)
        naction = noisy_action

        # init scheduler
        noise_scheduler = self.noise_scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)

        for k in noise_scheduler.timesteps:
            # predict noise
            noise_pred = self.noise_pred_net(
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        return naction




