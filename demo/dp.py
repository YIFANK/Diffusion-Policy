# policy.py (replace old DiffusionPolicy definition)
import math, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18
from network import ConditionalUnet1D
from vision_encoder import get_resnet, replace_bn_with_gn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
class SimpleTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, texts):
        """
        texts: list of strings, length B
        returns: tensor of shape (B, 1)
        """
        embeddings = []
        for text in texts:
            if "left" in text.lower():
                embeddings.append(-1.0)
            elif "right" in text.lower():
                embeddings.append(1.0)
            else:
                embeddings.append(0.0)
        return torch.tensor(embeddings, dtype=torch.float32).unsqueeze(1)
    
class DiffusionPolicy(nn.Module):
    def __init__(self, obs_horizon = 2, pred_horizon = 16,
                lowdim_obs_dim = 2, action_dim = 2,num_diffusion_iters=100,
                vision = False,text = False):
        super().__init__()

        # vision encoder
        vision_feature_dim = 0
        text_feature_dim = 0
        self.vision, self.text = vision, text
        if vision:
            self.vision_encoder = get_resnet('resnet18')
            self.vision_encoder = replace_bn_with_gn(self.vision_encoder)
            vision_feature_dim = 512  # resnet18 output
        if text:
            # self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
            # for param in self.text_encoder.parameters():
            #     param.requires_grad = False  # freeze
            text_feature_dim = 1
            #encode left as -1, right as 1, and none as 0
            self.text_encoder = SimpleTextEncoder()
        obs_dim = vision_feature_dim + lowdim_obs_dim + text_feature_dim
        lowdim_obs_dim = 2
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
    def forward(self, nimage, nagent_pos, naction, ntext = None, p_uncond = 0.1):
        """
        nimage: shape (B, obs_horizon, C, H, W)
        nagent_pos: shape (B, obs_horizon, 2)
        naction: shape (B, 2)
        """
        B = nimage.shape[0]

        # Vision encoding
        if self.vision:
            image_features = self.vision_encoder(nimage.flatten(end_dim=1))
            image_features = image_features.reshape(B, self.obs_horizon, -1)

            # Concatenate with low-dim observations
            obs_features = torch.cat([image_features, nagent_pos], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
        else:
            obs_features = nagent_pos
        obs_cond = obs_features.flatten(start_dim = 1)
        #text conditioning
        if self.text:
            # encode text (assume frozen encoder)
            text_emb = self.text_encoder(ntext)  # (B, text_emb_dim)

            # classifier-free guidance: randomly drop text conditioning
            mask = (torch.rand(B, device=naction.device) > p_uncond).float().unsqueeze(1)
            text_emb = text_emb * mask  # zero out some text conditions

            # concatenate text embedding into condition
            obs_cond = torch.cat([obs_cond, text_emb], dim=-1)
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
    def sample(self, nimages, nagent_poses, num_diffusion_iters=None, n_samples=1):
        """
        nimages: tensor of shape (obs_horizon, C, H, W)
        nagent_poses: tensor of shape (obs_horizon, 2)
        """
        device = nimages.device

        if self.vision:
            image_features = self.vision_encoder(nimages)  # (obs_horizon, 512)
            obs_features = torch.cat([image_features, nagent_poses], dim=-1)  # (obs_horizon, obs_dim)
        else:
            obs_features = nagent_poses  # (obs_horizon, 2)

        obs_cond_single = obs_features.flatten(start_dim=0)
        obs_cond = obs_cond_single.unsqueeze(0).repeat(n_samples, 1).to(device)
        # initialize Gaussian noise: (n_samples, pred_horizon, action_dim)
        naction = torch.randn(
            (n_samples, self.pred_horizon, self.action_dim), device=device
        )
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

        return naction  # shape: (n_samples, pred_horizon, action_dim)




