"""DiffusionPolicyXAttn — the language-injection ablation.

Identical to DiffusionPolicy EXCEPT: language enters as a CLIP TOKEN SEQUENCE
via cross-attention blocks inside the UNet (one per down level + mid), instead
of a pooled 64-d vector concatenated into the FiLM global conditioning. The
observation pathway (vision/state pooled into global_cond) is untouched, so
the comparison to the concat twin isolates a single variable.

Unconditional branch (CFG): the token sequence of the empty string "".
Requires a token cache built by cache_text_embeddings.py --tokens
(dict: description -> float16 array (L, 512), L = real tokens incl. BOS/EOS).
"""
import pickle

import numpy as np
import torch
import torch.nn as nn

from .diffusion_policy import DiffusionPolicy
from .network import ConditionalUnet1D

MAX_TOKENS = 24


class DiffusionPolicyXAttn(DiffusionPolicy):
    def __init__(self, *args, cached_tokens_path='../output/cached_tokens.pkl',
                 **kwargs):
        kwargs['noise_pred_net_type'] = 'unet'
        super().__init__(*args, **kwargs)
        # replace the denoiser: global cond WITHOUT the 64-d text slot,
        # text arrives through cross-attention instead
        obs_only = self.obs_dim * self.obs_horizon
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim, global_cond_dim=obs_only,
            text_xattn=True, token_dim=512)
        with open(cached_tokens_path, 'rb') as f:
            self._tokens = pickle.load(f)
        assert '' in self._tokens, 'token cache must include the empty string'

    def _tok(self, texts, device):
        """texts: list[str] -> (B, MAX_TOKENS, 512) float, (B, MAX_TOKENS) pad mask."""
        B = len(texts)
        out = torch.zeros(B, MAX_TOKENS, 512, device=device)
        mask = torch.ones(B, MAX_TOKENS, dtype=torch.bool, device=device)
        for i, t in enumerate(texts):
            arr = torch.from_numpy(np.asarray(self._tokens[t], dtype=np.float32))
            L = min(len(arr), MAX_TOKENS)
            out[i, :L] = arr[:L].to(device)
            mask[i, :L] = False
        return out, mask

    def _obs_cond(self, nimage, nagent_pos):
        B = nimage.shape[0]
        if self.vision:
            feats = self.vision_encoder(nimage.flatten(end_dim=1))
            feats = feats.reshape(B, self.obs_horizon, -1)
            obs = torch.cat([feats, nagent_pos], dim=-1)
        else:
            obs = nagent_pos
        return obs.flatten(start_dim=1)

    def forward(self, nimage, nagent_pos, naction, ntext=None, p_uncond=0.1,
                sample_weights=None):
        B = nimage.shape[0]
        device = naction.device
        cond = self._obs_cond(nimage, nagent_pos)
        toks, mask = self._tok(list(ntext), device)
        # CFG dropout: swap whole token sequences for the "" sequence
        drop = torch.rand(B, device=device) < p_uncond
        if drop.any():
            e_tok, e_mask = self._tok([''], device)
            toks[drop] = e_tok[0]
            mask[drop] = e_mask[0]

        noise = torch.randn_like(naction)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,),
            device=device).long()
        noisy = self.noise_scheduler.add_noise(naction, noise, timesteps)
        pred = self.noise_pred_net(noisy, timesteps, global_cond=cond,
                                   text_tokens=toks, token_mask=mask)
        if sample_weights is None:
            return nn.functional.mse_loss(pred, noise)
        per = nn.functional.mse_loss(pred, noise, reduction='none').mean(dim=(1, 2))
        w = sample_weights.to(device).float()
        return (per * w).sum() / w.sum()

    @torch.no_grad()
    def sample(self, nimages, nagent_poses, ntexts=None,
               num_diffusion_iters=None, n_samples=1, guidance_scale=1.5):
        device = next(self.parameters()).device
        cond = self._obs_cond(nimages, nagent_poses).repeat(n_samples, 1)
        toks, mask = self._tok(list(ntexts), device)
        toks = toks.repeat(n_samples, 1, 1)
        mask = mask.repeat(n_samples, 1)
        e_tok, e_mask = self._tok([''] * toks.shape[0], device)

        naction = torch.randn(
            (n_samples, self.pred_horizon, self.action_dim), device=device)
        sched = self.noise_scheduler
        if num_diffusion_iters is None:
            num_diffusion_iters = sched.config.num_train_timesteps
        sched.set_timesteps(num_diffusion_iters)
        for k in sched.timesteps:
            e_c = self.noise_pred_net(naction, k, global_cond=cond,
                                      text_tokens=toks, token_mask=mask)
            e_u = self.noise_pred_net(naction, k, global_cond=cond,
                                      text_tokens=e_tok, token_mask=e_mask)
            eps = e_u + guidance_scale * (e_c - e_u)
            naction = sched.step(model_output=eps, timestep=k,
                                 sample=naction).prev_sample
        return naction
