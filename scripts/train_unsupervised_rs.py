"""v3 auto-decoder on robosuite PickPlaceCan — unsupervised behavior-mode
discovery on a 7-DoF manipulator. Every episode gets a learnable 64-d latent
(embedding table, jointly optimized, L2-regularized); no labels are used for
training. Combo labels are saved separately for post-hoc probing only.

Usage:
    python train_unsupervised_rs.py --data_dir ../robosuite_data --epochs 150
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusion_policy.models.diffusion_policy import DiffusionPolicy
from diffusion_policy.data.dataset import sample_sequence
from train_robosuite_factored import FactoredRS, HELD_OUT_RS

OBS_HORIZON, PRED_HORIZON = 2, 16
Z_DIM = 64


class UnsupRS(FactoredRS):
    """FactoredRS plus per-sample episode index (for the z-table lookup)."""

    def __getitem__(self, idx):
        b0, b1, s0, s1 = self.indices[idx]
        ns = sample_sequence(self.train_data, PRED_HORIZON, b0, b1, s0, s1)
        ep = int(np.searchsorted(self.episode_ends, b0, side='right'))
        return {'image': self.dummy_img[:OBS_HORIZON],
                'agent_pos': ns['agent_pos'][:OBS_HORIZON],
                'action': ns['action'],
                'episode_idx': ep}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../robosuite_data')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--z_lr', type=float, default=1e-3)
    parser.add_argument('--z_reg', type=float, default=1e-4)
    parser.add_argument('--exclude_held_out', action='store_true')
    parser.add_argument('--model_path', default='../trained_models/rs_unsup_v1.pth')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    held = HELD_OUT_RS if args.exclude_held_out else ()
    ds = UnsupRS(args.data_dir, held_out=held)
    n_eps = len(ds.episode_ends)
    print(f'{n_eps} episodes, {len(ds)} samples, held out: {held}', flush=True)
    with open(args.model_path.replace('.pth', '_ep_labels.json'), 'w') as f:
        json.dump(list(ds.ep_combo), f)

    dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=2,
                                     shuffle=True, persistent_workers=True)

    policy = DiffusionPolicy(obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
                             lowdim_obs_dim=7, action_dim=7, num_diffusion_iters=100,
                             vision=False, cached_labels_path='../output/cached_labels.pkl',
                             noise_pred_net_type='unet').to(device)
    z_table = nn.Embedding(n_eps, Z_DIM).to(device)
    nn.init.normal_(z_table.weight, std=0.1)

    ema = EMAModel(parameters=policy.parameters(), power=0.75)
    text_params = set(id(p) for p in policy.text_encoder.parameters())
    other = [p for p in policy.parameters() if id(p) not in text_params]
    opt = torch.optim.AdamW([{'params': other, 'lr': 1e-4},
                             {'params': z_table.parameters(), 'lr': args.z_lr}],
                            weight_decay=1e-6)
    sched = get_scheduler('cosine', optimizer=opt, num_warmup_steps=500,
                          num_training_steps=len(dl) * args.epochs)

    with tqdm(range(args.epochs), desc='Epoch') as tg:
        for ep in tg:
            losses = []
            for b in dl:
                z = z_table(b['episode_idx'].to(device))
                loss = policy(b['image'].to(device, dtype=torch.float32),
                              b['agent_pos'].to(device, dtype=torch.float32),
                              b['action'].to(device, dtype=torch.float32),
                              z, p_uncond=0.1)
                loss = loss + args.z_reg * z.pow(2).mean()
                loss.backward(); opt.step(); opt.zero_grad(); sched.step()
                ema.step(policy.parameters())
                losses.append(loss.item())
            tg.set_postfix(loss=float(np.mean(losses)))

    ema.copy_to(policy.parameters())
    torch.save(policy.state_dict(), args.model_path)
    np.save(args.model_path.replace('.pth', '_ztable.npy'),
            z_table.weight.detach().cpu().numpy())
    with open(args.model_path.replace('.pth', '_stats.json'), 'w') as f:
        json.dump({k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()}
                   for k, v in ds.stats.items()}, f)
    print(f'RSUNSUP saved {args.model_path} ({n_eps} x {Z_DIM})', flush=True)
