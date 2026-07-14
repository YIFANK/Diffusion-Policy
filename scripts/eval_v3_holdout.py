"""Unsupervised composition test: reach HELD-OUT combos on a v3 auto-decoder
policy trained WITHOUT them, via two label-free routes.

Route 1 — latent arithmetic (composition):
    combo A (upper-left, cw, gentle):  z = centroid(ul,cw,fast) + gentle_dir
    combo B (lower-right, ccw, fast):  z = centroid(lr,ccw,gentle) - gentle_dir
  where gentle_dir is averaged over OTHER corners' (gentle - fast) centroid pairs.

Route 2 — few-shot MAP inversion (adaptation):
  optimize z on K=4 demos OF the held-out combo (excluded from training),
  policy frozen, direct conditioning (matches v3's training-time interface).

Metrics: corner success (10 eps, 500 steps) + mean steps-to-success
(the speed-mode signature).

Usage:
    python eval_v3_holdout.py --model_path ../trained_models/dp_unsup_v3_holdout.pth --lowdim
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import numpy as np
import torch
from collections import defaultdict
from diffusion_policy.models.diffusion_policy import DiffusionPolicy
from diffusion_policy.data.dataset import PushTImageDataset
from generate_demos import CORNER_NAMES
from train_modes import build_split
from eval_v3_arith import rollout  # reuses the direct-z rollout machinery

OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HELD = {'A': (2, 'cw', 'gentle'), 'B': (0, 'ccw', 'fast')}


def map_invert(policy, demo_path, state_keys, k_shot=4, epochs=200, lr=1e-2,
               z_init=None, prior_weight=0.0):
    """Optimize a single z on K demos, frozen policy, direct conditioning.

    z_init/prior_weight implement geometry-prior warm starting: initialize at
    the arithmetic composition point and (optionally) add a proximal term
    prior_weight * ||z - z_init||^2 so few demos refine rather than replace
    the prior."""
    ds = PushTImageDataset([demo_path], [0], pred_horizon=PRED_HORIZON,
                           obs_horizon=OBS_HORIZON, action_horizon=ACTION_HORIZON,
                           num_trajectories=k_shot, state_keys=state_keys)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    if z_init is None:
        z = torch.zeros(1, 64, device=device, requires_grad=True)
        z_anchor = None
    else:
        z = torch.tensor(z_init[None], dtype=torch.float32, device=device,
                         requires_grad=True)
        z_anchor = z.detach().clone()
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        for b in dl:
            nimage = b['image'][:, :OBS_HORIZON].to(device)
            nagent = b['agent_pos'][:, :OBS_HORIZON].to(device)
            naction = b['action'].to(device)
            loss = policy(nimage, nagent, naction, z.expand(nimage.shape[0], -1),
                          p_uncond=0.0)
            if z_anchor is not None and prior_weight > 0:
                loss = loss + prior_weight * (z - z_anchor).pow(2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return z.detach().cpu().numpy()[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/dp_unsup_v3_holdout.pth')
    parser.add_argument('--lowdim', action='store_true')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--data_dir', default='../dataset_modes_60')
    args = parser.parse_args()
    state_keys = ('agent', 'o1') if args.lowdim else ('agent',)

    # centroids from the holdout-trained z-table (14 combos only)
    Z = np.load(args.model_path.replace('.pth', '_ztable.npy'))
    descs = json.load(open(args.model_path.replace('.pth', '_ep_labels.json')))
    def parse(d):
        if d is None: return None
        return (next(i for i, x in enumerate(CORNER_NAMES) if x in d),
                'ccw' if 'counterclockwise' in d else 'cw',
                'gentle' if 'gently' in d else 'fast')
    groups = defaultdict(list)
    for z, d in zip(Z, descs):
        p = parse(d)
        if p: groups[p].append(z)
    cent = {k: np.mean(v, axis=0) for k, v in groups.items()}
    assert HELD['A'] not in cent and HELD['B'] not in cent, 'held-out leaked into training!'

    def gentle_dir(exclude_corner):
        ds_ = [cent[(c, s, 'gentle')] - cent[(c, s, 'fast')]
               for c in range(4) if c != exclude_corner for s in ['cw', 'ccw']
               if (c, s, 'gentle') in cent and (c, s, 'fast') in cent]
        return np.mean(ds_, axis=0)

    # normalization stats (train combos only — mirrors training)
    tp, td, _, _ = build_split()
    paths = [p.replace('../dataset_modes/', args.data_dir.rstrip('/') + '/') for p in tp]
    stats = PushTImageDataset(paths, td, pred_horizon=PRED_HORIZON,
                              obs_horizon=OBS_HORIZON, action_horizon=ACTION_HORIZON,
                              state_keys=state_keys).stats

    policy = DiffusionPolicy(obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
                             lowdim_obs_dim=2 * len(state_keys), action_dim=2,
                             num_diffusion_iters=100, vision=not args.lowdim,
                             cached_labels_path='../output/cached_labels.pkl',
                             noise_pred_net_type='unet').to(device)
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    policy.eval()

    # candidate z per held-out combo
    zs = {}
    cA, cB = HELD['A'][0], HELD['B'][0]
    zs['A/arithmetic'] = cent[(cA, 'cw', 'fast')] + gentle_dir(cA)
    zs['B/arithmetic'] = cent[(cB, 'ccw', 'gentle')] - gentle_dir(cB)
    colors_cap = {2: 'Blue_2', 0: 'Blue_0'}
    zs['A/inversion_K4'] = map_invert(
        policy, f'{args.data_dir}/Blue_{cA}_cw_gentle.pkl', state_keys)
    zs['B/inversion_K4'] = map_invert(
        policy, f'{args.data_dir}/Blue_{cB}_ccw_fast.pkl', state_keys)

    results = {}
    for tag, z in zs.items():
        combo = HELD[tag[0]]
        outs = [rollout(policy, stats, z.astype(np.float32), combo[0], 7000 + i,
                        state_keys) for i in range(args.episodes)]
        succ = sum(o[0] for o in outs)
        steps = [o[1] for o in outs if o[0]]
        results[tag] = {'success': succ,
                        'mean_steps': float(np.mean(steps)) if steps else None}
        ms = f" mean_steps={np.mean(steps):.0f}" if steps else ""
        print(f'V3H_RESULT {tag} ({CORNER_NAMES[combo[0]]},{combo[1]},{combo[2]}): '
              f'{succ}/{args.episodes}{ms}', flush=True)

    with open(args.model_path.replace('.pth', '_holdout_eval.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('V3H_DONE', json.dumps(results), flush=True)
