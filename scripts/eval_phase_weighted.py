"""E-PW: does PHASE-WEIGHTED inversion stabilize few-shot mode inference?

Known weakness: cold MAP inversion is high-variance (K=4 goal across seeds
8,4,3,6; K=1 at 66% of oracle). Hypothesis: the inversion loss weights all
demo frames uniformly, but z only influences the denoiser at decision-dense
phases -- uniform weighting drowns the informative frames in gradient noise.

Method: measure the host's consultation profile w(t) = ||eps(x,o,z_own) -
eps(x,o,0)|| binned by normalized episode time (forward passes on TRAINED
demos, no labels), then weight the inversion loss per-sample by w(t).

Arms (push v3-holdout host, held-out combos A/B):
    K in {1,4} x {uniform, phase-weighted} x 3 restarts, 10 eps each.

Usage:
    python eval_phase_weighted.py --lowdim
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
from eval_v3_arith import rollout
from eval_v3_holdout import HELD

OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8
N_BINS = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ep_lengths(ds):
    L = defaultdict(int)
    for e, r in zip(ds.episode_idx, ds.rel_frame):
        L[int(e)] = max(L[int(e)], int(r) + 1)
    return L


def phase_profile(policy, ds, Z, t_mid=50, n_batches=8):
    """Consultation profile: per-sample ||eps(z_own) - eps(0)|| binned by
    normalized episode time, on trained demos."""
    L = ep_lengths(ds)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    bins = [[] for _ in range(N_BINS)]
    it = iter(dl)
    for _ in range(n_batches):
        try:
            b = next(it)
        except StopIteration:
            break
        nimage = b['image'][:, :OBS_HORIZON].to(device)
        nagent = b['agent_pos'][:, :OBS_HORIZON].to(device)
        naction = b['action'].to(device)
        B = nagent.shape[0]
        z_own = torch.from_numpy(
            Z[b['episode_idx'].numpy()]).float().to(device)
        z0 = torch.zeros_like(z_own)
        with torch.no_grad():
            c1 = policy.get_cond(nimage, nagent, z_own)
            c0 = policy.get_cond(nimage, nagent, z0)
            noise = torch.randn_like(naction)
            ts = torch.full((B,), t_mid, device=device, dtype=torch.long)
            noisy = policy.noise_scheduler.add_noise(naction, noise, ts)
            e1 = policy.noise_pred_net(noisy, ts, global_cond=c1)
            e0 = policy.noise_pred_net(noisy, ts, global_cond=c0)
            d = (e1 - e0).flatten(1).norm(dim=1).cpu().numpy()
        for i in range(B):
            e = int(b['episode_idx'][i])
            frac = min(float(b['rel_frame'][i]) / max(L[e], 1), 0.999)
            bins[int(frac * N_BINS)].append(float(d[i]))
    prof = np.array([np.mean(x) if x else np.nan for x in bins])
    prof = np.where(np.isnan(prof), np.nanmean(prof), prof)
    prof = prof / prof.mean()
    prof = np.maximum(prof, 0.25)  # floor: never fully zero out a phase
    return prof / prof.mean()


def map_invert_w(policy, ds, profile=None, epochs=200, lr=1e-2, seed=0):
    """MAP inversion with optional per-sample phase weights."""
    torch.manual_seed(seed)
    L = ep_lengths(ds) if profile is not None else None
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    z = torch.zeros(1, 64, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        for b in dl:
            nimage = b['image'][:, :OBS_HORIZON].to(device)
            nagent = b['agent_pos'][:, :OBS_HORIZON].to(device)
            naction = b['action'].to(device)
            B = nagent.shape[0]
            if profile is not None:
                fr = np.array([min(float(r) / max(L[int(e)], 1), 0.999)
                               for r, e in zip(b['rel_frame'], b['episode_idx'])])
                w = torch.from_numpy(
                    profile[(fr * N_BINS).astype(int)]).float()
            else:
                w = None
            loss = policy(nimage, nagent, naction, z.expand(B, -1),
                          p_uncond=0.0, sample_weights=w)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return z.detach().cpu().numpy()[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        default='../trained_models/dp_unsup_v3_holdout.pth')
    parser.add_argument('--lowdim', action='store_true')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--data_dir', default='../dataset_modes_60')
    parser.add_argument('--restarts', type=int, default=3)
    args = parser.parse_args()
    state_keys = ('agent', 'o1') if args.lowdim else ('agent',)

    Z = np.load(args.model_path.replace('.pth', '_ztable.npy'))
    tp, td, _, _ = build_split()
    paths = [p.replace('../dataset_modes/', args.data_dir.rstrip('/') + '/')
             for p in tp]
    train_ds = PushTImageDataset(paths, td, pred_horizon=PRED_HORIZON,
                                 obs_horizon=OBS_HORIZON,
                                 action_horizon=ACTION_HORIZON,
                                 state_keys=state_keys)
    stats = train_ds.stats

    policy = DiffusionPolicy(obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
                             lowdim_obs_dim=2 * len(state_keys), action_dim=2,
                             num_diffusion_iters=100, vision=not args.lowdim,
                             cached_labels_path='../output/cached_labels.pkl',
                             noise_pred_net_type='unet').to(device)
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    policy.eval()

    prof = phase_profile(policy, train_ds, Z)
    print('EPW_PROFILE', ' '.join(f'{x:.2f}' for x in prof), flush=True)

    results = {}
    for tag, (corner, side, speed) in HELD.items():
        demo = f'{args.data_dir}/Blue_{corner}_{side}_{speed}.pkl'
        for K in [1, 4]:
            kds = PushTImageDataset([demo], [0], pred_horizon=PRED_HORIZON,
                                    obs_horizon=OBS_HORIZON,
                                    action_horizon=ACTION_HORIZON,
                                    num_trajectories=K, state_keys=state_keys)
            for wtag, p in [('uniform', None), ('phase', prof)]:
                succ = []
                for r in range(args.restarts):
                    z = map_invert_w(policy, kds, profile=p, seed=r)
                    outs = [rollout(policy, stats, z.astype(np.float32),
                                    corner, 7000 + i, state_keys)
                            for i in range(args.episodes)]
                    succ.append(sum(o[0] for o in outs))
                results[f'{tag}/K{K}/{wtag}'] = succ
                print(f'EPW {tag} K={K} {wtag}: {succ} '
                      f'(mean {np.mean(succ):.1f} std {np.std(succ):.1f})',
                      flush=True)

    with open('../output/epw_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('EPW_DONE', flush=True)
