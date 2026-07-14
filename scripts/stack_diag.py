"""Host diagnosis for the failing stack policy: WHERE does it fail?

Per rollout, logs: grasp events (which cube lifted, max height), min xy
distance of the lifted cube to the commanded target, dithering (eef path
length / net displacement), achieved pair, steps. Plus a text-pairwise
conditional delta (src-differing and tgt-differing) on demo states.

Usage:
    MUJOCO_GL=egl python stack_diag.py --episodes 6 --guidance 1.5
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import collections
import json
import numpy as np
import torch
from diffusion_policy.data.dataset import normalize_data, unnormalize_data
from robosuite_stack import make_stack_env, state22, desc_stack, TRAIN_COLORS, rel22
from eval_stack import load_policy, achieved_pair, OBS_HORIZON, ACTION_HORIZON, device
from eval_rs_holdout import KShotRS, cond_delta

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/stack_v1.pth')
    parser.add_argument('--episodes', type=int, default=6)
    parser.add_argument('--guidance', type=float, default=1.5)
    parser.add_argument('--max_steps', type=int, default=450)
    parser.add_argument('--train_seeds', action='store_true',
                        help='use gen-time seeds (in-distribution layouts)')
    args = parser.parse_args()

    policy, stats = load_policy(args.model_path)

    # ---- forward-pass delta on demo tuples: is the text channel alive? ----
    ds = KShotRS('../robosuite_stack_data/train.npz', 8, stats)
    meta = json.load(open('../robosuite_stack_data/train_meta.json'))
    m0 = meta[0]
    alt_src = next(c for c in TRAIN_COLORS if c not in (m0['src'], m0['tgt']))
    alt_tgt = next(c for c in TRAIN_COLORS
                   if c not in (m0['src'], m0['tgt'], alt_src))
    z_base = policy.encode_text([desc_stack(m0['src'], m0['tgt'])]).detach()
    z_src2 = policy.encode_text([desc_stack(alt_src, m0['tgt'])]).detach()
    z_tgt2 = policy.encode_text([desc_stack(m0['src'], alt_tgt)]).detach()
    d_uncond = cond_delta(policy, ds, z_base.cpu().numpy()[0])
    print(f'STACKDIAG delta vs uncond: {d_uncond:.3f}', flush=True)
    # pairwise via get_cond twice (reuse cond_delta by passing tensors)
    for tag, za, zb in [('src-pair', z_base, z_src2), ('tgt-pair', z_base, z_tgt2)]:
        dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
        b = next(iter(dl))
        na = b['agent_pos'].to(device, torch.float32)
        nact = b['action'].to(device, torch.float32)
        B = na.shape[0]
        dummy = torch.zeros(B, OBS_HORIZON, 3, 8, 8, device=device)
        with torch.no_grad():
            ca = policy.get_cond(dummy, na, za.to(device).expand(B, -1))
            cb = policy.get_cond(dummy, na, zb.to(device).expand(B, -1))
            noise = torch.randn_like(nact)
            ts = torch.full((B,), 50, device=device, dtype=torch.long)
            noisy = policy.noise_scheduler.add_noise(nact, noise, ts)
            ea = policy.noise_pred_net(noisy, ts, global_cond=ca)
            eb = policy.noise_pred_net(noisy, ts, global_cond=cb)
        print(f'STACKDIAG delta {tag}: '
              f'{float((ea - eb).flatten(1).norm(dim=1).mean()):.3f}', flush=True)

    # ---- closed-loop anatomy ----
    for i in range(args.episodes):
        seed = (30000 + i) if args.train_seeds else (5000 + i)
        r2 = np.random.default_rng(seed)
        colors = list(r2.choice(TRAIN_COLORS, size=3, replace=False))
        si, ti = [int(x) for x in r2.choice(3, size=2, replace=False)]
        desc = desc_stack(colors[si], colors[ti])
        env = make_stack_env(seed, colors,
                             constrained=stats.get('constrained', False))
        obs = env.reset()
        env.set_target_pair(si, ti)
        z0 = [env.cube_pos(k)[2] for k in range(3)]
        tf = rel22 if stats.get('relative') else (lambda s: s)
        dq = collections.deque([tf(state22(obs, env))] * OBS_HORIZON,
                               maxlen=OBS_HORIZON)
        dummy = torch.zeros(1, OBS_HORIZON, 3, 8, 8, device=device)
        eef_path, lift = [], {k: 0.0 for k in range(3)}
        min_tgt_dist = 99.0
        steps, success = 0, False
        for _ in range(0, args.max_steps, ACTION_HORIZON):
            nag = normalize_data(np.stack(list(dq)), stats=stats['agent_pos'])
            na = torch.from_numpy(nag).unsqueeze(0).to(device, torch.float32)
            with torch.no_grad():
                act = policy.sample(nimages=dummy, nagent_poses=na,
                                    ntexts=[desc], num_diffusion_iters=100,
                                    n_samples=1, guidance_scale=args.guidance)
            acts = unnormalize_data(act.cpu().numpy()[0], stats=stats['action'])
            for a in acts[OBS_HORIZON - 1: OBS_HORIZON - 1 + ACTION_HORIZON]:
                obs, r, d, info = env.step(a)
                dq.append(tf(state22(obs, env)))
                eef_path.append(obs['robot0_eef_pos'].copy())
                for k in range(3):
                    h = env.cube_pos(k)[2] - z0[k]
                    lift[k] = max(lift[k], h)
                    if h > 0.03:  # cube k airborne: distance to commanded tgt
                        if k == si:
                            min_tgt_dist = min(min_tgt_dist, float(np.linalg.norm(
                                env.cube_pos(k)[:2] - env.cube_pos(ti)[:2])))
                steps += 1
                if env._check_success():
                    success = True
                    break
            if success:
                break
        path = np.array(eef_path)
        wander = (np.linalg.norm(np.diff(path, axis=0), axis=1).sum() /
                  (np.linalg.norm(path[-1] - path[0]) + 1e-6))
        ach = (si, ti) if success else achieved_pair(env)
        lifts = ' '.join(f'{colors[k]}:{lift[k]:.03f}' for k in range(3))
        print(f'STACKDIAG ep{i} [{desc}] success={success} steps={steps} '
              f'lifted[{lifts}] min_tgt_dist={min_tgt_dist:.3f} '
              f'wander={wander:.1f} achieved={ach}', flush=True)
        env.close()
    print('STACKDIAG_DONE', flush=True)
