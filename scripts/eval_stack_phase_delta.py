"""Phase-resolved conditional delta on the stacking policy.

At every demo frame, measure how much the denoiser output moves when the
instruction changes in its SRC color vs its TGT color:

    d_src(t) = ||eps(x,o_t, z[A->C]) - eps(x,o_t, z[B->C])||   (src differs)
    d_tgt(t) = ||eps(x,o_t, z[A->B]) - eps(x,o_t, z[A->C])||   (tgt differs)

binned by normalized episode time. Prediction: d_src peaks before the grasp
(the pick decision), d_tgt stays high through transport until the place
commitment — the policy consults each clause of the instruction exactly when
that decision is live.

Usage:
    python eval_stack_phase_delta.py --model_path ../trained_models/stack_v1.pth
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
from diffusion_policy.data.dataset import normalize_data
from robosuite_stack import desc_stack
from eval_stack import load_policy, OBS_HORIZON, PRED_HORIZON, device

N_BINS = 20
T_MID = 50
GRIP_OPEN = 0.037  # gripper qpos above this = open (state[3])


def episode_slices(d):
    ends = d['episode_ends']
    starts = np.concatenate([[0], ends[:-1]])
    return list(zip(starts, ends))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/stack_v1.pth')
    parser.add_argument('--data', default='../robosuite_stack_data/train.npz')
    parser.add_argument('--episodes', type=int, default=60)
    args = parser.parse_args()

    policy, stats = load_policy(args.model_path)
    d = np.load(args.data)
    if stats.get('relative'):
        from robosuite_stack import rel22
        d = {'state': rel22(d['state']), 'action': d['action'],
             'episode_ends': d['episode_ends']}
    meta = json.load(open(args.data.replace('.npz', '_meta.json')))
    slices = episode_slices(d)

    bins_src = [[] for _ in range(N_BINS)]
    bins_tgt = [[] for _ in range(N_BINS)]
    bins_grip = [[] for _ in range(N_BINS)]
    rng = np.random.default_rng(0)
    ep_ids = rng.choice(len(slices), size=min(args.episodes, len(slices)),
                        replace=False)
    colors_all = ['red', 'green', 'blue', 'yellow']

    for e in ep_ids:
        s0, s1 = slices[e]
        m = meta[e]
        src, tgt = m['src'], m['tgt']
        # alternates present in ANY scene vocab (text-side counterfactuals)
        alt_src = next(c for c in colors_all if c not in (src, tgt))
        alt_tgt = next(c for c in colors_all if c not in (src, tgt, alt_src))
        z = {
            'base': policy.encode_text([desc_stack(src, tgt)]),
            'src2': policy.encode_text([desc_stack(alt_src, tgt)]),
            'tgt2': policy.encode_text([desc_stack(src, alt_tgt)]),
        }
        state = d['state'][s0:s1]
        action = d['action'][s0:s1]
        T = len(state)
        # sample frames along the episode
        for t in np.linspace(0, T - PRED_HORIZON - 1, 24).astype(int):
            obs_win = normalize_data(state[t:t + OBS_HORIZON],
                                     stats['agent_pos'])
            act_win = normalize_data(action[t:t + PRED_HORIZON],
                                     stats['action'])
            na = torch.from_numpy(obs_win[None]).to(device, torch.float32)
            nact = torch.from_numpy(act_win[None]).to(device, torch.float32)
            dummy = torch.zeros(1, OBS_HORIZON, 3, 8, 8, device=device)
            with torch.no_grad():
                conds = {k: policy.get_cond(dummy, na, v.to(device))
                         for k, v in z.items()}
                noise = torch.randn_like(nact)
                ts = torch.full((1,), T_MID, device=device, dtype=torch.long)
                noisy = policy.noise_scheduler.add_noise(nact, noise, ts)
                eps = {k: policy.noise_pred_net(noisy, ts, global_cond=c)
                       for k, c in conds.items()}
            b = min(int(t / T * N_BINS), N_BINS - 1)
            bins_src[b].append(float((eps['base'] - eps['src2']).norm()))
            bins_tgt[b].append(float((eps['base'] - eps['tgt2']).norm()))
            bins_grip[b].append(float(state[t, 3] > GRIP_OPEN))

    out = {
        'bin_centers': [(i + 0.5) / N_BINS for i in range(N_BINS)],
        'd_src': [float(np.mean(b)) if b else None for b in bins_src],
        'd_tgt': [float(np.mean(b)) if b else None for b in bins_tgt],
        'frac_gripper_open': [float(np.mean(b)) if b else None for b in bins_grip],
    }
    with open(args.model_path.replace('.pth', '_phase_delta.json'), 'w') as f:
        json.dump(out, f, indent=2)
    for i in range(N_BINS):
        if out['d_src'][i] is None:
            continue
        print(f'PHASEDELTA bin {out["bin_centers"][i]:.2f}: '
              f'src {out["d_src"][i]:.3f} tgt {out["d_tgt"][i]:.3f} '
              f'open {out["frac_gripper_open"][i]:.2f}', flush=True)
    print('PHASEDELTA_DONE', flush=True)
