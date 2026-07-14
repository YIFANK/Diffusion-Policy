"""Failure-mode census for StackThree policies: N seeded rollouts on novel
layouts, each auto-classified from the state log.

Taxonomy (per rollout, from cube height/xy traces):
  SUCCESS          commanded stack completed
  ALIGNED_NO_REL   carried src to within 4cm xy of tgt, never completed
  MISPLACE         carried src, closest approach to tgt 4-10cm
  TRANSPORT_LOST   carried src, never came within 10cm of tgt
  ROLE_SWAP        carried the TGT cube (onto/near src)
  DISTRACTOR_GRAB  carried the third cube
  NO_GRASP         no cube ever lifted >5cm

Usage:
    MUJOCO_GL=egl python stack_failure_census.py \
        --model_path ../trained_models/stack_v3rel.pth --episodes 30
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
from robosuite_stack import (make_stack_env, state22, desc_stack, rel22,
                             TRAIN_COLORS)
from eval_stack import load_policy, OBS_HORIZON, ACTION_HORIZON, device

LIFT = 0.05


def census_rollout(policy, stats, seed, max_steps=450):
    torch.manual_seed(seed)
    r2 = np.random.default_rng(seed)
    colors = list(r2.choice(TRAIN_COLORS, size=3, replace=False))
    si, ti = [int(x) for x in r2.choice(3, size=2, replace=False)]
    desc = desc_stack(colors[si], colors[ti])
    tf = rel22 if stats.get('relative') else (lambda s: s)
    env = make_stack_env(seed, colors,
                         constrained=stats.get('constrained', False))
    obs = env.reset()
    env.set_target_pair(si, ti)
    dq = collections.deque([tf(state22(obs, env))] * OBS_HORIZON,
                           maxlen=OBS_HORIZON)
    dummy = torch.zeros(1, OBS_HORIZON, 3, 8, 8, device=device)
    z0 = [env.cube_pos(k)[2] for k in range(3)]
    lift_max = [0.0] * 3
    # min xy distance from each cube (while airborne) to each other cube
    approach = np.full((3, 3), 99.0)
    steps, success = 0, False
    for _ in range(0, max_steps, ACTION_HORIZON):
        nag = normalize_data(np.stack(list(dq)), stats=stats['agent_pos'])
        na = torch.from_numpy(nag).unsqueeze(0).to(device, torch.float32)
        with torch.no_grad():
            act = policy.sample(nimages=dummy, nagent_poses=na, ntexts=[desc],
                                num_diffusion_iters=100, n_samples=1)
        acts = unnormalize_data(act.cpu().numpy()[0], stats=stats['action'])
        for a in acts[OBS_HORIZON - 1: OBS_HORIZON - 1 + ACTION_HORIZON]:
            obs, r, d, i = env.step(a)
            dq.append(tf(state22(obs, env)))
            steps += 1
            ps = [env.cube_pos(k) for k in range(3)]
            for k in range(3):
                h = ps[k][2] - z0[k]
                lift_max[k] = max(lift_max[k], h)
                if h > LIFT:
                    for j in range(3):
                        if j != k:
                            approach[k, j] = min(
                                approach[k, j],
                                float(np.linalg.norm(ps[k][:2] - ps[j][:2])))
            if env._check_success():
                success = True
                break
        if success:
            break
    env.close()

    if success:
        cls = 'SUCCESS'
    else:
        lifted = [k for k in range(3) if lift_max[k] > LIFT]
        if not lifted:
            cls = 'NO_GRASP'
        else:
            main = max(lifted, key=lambda k: lift_max[k])
            other = int(np.argmin(approach[main]))
            if main == si:
                d_tgt = approach[si, ti]
                if d_tgt < 0.04:
                    cls = 'ALIGNED_NO_REL'
                elif d_tgt < 0.10:
                    cls = 'MISPLACE'
                else:
                    cls = 'TRANSPORT_LOST'
            elif main == ti:
                cls = 'ROLE_SWAP'
            else:
                cls = 'DISTRACTOR_GRAB'
    return cls, steps, desc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--seed_base', type=int, default=5000)
    args = parser.parse_args()
    policy, stats = load_policy(args.model_path)
    hist = collections.Counter()
    for i in range(args.episodes):
        cls, steps, desc = census_rollout(policy, stats, args.seed_base + i)
        hist[cls] += 1
        print(f'CENSUS ep{i} [{desc}] -> {cls} ({steps} steps)', flush=True)
    name = os.path.basename(args.model_path)
    total = sum(hist.values())
    summary = ' '.join(f'{k}:{v}' for k, v in hist.most_common())
    print(f'CENSUS_SUMMARY {name} n={total}: {summary}', flush=True)
    with open(args.model_path.replace('.pth', '_census.json'), 'w') as f:
        json.dump(dict(hist), f, indent=2)
    print('CENSUS_DONE', flush=True)
