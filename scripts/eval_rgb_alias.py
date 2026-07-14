"""RGB-alias control: can we ground a NOVEL color by aliasing its observation
RGB to a trained prototype and commanding that prototype's name?

Scenes contain purple; we overwrite purple's RGB features in the STATE with
red's RGB and command "stack the red block onto the {tgt}". If this reaches
the trained-color ceiling, the novel-value failure is pinned entirely to the
value-lookup circuit (and aliasing is a deployable bridge).

Usage:
    MUJOCO_GL=egl python eval_rgb_alias.py --model_path ../trained_models/stack_v4c.pth
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import collections
import numpy as np
import torch
from diffusion_policy.data.dataset import normalize_data, unnormalize_data
from robosuite_stack import (make_stack_env, state22, desc_stack, rel22,
                             TRAIN_COLORS, HELD_COLOR, COLOR_RGB)
from eval_stack import load_policy, achieved_pair, OBS_HORIZON, ACTION_HORIZON, device

ALIAS_TO = 'red'


def alias22(s, slot):
    s = np.array(s, dtype=np.float32, copy=True)
    s[..., 7 + 6 * slot:10 + 6 * slot] = COLOR_RGB[ALIAS_TO]
    return s


def rollout_alias(policy, stats, seed, max_steps=450):
    torch.manual_seed(seed)
    r2 = np.random.default_rng(seed)
    # scene: purple + two trained colors EXCLUDING the alias prototype (so
    # "red" uniquely denotes the aliased purple cube)
    two = list(r2.choice([c for c in TRAIN_COLORS if c != ALIAS_TO],
                         size=2, replace=False))
    colors = two + [HELD_COLOR]
    r2.shuffle(colors)
    p = colors.index(HELD_COLOR)
    other = int(r2.choice([j for j in range(3) if j != p]))
    si, ti = p, other  # purple as src, like the zero_purple_src arm
    desc = desc_stack(ALIAS_TO, colors[ti])
    tf0 = rel22 if stats.get('relative') else (lambda s: s)
    tf = lambda s: tf0(alias22(s, p))
    env = make_stack_env(seed, colors,
                         constrained=stats.get('constrained', False))
    obs = env.reset()
    env.set_target_pair(si, ti)
    dq = collections.deque([tf(state22(obs, env))] * OBS_HORIZON,
                           maxlen=OBS_HORIZON)
    dummy = torch.zeros(1, OBS_HORIZON, 3, 8, 8, device=device)
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
            if env._check_success():
                success = True
                break
        if success:
            break
    ach = (si, ti) if success else achieved_pair(env)
    env.close()
    return success, steps, ach


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/stack_v4c.pth')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed_base', type=int, default=6000)
    args = parser.parse_args()
    policy, stats = load_policy(args.model_path)
    n_ok, wrong = 0, 0
    for i in range(args.episodes):
        ok, steps, ach = rollout_alias(policy, stats, args.seed_base + i)
        n_ok += ok
        wrong += (not ok) and (ach is not None)
        print(f'ALIAS ep{i}: success={ok} steps={steps} achieved={ach}',
              flush=True)
    print(f'ALIAS_SUMMARY {n_ok}/{args.episodes} (wrong-pair {wrong}) — '
          f'compare zero_purple_src 8/20, trained 18/20', flush=True)
    print('ALIAS_DONE', flush=True)
