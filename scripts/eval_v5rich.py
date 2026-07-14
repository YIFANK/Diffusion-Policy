"""Value-manifold verdict: does a 20-color training palette make HELD-OUT
colors work zero-shot? (v4c 4-color baseline: zero-shot purple-as-src 8/20.)

Arms: trained_rich (sanity) + each held color as src, zero-shot text.

Usage:
    MUJOCO_GL=egl python eval_v5rich.py --model_path ../trained_models/stack_v5rich.pth
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
from robosuite_stack import (make_stack_env, desc_stack, RICH_TRAIN_COLORS,
                             RICH_HELD_COLORS)
from eval_stack import load_policy, rollout_stack

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/stack_v5rich.pth')
    parser.add_argument('--episodes', type=int, default=20)
    args = parser.parse_args()
    policy, stats = load_policy(args.model_path)
    results = {}

    # sanity: trained rich colors
    n_ok, wrong = 0, 0
    for i in range(args.episodes):
        seed = 5000 + i
        r2 = np.random.default_rng(seed)
        colors = list(r2.choice(list(RICH_TRAIN_COLORS), size=3, replace=False))
        si, ti = [int(x) for x in r2.choice(3, size=2, replace=False)]
        ok, steps, ach = rollout_stack(policy, stats,
                                       desc_stack(colors[si], colors[ti]),
                                       seed, colors, (si, ti))
        n_ok += ok
        wrong += (not ok) and (ach is not None)
    results['trained_rich'] = {'success': n_ok, 'episodes': args.episodes,
                               'wrong_pair': wrong}
    print(f'V5EVAL trained_rich: {n_ok}/{args.episodes} (wrong-pair {wrong})',
          flush=True)

    # zero-shot held colors as src
    for hi, held in enumerate(RICH_HELD_COLORS):
        n_ok, wrong = 0, 0
        n = 10
        for i in range(n):
            seed = 6000 + 100 * hi + i
            r2 = np.random.default_rng(seed)
            two = list(r2.choice(list(RICH_TRAIN_COLORS), size=2, replace=False))
            colors = two + [held]
            r2.shuffle(colors)
            p = colors.index(held)
            other = int(r2.choice([j for j in range(3) if j != p]))
            ok, steps, ach = rollout_stack(policy, stats,
                                           desc_stack(held, colors[other]),
                                           seed, colors, (p, other))
            n_ok += ok
            wrong += (not ok) and (ach is not None)
        results[f'zero_{held}'] = {'success': n_ok, 'episodes': n,
                                   'wrong_pair': wrong}
        print(f'V5EVAL zero_{held}_src: {n_ok}/{n} (wrong-pair {wrong})',
              flush=True)

    pooled = sum(results[f'zero_{h}']['success'] for h in RICH_HELD_COLORS)
    print(f'V5EVAL ZERO_HELD_POOLED: {pooled}/40 '
          f'(v4c 4-color baseline: 8/20)', flush=True)
    with open(args.model_path.replace('.pth', '_color_eval.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('V5EVAL_DONE', flush=True)
