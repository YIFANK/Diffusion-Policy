"""Counterfactually-paired stacking demos: per scene seed (fixed layout AND
fixed colors), demonstrate ALL 6 ordered (src, tgt) pairs. A seed is kept only
if all 6 succeed — by construction the data cannot be fit without reading the
instruction.

Usage:
    MUJOCO_GL=egl python robosuite_stack_gen.py --seeds 120 --palette train \
        --out ../robosuite_stack_data/train.npz
    MUJOCO_GL=egl python robosuite_stack_gen.py --seeds 40 --seed_base 50000 \
        --palette held --out ../robosuite_stack_data/held_purple.npz
"""
import argparse
import itertools
import json
import os
import numpy as np
from robosuite_stack import (rollout_expert, TRAIN_COLORS, HELD_COLOR,
                             RICH_TRAIN_COLORS, RICH_HELD_COLORS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=120)
    parser.add_argument('--seed_base', type=int, default=30000)
    parser.add_argument('--palette',
                        choices=['train', 'held', 'rich', 'rich_held', 'rich10', 'rich10_held'],
                        default='train')
    parser.add_argument('--out', default='../robosuite_stack_data/train.npz')
    parser.add_argument('--constrained', action='store_true',
                        help='each cube slot confined to its own y-band')
    parser.add_argument('--save_img', action='store_true',
                        help='also store 96x96 agentview frames (vision host)')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    S, A, ends, meta = [], [], [], []
    IMGS = []
    total = 0
    kept = 0
    for k in range(args.seeds):
        seed = args.seed_base + k
        rng = np.random.default_rng(seed)
        if args.palette == 'train':
            colors = list(rng.choice(TRAIN_COLORS, size=3, replace=False))
        elif args.palette == 'rich10':
            from robosuite_stack import RICH10
            colors = list(rng.choice(list(RICH10), size=3, replace=False))
        elif args.palette == 'rich10_held':
            from robosuite_stack import RICH10
            held = str(rng.choice(list(RICH_HELD_COLORS)))
            two = list(rng.choice(list(RICH10), size=2, replace=False))
            colors = two + [held]
            rng.shuffle(colors)
        elif args.palette == 'rich':
            colors = list(rng.choice(list(RICH_TRAIN_COLORS), size=3,
                                     replace=False))
        elif args.palette == 'rich_held':
            held = str(rng.choice(list(RICH_HELD_COLORS)))
            two = list(rng.choice(list(RICH_TRAIN_COLORS), size=2,
                                  replace=False))
            colors = two + [held]
            rng.shuffle(colors)
        else:
            two = list(rng.choice(TRAIN_COLORS, size=2, replace=False))
            colors = two + [HELD_COLOR]
            rng.shuffle(colors)
        eps = []
        ok_all = True
        for si, ti in itertools.permutations(range(3), 2):
            ok, t, frames, sts, acts = rollout_expert(
                seed, colors, si, ti, max_steps=300, record=True,
                constrained=args.constrained)
            if not ok:
                ok_all = False
                break
            eps.append((np.array(sts, dtype=np.float32),
                        np.array(acts, dtype=np.float32),
                        np.array(frames, dtype=np.uint8) if args.save_img else None,
                        colors[si], colors[ti]))
        if not ok_all:
            print(f'STACKGEN seed {seed}: dropped (pair '
                  f'{colors[si]}->{colors[ti]} failed)', flush=True)
            continue
        for sts, acts, frm, cs, ct in eps:
            S.append(sts); A.append(acts)
            if args.save_img:
                IMGS.append(frm)
            total += len(sts)
            ends.append(total)
            meta.append({'seed': seed, 'src': cs, 'tgt': ct})
        kept += 1
        if kept % 10 == 0:
            print(f'STACKGEN {kept} seeds kept ({len(ends)} episodes)', flush=True)

    arrs = {'state': np.concatenate(S), 'action': np.concatenate(A),
            'episode_ends': np.array(ends, dtype=np.int64)}
    if args.save_img:
        arrs['img'] = np.concatenate(IMGS)
    np.savez_compressed(args.out, **arrs)
    with open(args.out.replace('.npz', '_meta.json'), 'w') as f:
        json.dump(meta, f)
    print(f'STACKGEN_DONE {kept}/{args.seeds} seeds, {len(ends)} episodes, '
          f'{total} frames -> {args.out}', flush=True)
