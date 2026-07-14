"""Batch-generate factored PickPlaceCan demonstrations.

Per (side, arc, speed) combo: N successful episodes, saved as one .npz with
  img          (T_total, 96, 96, 3) uint8
  state        (T_total, 7)  = [eef_xyz, can_xyz, gripper_width]
  action       (T_total, 7)  = OSC_POSE deltas + gripper
  episode_ends (N,)

Usage:
    MUJOCO_GL=egl python robosuite_gen.py --episodes 60 \
        --combos left_direct_fast right_direct_fast --out_dir ../robosuite_data
"""
import argparse
import os
import numpy as np
from robosuite_expert import rollout

ALL_COMBOS = [f'{s}_{a}_{sp}' for s in ['left', 'right']
              for a in ['direct', 'high'] for sp in ['fast', 'slow']]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=60)
    parser.add_argument('--combos', nargs='+', default=ALL_COMBOS)
    parser.add_argument('--out_dir', default='../robosuite_data')
    parser.add_argument('--suffix', default='',
                        help="output filename suffix, e.g. '_b2' for a second batch")
    parser.add_argument('--seed_base', type=int, default=None,
                        help='explicit seed base (avoids overlap with batch-1 '
                             'hash-derived seeds and with eval seeds 500+)')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for ci, combo in enumerate(args.combos):
        side, arc, speed = combo.split('_')
        max_steps = 900 if speed == 'slow' else 400
        imgs, states, actions, ends = [], [], [], []
        total = 0
        n_ok = 0
        if args.seed_base is not None:
            seed = args.seed_base + ci * 1000
        else:
            seed = hash(('rs', combo)) % 100000
        attempts = 0
        while n_ok < args.episodes and attempts < args.episodes * 3:
            ok, t, frames, sts, acts = rollout(side, arc, speed, seed=seed,
                                               max_steps=max_steps, record=True)
            seed += 1
            attempts += 1
            if not ok:
                continue
            imgs.append(np.array(frames, dtype=np.uint8))
            states.append(np.array(sts, dtype=np.float32))
            actions.append(np.array(acts, dtype=np.float32))
            total += len(frames)
            ends.append(total)
            n_ok += 1
            if n_ok % 10 == 0:
                print(f'{combo}: {n_ok}/{args.episodes}', flush=True)
        out = os.path.join(args.out_dir, f'{combo}{args.suffix}.npz')
        np.savez_compressed(out,
                            img=np.concatenate(imgs),
                            state=np.concatenate(states),
                            action=np.concatenate(actions),
                            episode_ends=np.array(ends, dtype=np.int64))
        print(f'RSGEN {combo}: {n_ok}/{args.episodes} in {attempts} attempts '
              f'({total} frames)', flush=True)
    print('RSGEN_DONE', flush=True)
