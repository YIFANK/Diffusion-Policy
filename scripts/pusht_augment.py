"""PushT factored dataset via augmentation of the human demos.

goal  (4 values): rotate the whole episode by 90°·k about the workspace center.
        With a STATE-based policy the goal pose is absent from the observation,
        so the rotation variant is a true conditioning factor (the policy cannot
        know which goal orientation is commanded without z/text).
speed (2 values): temporal resampling — 'slow' doubles the sequence by linear
        interpolation of states and actions (position setpoints, quasi-static
        pushing, so slowed replay is physically meaningful).

Output: a dict of numpy arrays per (rot, speed) combo, saved as .npz shards
compatible with pusht_p3.PushTZarrDataset-style loading.

Usage:
    python pusht_augment.py --zarr ../pusht_data/pusht/pusht_cchi_v7_replay.zarr \
        --out_dir ../pusht_data/factored
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np

CENTER = np.array([256.0, 256.0])


def rot_xy(xy, k):
    """Rotate points by 90°*k about the workspace center. xy: (...,2)."""
    for _ in range(k % 4):
        x = xy[..., 0] - CENTER[0]
        y = xy[..., 1] - CENTER[1]
        xy = np.stack([CENTER[0] - y, CENTER[1] + x], axis=-1)
    return xy


def rotate_episode(state, action, k):
    """state (T,5)=[ax,ay,bx,by,theta], action (T,2)."""
    s = state.copy()
    s[:, 0:2] = rot_xy(state[:, 0:2], k)
    s[:, 2:4] = rot_xy(state[:, 2:4], k)
    s[:, 4] = state[:, 4] + k * np.pi / 2
    a = rot_xy(action.copy(), k)
    return s, a


def slow_episode(state, action, factor=2):
    """Temporal upsample by linear interpolation (slow execution variant)."""
    T = len(state)
    t_old = np.arange(T)
    t_new = np.linspace(0, T - 1, factor * T - (factor - 1))
    def interp(arr):
        return np.stack([np.interp(t_new, t_old, arr[:, i])
                         for i in range(arr.shape[1])], axis=1)
    s = interp(state)
    # unwrap angle before interpolation to avoid 2pi jumps
    ang = np.interp(t_new, t_old, np.unwrap(state[:, 4]))
    s[:, 4] = ang
    return s.astype(np.float32), interp(action).astype(np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr', default='../pusht_data/pusht/pusht_cchi_v7_replay.zarr')
    parser.add_argument('--out_dir', default='../pusht_data/factored')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='cap source episodes per combo (data-scale ablations)')
    args = parser.parse_args()

    import zarr
    z = zarr.open(args.zarr, mode='r')
    state = np.array(z['data']['state'])
    action = np.array(z['data']['action'])
    ee = np.array(z['meta']['episode_ends'])
    starts = np.concatenate([[0], ee[:-1]])
    n_eps = len(ee) if args.max_episodes is None else min(len(ee), args.max_episodes)

    os.makedirs(args.out_dir, exist_ok=True)
    for k in range(4):
        for speed in ['fast', 'slow']:
            S, A, ends = [], [], []
            total = 0
            for e in range(n_eps):
                s = state[starts[e]:ee[e]]
                a = action[starts[e]:ee[e]]
                s, a = rotate_episode(s, a, k)
                if speed == 'slow':
                    s, a = slow_episode(s, a, 2)
                S.append(s); A.append(a)
                total += len(s)
                ends.append(total)
            out = os.path.join(args.out_dir, f'rot{k}_{speed}.npz')
            np.savez_compressed(out,
                                state=np.concatenate(S).astype(np.float32),
                                action=np.concatenate(A).astype(np.float32),
                                episode_ends=np.array(ends, dtype=np.int64))
            print(f'{out}: {n_eps} eps, {total} frames', flush=True)
    print('AUGMENT_DONE', flush=True)
