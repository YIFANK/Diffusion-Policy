"""Mode-adherence metrics for factored robosuite PickPlaceCan rollouts,
computed from state logs (state = [eef_xyz, can_xyz, gripper_width]).

  side:  sign of (eef_y - can_y) at the pre-grasp staging extremum
         (expert stages at can_y + 0.10 for left, -0.10 for right)
  arc:   max eef_z during the carry window (can lifted >5cm above its rest
         height); expert transports at z=1.10 (direct) vs 1.25 (high)
  speed: episode length in steps

Thresholds are calibrated from the demonstrations (95th/5th percentile
midpoint between the two classes, mirroring the PushT adherence protocol).

Run directly to calibrate and print the demo truth table:
    python rs_adherence.py --data_dir ../robosuite_data
"""
import argparse
import glob
import json
import os
import numpy as np

CARRY_LIFT = 0.05   # can this far above its rest height counts as carried
SIDE_MIN = 0.04     # min |eef_y - can_y| for the staging extremum to count


def split_episodes(d):
    ends = d['episode_ends']
    starts = np.concatenate([[0], ends[:-1]])
    return [d['state'][s:e] for s, e in zip(starts, ends)]


def side_signed(ep):
    """Signed y-offset of the eef relative to the can at the LAST pre-grasp
    frame with |dy| above threshold — i.e. the staging side just before final
    centering (robust to where the arm starts). Positive = left."""
    can_z0 = ep[0, 5]
    carried = ep[:, 5] > can_z0 + CARRY_LIFT
    pre = ep[:np.argmax(carried)] if carried.any() else ep
    if len(pre) == 0:
        return 0.0
    dy = pre[:, 1] - pre[:, 4]
    off = np.abs(dy) > SIDE_MIN
    if not off.any():
        return 0.0
    i = len(dy) - 1 - np.argmax(off[::-1])
    return float(dy[i])


def arc_height(ep):
    """Max eef z while the can is carried (lifted > CARRY_LIFT)."""
    can_z0 = ep[0, 5]
    carried = ep[:, 5] > can_z0 + CARRY_LIFT
    if not carried.any():
        return float(ep[:, 2].max())
    return float(ep[carried, 2].max())


def labels_for(ep, thresholds):
    s = side_signed(ep)
    return {
        'side': 'left' if s > 0 else ('right' if s < 0 else 'none'),
        'arc': 'high' if arc_height(ep) > thresholds['arc_z'] else 'direct',
        'speed': 'slow' if len(ep) > thresholds['speed_steps'] else 'fast',
    }


def calibrate(data_dir, pattern='*.npz'):
    """Class-boundary thresholds from the demos: midpoint of the 95th pct of
    the lower class and 5th pct of the upper class."""
    arcs = {'direct': [], 'high': []}
    lens = {'fast': [], 'slow': []}
    for fp in sorted(glob.glob(os.path.join(data_dir, pattern))):
        name = os.path.basename(fp).replace('.npz', '')
        parts = name.split('_')
        if len(parts) < 3:
            continue
        _, arc, speed = parts[0], parts[1], parts[2]
        d = np.load(fp)
        for ep in split_episodes(d):
            arcs[arc].append(arc_height(ep))
            lens[speed].append(len(ep))
    thr = {
        'arc_z': float((np.percentile(arcs['direct'], 95) +
                        np.percentile(arcs['high'], 5)) / 2),
        'speed_steps': float((np.percentile(lens['fast'], 95) +
                              np.percentile(lens['slow'], 5)) / 2),
    }
    return thr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../robosuite_data')
    parser.add_argument('--out', default='../output/rs_adherence_thresholds.json')
    args = parser.parse_args()

    thr = calibrate(args.data_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(thr, f, indent=2)
    print(f'RSADH thresholds: arc_z={thr["arc_z"]:.3f} '
          f'speed_steps={thr["speed_steps"]:.0f}', flush=True)

    # demo truth table: fraction of demos whose measured labels match their combo
    for fp in sorted(glob.glob(os.path.join(args.data_dir, '*.npz'))):
        name = os.path.basename(fp).replace('.npz', '')
        parts = name.split('_')
        if len(parts) < 3:
            continue
        side, arc, speed = parts[0], parts[1], parts[2]
        eps = split_episodes(np.load(fp))
        n = len(eps)
        hits = {'side': 0, 'arc': 0, 'speed': 0}
        for ep in eps:
            lab = labels_for(ep, thr)
            hits['side'] += lab['side'] == side
            hits['arc'] += lab['arc'] == arc
            hits['speed'] += lab['speed'] == speed
        print(f'RSADH {name}: side {hits["side"]}/{n} arc {hits["arc"]}/{n} '
              f'speed {hits["speed"]}/{n}', flush=True)
    print('RSADH_DONE', flush=True)
