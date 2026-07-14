"""Mode-adherence metrics: did a rollout actually execute the commanded
side (cw/ccw approach) and speed (gentle/fast), beyond reaching the goal?

side:  signed winding of the agent-around-block angle during the approach
       phase (before first sustained contact). cw (negative sweep in this
       coordinate convention) vs ccw (positive) — sign calibrated on demos.
speed: steps-to-success, thresholded at the midpoint between the demo
       distributions (per corner), calibrated on demos.

Calibration entry point prints demo separability so thresholds are honest.
"""
import numpy as np

CONTACT_DIST = 75.0   # agent-block distance regarded as "in contact"


def approach_winding(agent_pts, block_pts):
    """Signed total angle (radians) swept by the agent around the block
    before first sustained contact."""
    agent_pts = np.asarray(agent_pts, dtype=np.float64)
    block_pts = np.asarray(block_pts, dtype=np.float64)
    rel = agent_pts - block_pts
    d = np.linalg.norm(rel, axis=1)
    # first index of sustained contact (3 consecutive close frames)
    close = d < CONTACT_DIST
    t_contact = len(d)
    run = 0
    for i, c in enumerate(close):
        run = run + 1 if c else 0
        if run >= 3:
            t_contact = i - 2
            break
    seg = rel[:max(t_contact, 2)]
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    dang = np.diff(ang)
    dang = (dang + np.pi) % (2 * np.pi) - np.pi
    return float(dang.sum())


def side_label(winding, thresh=0.0):
    """ccw if winding > thresh else cw (matches generator convention:
    SIDE_MODES = {'ccw': +1, 'cw': -1} — positive angle steps are ccw)."""
    return 'ccw' if winding > thresh else 'cw'


def speed_label(steps, threshold):
    return 'gentle' if steps > threshold else 'fast'


def calibrate_from_demos(data_dir, corners=range(4), n_per=20):
    """Returns per-corner speed thresholds and reports demo separability."""
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tiny_embodied_reasoning.workspace import utils as wutils

    def episode_metrics(pkl, n):
        scenes = wutils.load_trajectories_pickle(pkl)
        out = []
        for tr in scenes[0].trajectories[:n]:
            a = [s.state['agent']['position'] for s in tr.data]
            b = [s.state['o1']['position'] for s in tr.data]
            out.append((approach_winding(a, b), len(tr.data)))
        return out

    speed_thresh = {}
    side_ok = speed_sep = total = 0
    for c in corners:
        lens = {}
        for side in ['cw', 'ccw']:
            for sp in ['gentle', 'fast']:
                ms = episode_metrics(f'{data_dir}/Blue_{c}_{side}_{sp}.pkl', n_per)
                lens[(side, sp)] = [m[1] for m in ms]
                for w, _ in ms:
                    side_ok += (side_label(w) == side)
                    total += 1
        g = np.concatenate([lens[('cw', 'gentle')], lens[('ccw', 'gentle')]])
        f = np.concatenate([lens[('cw', 'fast')], lens[('ccw', 'fast')]])
        thr = (np.percentile(f, 95) + np.percentile(g, 5)) / 2
        speed_thresh[c] = float(thr)
        speed_sep += (np.sum(g > thr) + np.sum(f <= thr))
    print(f'side adherence on demos: {side_ok}/{total} = {side_ok/total:.2f}')
    print(f'speed thresholds per corner: {speed_thresh}')
    return speed_thresh


if __name__ == '__main__':
    calibrate_from_demos('../dataset_modes_60')
