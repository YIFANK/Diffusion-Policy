"""Probe the robosuite unsupervised z-table: factor structure, direction
consistency, centroid arithmetic — the v3 battery on a 7-DoF manipulator.

Offline (default): kNN factor probes, per-factor direction consistency,
centroid-arithmetic transfer (24 single-factor jumps).

--closed_loop adds: centroid indexing (8 combos x N eps, adherence-scored)
and causal arithmetic steering (fast centroid + slow direction -> does the
speed signature change?).

Usage:
    python eval_v3_rs.py --model_path ../trained_models/rs_unsup_v1.pth
    MUJOCO_GL=egl python eval_v3_rs.py --model_path ... --closed_loop
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import itertools
import json
import numpy as np

SIDES, ARCS, SPEEDS = ['left', 'right'], ['direct', 'high'], ['fast', 'slow']
FACTORS = {'side': (0, SIDES), 'arc': (1, ARCS), 'speed': (2, SPEEDS)}


def knn_probe(Z, y, k=5):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    return float(np.mean(cross_val_score(
        KNeighborsClassifier(n_neighbors=k), Z, y, cv=5)))


def centroids(Z, combos):
    cent = {}
    for c in set(combos):
        cent[c] = Z[[i for i, x in enumerate(combos) if x == c]].mean(axis=0)
    return cent


def direction_consistency(cent, factor):
    """Pairwise cosine of the factor direction across the cells of the other
    two factors (the 'globally shared direction' test)."""
    idx, vals = FACTORS[factor]
    others = [f for f in FACTORS if f != factor]
    cells = list(itertools.product(*[FACTORS[f][1] for f in others]))
    dirs = []
    for cell in cells:
        a = {others[0]: cell[0], others[1]: cell[1], factor: vals[1]}
        b = {others[0]: cell[0], others[1]: cell[1], factor: vals[0]}
        key = lambda d: (d['side'], d['arc'], d['speed'])
        if key(a) in cent and key(b) in cent:
            dirs.append(cent[key(a)] - cent[key(b)])
    cos = []
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            cos.append(float(np.dot(dirs[i], dirs[j]) /
                             (np.linalg.norm(dirs[i]) * np.linalg.norm(dirs[j]))))
    return float(np.mean(cos)), float(np.min(cos)), len(dirs)


def arithmetic_transfer(cent):
    """For every combo and factor: start at the factor-flipped centroid, add
    the factor direction averaged over the OTHER cells; nearest centroid must
    be the target. 8 combos x 3 factors = 24 tests."""
    hits, total = 0, 0
    all_combos = list(itertools.product(SIDES, ARCS, SPEEDS))
    for c in all_combos:
        d = {'side': c[0], 'arc': c[1], 'speed': c[2]}
        for factor, (idx, vals) in FACTORS.items():
            flip = dict(d)
            flip[factor] = vals[0] if d[factor] == vals[1] else vals[1]
            src = (flip['side'], flip['arc'], flip['speed'])
            others = [f for f in FACTORS if f != factor]
            dirs = []
            for cell in itertools.product(*[FACTORS[f][1] for f in others]):
                if (cell[0], cell[1]) == (d[others[0]], d[others[1]]):
                    continue  # exclude the target's own cell
                hi = {others[0]: cell[0], others[1]: cell[1], factor: d[factor]}
                lo = {others[0]: cell[0], others[1]: cell[1], factor: flip[factor]}
                key = lambda x: (x['side'], x['arc'], x['speed'])
                dirs.append(cent[key(hi)] - cent[key(lo)])
            z = cent[src] + np.mean(dirs, axis=0)
            nearest = min(cent, key=lambda k: np.linalg.norm(cent[k] - z))
            hits += nearest == c
            total += 1
    return hits, total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/rs_unsup_v1.pth')
    parser.add_argument('--closed_loop', action='store_true')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--thresholds', default='../output/rs_adherence_thresholds.json')
    args = parser.parse_args()

    Z = np.load(args.model_path.replace('.pth', '_ztable.npy'))
    labels = json.load(open(args.model_path.replace('.pth', '_ep_labels.json')))
    combos = [tuple(l.split('_')) for l in labels]
    print(f'{len(Z)} episodes, {len(set(combos))} combos', flush=True)

    for factor, (idx, _) in FACTORS.items():
        acc = knn_probe(Z, [c[idx] for c in combos])
        print(f'RSUNSUP_PROBE {factor}: knn {acc:.2f}', flush=True)

    cent = centroids(Z, combos)
    for factor in FACTORS:
        mean_c, min_c, n = direction_consistency(cent, factor)
        print(f'RSUNSUP_DIR {factor}: mean-cos {mean_c:.2f} min {min_c:.2f} '
              f'({n} cells)', flush=True)

    hits, total = arithmetic_transfer(cent)
    print(f'RSUNSUP_ARITH {hits}/{total}', flush=True)

    if args.closed_loop:
        from eval_rs_full import load_policy, eval_arm
        thr = json.load(open(args.thresholds))
        policy, stats = load_policy(args.model_path)
        results = {}
        # centroid indexing: does z alone select the full combination?
        for c in sorted(cent):
            results['_'.join(c)] = eval_arm(policy, stats, cent[c], c,
                                            args.episodes, thr,
                                            tag='centroid', seed_base=700)
        # causal steering: fast centroid + slow direction (other cells only)
        for s in SIDES:
            for a in ARCS:
                dirs = [cent[(s2, a2, 'slow')] - cent[(s2, a2, 'fast')]
                        for s2 in SIDES for a2 in ARCS if (s2, a2) != (s, a)]
                z = cent[(s, a, 'fast')] + np.mean(dirs, axis=0)
                results[f'{s}_{a}_steered'] = eval_arm(
                    policy, stats, z, (s, a, 'slow'), args.episodes, thr,
                    tag='steer_fast+slowdir', seed_base=700)
        with open(args.model_path.replace('.pth', '_closedloop.json'), 'w') as f:
            json.dump(results, f, indent=2)
    print('RSUNSUP_DONE', flush=True)
