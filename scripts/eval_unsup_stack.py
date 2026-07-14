"""Probe the stacking auto-decoder z-table for OBJECT-factor structure:
kNN probes for src/tgt color, cross-context color-direction consistency,
and centroid arithmetic over instruction pairs.

Usage:
    python eval_unsup_stack.py --model_path ../trained_models/stack_unsup_v1.pth
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import itertools
import json
import numpy as np

COLORS = ['red', 'green', 'blue', 'yellow']


def knn_probe(Z, y, k=5):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    return float(np.mean(cross_val_score(
        KNeighborsClassifier(n_neighbors=k), Z, y, cv=5)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/stack_unsup_v1.pth')
    args = parser.parse_args()

    Z = np.load(args.model_path.replace('.pth', '_ztable.npy'))
    labels = json.load(open(args.model_path.replace('.pth', '_ep_labels.json')))
    pairs = [tuple(l.split('_')) for l in labels]

    for name, idx in [('src', 0), ('tgt', 1)]:
        acc = knn_probe(Z, [p[idx] for p in pairs])
        print(f'STACKUNSUP_PROBE {name}: knn {acc:.2f}', flush=True)
    acc = knn_probe(Z, ['_'.join(p) for p in pairs])
    print(f'STACKUNSUP_PROBE pair (12-way): knn {acc:.2f}', flush=True)

    # SLOT labels (the geometric parameterization: which cube slot is picked /
    # targeted, independent of its color) and layout identity
    meta_path = args.model_path.replace('.pth', '_meta_ref.json')
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        src_slot, tgt_slot, seeds = [], [], []
        for m in meta:
            rng = np.random.default_rng(m['seed'])
            cols = list(rng.choice(COLORS, size=3, replace=False))
            src_slot.append(cols.index(m['src']))
            tgt_slot.append(cols.index(m['tgt']))
            seeds.append(m['seed'])
        for name, y in [('src_slot', src_slot), ('tgt_slot', tgt_slot),
                        ('pair_slot', [f'{a}{b}' for a, b in
                                       zip(src_slot, tgt_slot)])]:
            acc = knn_probe(Z, y)
            print(f'STACKUNSUP_PROBE {name}: knn {acc:.2f} '
                  f'(chance {1/len(set(y)):.2f})', flush=True)
        acc = knn_probe(Z, seeds)
        print(f'STACKUNSUP_PROBE layout: knn {acc:.3f} '
              f'(chance {1/len(set(seeds)):.4f})', flush=True)

    cent = {}
    for p in set(pairs):
        cent[p] = Z[[i for i, q in enumerate(pairs) if q == p]].mean(axis=0)

    # color-direction consistency: dir(a-b as SRC) measured in each shared
    # tgt context; cosine across contexts (and same for TGT role)
    for role, other_role in [('src', 'tgt'), ('tgt', 'src')]:
        cos_all = []
        for a, b in itertools.combinations(COLORS, 2):
            dirs = []
            for c in COLORS:
                if c in (a, b):
                    continue
                ka = (a, c) if role == 'src' else (c, a)
                kb = (b, c) if role == 'src' else (c, b)
                if ka in cent and kb in cent:
                    dirs.append(cent[ka] - cent[kb])
            for i in range(len(dirs)):
                for j in range(i + 1, len(dirs)):
                    cos_all.append(float(np.dot(dirs[i], dirs[j]) /
                                         (np.linalg.norm(dirs[i]) *
                                          np.linalg.norm(dirs[j]))))
        print(f'STACKUNSUP_DIR {role}-color: mean-cos {np.mean(cos_all):.2f} '
              f'min {np.min(cos_all):.2f} (n={len(cos_all)})', flush=True)

    # centroid arithmetic: reach (a->c) from (b->c) + src-direction(a-b)
    # estimated in OTHER tgt contexts (and the tgt-role analogue)
    hits, total = 0, 0
    for (s, t) in cent:
        for role in ['src', 'tgt']:
            for alt in COLORS:
                if alt in (s, t):
                    continue
                start = (alt, t) if role == 'src' else (s, alt)
                if start not in cent:
                    continue
                dirs = []
                for c in COLORS:
                    if c in (s, t, alt):
                        continue
                    if role == 'src':
                        ka, kb = (s, c), (alt, c)
                    else:
                        ka, kb = (c, t), (c, alt)
                    if ka in cent and kb in cent:
                        dirs.append(cent[ka] - cent[kb])
                if not dirs:
                    continue
                z = cent[start] + np.mean(dirs, axis=0)
                nearest = min(cent, key=lambda k: np.linalg.norm(cent[k] - z))
                hits += nearest == (s, t)
                total += 1
    print(f'STACKUNSUP_ARITH {hits}/{total}', flush=True)
    print('STACKUNSUP_DONE', flush=True)
