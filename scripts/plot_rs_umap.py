"""UMAP of the robosuite unsupervised z-table, colored by the ground-truth
factors the model never saw (side / arc / speed).

Usage:
    python plot_rs_umap.py --model rs_unsup_v2
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import umap

FACTORS = {'side': ({'left': 0, 'right': 1}, ['#d62728', '#1f77b4']),
           'arc': ({'direct': 0, 'high': 1}, ['#2ca02c', '#9467bd']),
           'speed': ({'fast': 0, 'slow': 1}, ['#ff7f0e', '#17becf'])}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='rs_unsup_v2')
    parser.add_argument('--dir', default='../trained_models')
    parser.add_argument('--out', default='../paper/figures/rs_z_umap.pdf')
    args = parser.parse_args()

    Z = np.load(os.path.join(args.dir, f'{args.model}_ztable.npy'))
    labels = json.load(open(os.path.join(args.dir, f'{args.model}_ep_labels.json')))
    parts = [l.split('_') for l in labels]

    emb = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=0).fit_transform(Z)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, (fname, (vmap, colors)) in zip(axes, FACTORS.items()):
        idx = ['side', 'arc', 'speed'].index(fname)
        vals = [p[idx] for p in parts]
        for v, c in zip(vmap, colors):
            m = np.array([x == v for x in vals])
            ax.scatter(emb[m, 0], emb[m, 1], s=6, c=c, label=v, alpha=0.75,
                       linewidths=0)
        ax.set_title(f'colored by {fname}', fontsize=11)
        ax.legend(frameon=False, fontsize=9, markerscale=2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_alpha(0.3)
    fig.suptitle(f'{len(Z)} robosuite episode latents, zero labels '
                 f'({args.model})', fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight')
    fig.savefig(args.out.replace('.pdf', '.png'), dpi=160, bbox_inches='tight')
    print('saved', args.out)
