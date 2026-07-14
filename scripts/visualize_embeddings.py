"""UMAP visualization of task-description embeddings, colored by behavior factors.

Two spaces:
  1. raw cached embeddings (CLIP 512-d) — what the policy is given;
  2. the policy's trained text_encoder projection (64-d) — what actually
     conditions the denoiser (requires --checkpoint).

Each figure has one panel per factor (color / corner / side / speed) so factor
clustering can be judged independently.

Usage:
    python visualize_embeddings.py --out_dir ../output/embedding_viz
    python visualize_embeddings.py --checkpoint ../trained_models/dp_unet_clip_scripted_v1.pth
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import pickle
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap

FACTORS = {
    'color':  ['blue', 'red', 'green'],
    'corner': ['lower-right', 'upper-right', 'upper-left', 'lower-left'],
    'side':   ['clockwise', 'counterclockwise'],
    'speed':  ['gently', 'quickly'],
}
MARKERS = ['o', 's', '^', 'D']


def parse_factors(desc):
    out = {}
    for factor, values in FACTORS.items():
        out[factor] = None
        # counterclockwise contains clockwise — match longest value first
        for v in sorted(values, key=len, reverse=True):
            if re.search(rf'\b{re.escape(v)}\b', desc):
                out[factor] = v
                break
    return out


def factor_panels(emb2d, descs, title, out_path):
    fig, axes = plt.subplots(1, len(FACTORS), figsize=(5.2 * len(FACTORS), 5))
    for ax, (factor, values) in zip(axes, FACTORS.items()):
        palette = plt.get_cmap('tab10')
        for i, v in enumerate(values):
            idx = [j for j, d in enumerate(descs) if parse_factors(d)[factor] == v]
            if idx:
                ax.scatter(emb2d[idx, 0], emb2d[idx, 1], s=70, alpha=0.85,
                           color=palette(i), marker=MARKERS[i % len(MARKERS)], label=v)
        none_idx = [j for j, d in enumerate(descs) if parse_factors(d)[factor] is None]
        if none_idx:
            ax.scatter(emb2d[none_idx, 0], emb2d[none_idx, 1], s=30, alpha=0.3,
                       color='gray', marker='x', label='n/a')
        ax.set_title(f'colored by {factor}')
        ax.legend(fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cached_labels', default='../output/cached_labels.pkl')
    parser.add_argument('--checkpoint', default=None,
                        help='policy .pth; if set, also plot the trained 64-d projection')
    parser.add_argument('--out_dir', default='../output/embedding_viz')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.cached_labels, 'rb') as f:
        labels = pickle.load(f)
    descs = list(labels.keys())
    raw = np.concatenate([labels[d].detach().cpu().numpy() for d in descs], axis=0)
    print(f'{len(descs)} descriptions, raw dim {raw.shape[1]}')

    n_neighbors = min(8, len(descs) - 1)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.15,
                        n_components=2, random_state=args.seed)
    raw2d = reducer.fit_transform(raw)
    factor_panels(raw2d, descs, f'Raw CLIP text embeddings ({raw.shape[1]}-d) — UMAP',
                  os.path.join(args.out_dir, 'umap_raw_clip.png'))

    if args.checkpoint:
        import torch
        from diffusion_policy.models.diffusion_policy import DiffusionPolicy
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps' if torch.backends.mps.is_available() else 'cpu')
        policy = DiffusionPolicy(obs_horizon=2, pred_horizon=16, lowdim_obs_dim=2,
                                 action_dim=2, num_diffusion_iters=100, vision=True,
                                 cached_labels_path=args.cached_labels,
                                 noise_pred_net_type='unet')
        policy.to(device)
        policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
        policy.eval()
        with torch.no_grad():
            proj = policy.encode_text(descs).cpu().numpy()
        proj2d = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.15, n_components=2,
                           random_state=args.seed).fit_transform(proj)
        name = os.path.basename(args.checkpoint).replace('.pth', '')
        factor_panels(proj2d, descs,
                      f'Trained text-encoder projection (64-d) — {name} — UMAP',
                      os.path.join(args.out_dir, f'umap_projected_{name}.png'))
