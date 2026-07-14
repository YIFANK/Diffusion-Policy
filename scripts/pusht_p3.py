"""P3: unsupervised behavior-mode discovery on the ORIGINAL human PushT
demonstrations (Chi et al., 206 episodes) — no scripting, no labels.

Pipeline: zarr -> per-episode learnable z (auto-decoder, state-based policy)
-> post-hoc factor labels measured from the data itself:
    speed:   episode duration (median split)
    side:    sign of the pusher's winding around the T before first contact
    contact: first-contact location in the T's local frame (stem vs crossbar)
-> probes + UMAP of the learned z-table against these labels.

The model never sees any label; labels exist only to interrogate the space.

Usage (on the cluster):
    python pusht_p3.py --zarr ../pusht_data/pusht/pusht_cchi_v7_replay.zarr --epochs 150
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusion_policy.models.diffusion_policy import DiffusionPolicy
from diffusion_policy.data.dataset import (create_sample_indices, sample_sequence,
                                           get_data_stats, normalize_data)
from mode_adherence import approach_winding

Z_DIM = 64
OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8


class PushTZarrDataset(torch.utils.data.Dataset):
    def __init__(self, zarr_path):
        import zarr
        z = zarr.open(zarr_path, mode='r')
        self.img = np.array(z['data']['img'])          # (N,96,96,3) float32
        if self.img.max() > 1.5:
            self.img = self.img / 255.0
        self.state = np.array(z['data']['state'])      # (N,5)
        self.action = np.array(z['data']['action'])    # (N,2)
        self.n_contacts = np.array(z['data']['n_contacts'])[:, 0]
        self.episode_ends = np.array(z['meta']['episode_ends'])
        self.n_episodes = len(self.episode_ends)

        self.indices = create_sample_indices(
            episode_ends=self.episode_ends, sequence_length=PRED_HORIZON,
            pad_before=OBS_HORIZON - 1, pad_after=ACTION_HORIZON - 1)
        ep_starts = np.concatenate([[0], self.episode_ends[:-1]])
        self.episode_idx = np.searchsorted(self.episode_ends, self.indices[:, 0],
                                           side='right').astype(np.int64)
        self.stats = {'agent_pos': get_data_stats(self.state),
                      'action': get_data_stats(self.action)}
        self.nstate = normalize_data(self.state, self.stats['agent_pos'])
        self.naction = normalize_data(self.action, self.stats['action'])
        self.train_data = {'agent_pos': self.nstate, 'action': self.naction,
                           'image': np.moveaxis(self.img, -1, 1)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        b0, b1, s0, s1 = self.indices[idx]
        ns = sample_sequence(self.train_data, PRED_HORIZON, b0, b1, s0, s1)
        return {'image': ns['image'][:OBS_HORIZON],
                'agent_pos': ns['agent_pos'][:OBS_HORIZON],
                'action': ns['action'],
                'episode_idx': self.episode_idx[idx]}


def posthoc_labels(ds):
    """Measure factor labels per episode from the raw data."""
    labels = []
    starts = np.concatenate([[0], ds.episode_ends[:-1]])
    for e in range(ds.n_episodes):
        s, t = starts[e], ds.episode_ends[e]
        agent = ds.state[s:t, :2]
        block = ds.state[s:t, 2:4]
        angle0 = ds.state[s, 4]
        dur = t - s
        # first contact from the recorded contact counts
        cidx = np.argmax(ds.n_contacts[s:t] > 0) if (ds.n_contacts[s:t] > 0).any() else dur - 1
        w = approach_winding(agent[:max(cidx, 2)], block[:max(cidx, 2)])
        # contact point in the T's local frame at first touch
        rel = agent[cidx] - block[cidx]
        c, s_ = np.cos(-angle0), np.sin(-angle0)
        local = np.array([c * rel[0] - s_ * rel[1], s_ * rel[0] + c * rel[1]])
        labels.append({'duration': int(dur), 'winding': float(w),
                       'contact_local_x': float(local[0]),
                       'contact_local_y': float(local[1])})
    dur = np.array([l['duration'] for l in labels])
    wind = np.array([l['winding'] for l in labels])
    cy = np.array([l['contact_local_y'] for l in labels])
    for l, d, w, y in zip(labels, dur, wind, cy):
        l['speed'] = 'slow' if d > np.median(dur) else 'fast'
        l['side'] = 'ccw' if w > 0 else 'cw'
        l['contact'] = 'crossbar' if y > np.median(cy) else 'stem'
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr', default='../pusht_data/pusht/pusht_cchi_v7_replay.zarr')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--z_lr', type=float, default=1e-3)
    parser.add_argument('--z_reg', type=float, default=1e-4)
    parser.add_argument('--model_path', default='../trained_models/pusht_unsup_p3.pth')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = PushTZarrDataset(args.zarr)
    print(f'{ds.n_episodes} episodes, {len(ds)} samples')
    labels = posthoc_labels(ds)
    with open(args.model_path.replace('.pth', '_labels.json'), 'w') as f:
        json.dump(labels, f)

    dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=2,
                                     shuffle=True, persistent_workers=True)
    policy = DiffusionPolicy(obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
                             lowdim_obs_dim=5, action_dim=2, num_diffusion_iters=100,
                             vision=False, cached_labels_path='../output/cached_labels.pkl',
                             noise_pred_net_type='unet').to(device)
    z_table = nn.Embedding(ds.n_episodes, Z_DIM).to(device)
    nn.init.normal_(z_table.weight, std=0.1)
    ema = EMAModel(parameters=policy.parameters(), power=0.75)
    text_params = set(id(p) for p in policy.text_encoder.parameters())
    other = [p for p in policy.parameters() if id(p) not in text_params]
    opt = torch.optim.AdamW([{'params': other, 'lr': 1e-4},
                             {'params': z_table.parameters(), 'lr': args.z_lr}],
                            weight_decay=1e-6)
    sched = get_scheduler('cosine', optimizer=opt, num_warmup_steps=500,
                          num_training_steps=len(dl) * args.epochs)

    with tqdm(range(args.epochs), desc='Epoch') as tg:
        for ep in tg:
            losses = []
            for b in dl:
                nimage = b['image'][:, :OBS_HORIZON].to(device, dtype=torch.float32)
                nagent = b['agent_pos'][:, :OBS_HORIZON].to(device, dtype=torch.float32)
                naction = b['action'].to(device, dtype=torch.float32)
                z = z_table(b['episode_idx'].to(device))
                loss = policy(nimage, nagent, naction, z, p_uncond=0.1)
                loss = loss + args.z_reg * z.pow(2).mean()
                loss.backward(); opt.step(); opt.zero_grad(); sched.step()
                ema.step(policy.parameters())
                losses.append(loss.item())
            tg.set_postfix(loss=float(np.mean(losses)))

    ema.copy_to(policy.parameters())
    torch.save(policy.state_dict(), args.model_path)
    Z = z_table.weight.detach().cpu().numpy()
    np.save(args.model_path.replace('.pth', '_ztable.npy'), Z)

    # ---- probes ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    def knn(X, y):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        S = Xn @ Xn.T; np.fill_diagonal(S, -9)
        idx = list(range(len(y)))
        return sum(y[i] == y[max((j for j in idx if j != i), key=lambda j: S[i, j])]
                   for i in idx) / len(idx)
    print('P3_PROBES factor chance 1-NN linear')
    for f in ['speed', 'side', 'contact']:
        y = [l[f] for l in labels]
        base = max(np.mean([v == y[0] for v in y]), 0.5)
        acc = cross_val_score(LogisticRegression(max_iter=3000), Z, y, cv=5).mean()
        print(f'P3_PROBE {f}: chance~0.5 knn {knn(Z, y):.2f} linear {acc:.2f}', flush=True)
    print('P3_DONE', flush=True)
