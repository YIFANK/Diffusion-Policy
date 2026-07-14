"""Train a text-conditioned STATE-based diffusion policy on factored
robosuite PickPlaceCan demonstrations (side x arc x speed).

State (7-d): eef xyz, can xyz, gripper width. Action (7-d): OSC_POSE + gripper.

Usage:
    python train_robosuite_factored.py --data_dir ../robosuite_data --epochs 150
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusion_policy.models.diffusion_policy import DiffusionPolicy
from diffusion_policy.data.dataset import (create_sample_indices, sample_sequence,
                                           get_data_stats, normalize_data)

OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8

ARC_PHRASE = {'direct': 'directly', 'high': 'via a high arc'}
SPEED_PHRASE = {'fast': 'quickly', 'slow': 'slowly'}
ALL_COMBOS = [(s, a, sp) for s in ['left', 'right']
              for a in ['direct', 'high'] for sp in ['fast', 'slow']]
HELD_OUT_RS = [('left', 'high', 'fast'), ('right', 'direct', 'slow')]


def desc_for(s, a, sp):
    return (f'pick the can approaching from the {s} and place it in the bin, '
            f'{ARC_PHRASE[a]}, {SPEED_PHRASE[sp]}')


class FactoredRS(torch.utils.data.Dataset):
    def __init__(self, data_dir, held_out=()):
        import glob
        S, A, E, texts = [], [], [], []
        self.ep_combo = []  # combo string per episode (for unsupervised probing)
        off = 0
        for (s, a, sp) in ALL_COMBOS:
            if (s, a, sp) in held_out:
                continue
            # batch 1 ({combo}.npz) plus any later batches ({combo}_b2.npz, ...)
            files = sorted(glob.glob(os.path.join(data_dir, f'{s}_{a}_{sp}.npz')) +
                           glob.glob(os.path.join(data_dir, f'{s}_{a}_{sp}_b*.npz')))
            for fp in files:
                d = np.load(fp)
                S.append(d['state']); A.append(d['action'])
                E.append(d['episode_ends'] + off)
                texts += [desc_for(s, a, sp)] * len(d['state'])
                self.ep_combo += [f'{s}_{a}_{sp}'] * len(d['episode_ends'])
                off += len(d['state'])
        self.state = np.concatenate(S); self.action = np.concatenate(A)
        self.episode_ends = np.concatenate(E)
        self.text = np.array(texts)
        self.indices = create_sample_indices(self.episode_ends, PRED_HORIZON,
                                             OBS_HORIZON - 1, ACTION_HORIZON - 1)
        self.stats = {'agent_pos': get_data_stats(self.state),
                      'action': get_data_stats(self.action)}
        self.train_data = {
            'agent_pos': normalize_data(self.state, self.stats['agent_pos']),
            'action': normalize_data(self.action, self.stats['action'])}
        self.dummy_img = np.zeros((PRED_HORIZON, 3, 8, 8), dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        b0, b1, s0, s1 = self.indices[idx]
        ns = sample_sequence(self.train_data, PRED_HORIZON, b0, b1, s0, s1)
        return {'image': self.dummy_img[:OBS_HORIZON],
                'agent_pos': ns['agent_pos'][:OBS_HORIZON],
                'action': ns['action'],
                'text': str(self.text[b0])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../robosuite_data')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--exclude_held_out', action='store_true')
    parser.add_argument('--text_lr', type=float, default=1e-4)
    parser.add_argument('--model_path', default='../trained_models/rs_factored_v1.pth')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    held = HELD_OUT_RS if args.exclude_held_out else ()
    ds = FactoredRS(args.data_dir, held_out=held)
    print(f'{len(ds)} samples, held out: {held}')
    dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=2,
                                     shuffle=True, persistent_workers=True)

    policy = DiffusionPolicy(obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
                             lowdim_obs_dim=7, action_dim=7, num_diffusion_iters=100,
                             vision=False, cached_labels_path='../output/cached_labels.pkl',
                             noise_pred_net_type='unet').to(device)
    ema = EMAModel(parameters=policy.parameters(), power=0.75)
    tp = list(policy.text_encoder.parameters())
    tp_ids = set(id(p) for p in tp)
    other = [p for p in policy.parameters() if id(p) not in tp_ids]
    opt = torch.optim.AdamW([{'params': other, 'lr': 1e-4},
                             {'params': tp, 'lr': args.text_lr}], weight_decay=1e-6)
    sched = get_scheduler('cosine', optimizer=opt, num_warmup_steps=500,
                          num_training_steps=len(dl) * args.epochs)

    with tqdm(range(args.epochs), desc='Epoch') as tg:
        for ep in tg:
            losses = []
            for b in dl:
                loss = policy(b['image'].to(device, dtype=torch.float32),
                              b['agent_pos'].to(device, dtype=torch.float32),
                              b['action'].to(device, dtype=torch.float32),
                              list(b['text']), p_uncond=0.1)
                loss.backward(); opt.step(); opt.zero_grad(); sched.step()
                ema.step(policy.parameters())
                losses.append(loss.item())
            tg.set_postfix(loss=float(np.mean(losses)))
    ema.copy_to(policy.parameters())
    torch.save(policy.state_dict(), args.model_path)
    with open(args.model_path.replace('.pth', '_stats.json'), 'w') as f:
        json.dump({k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()}
                   for k, v in ds.stats.items()}, f)
    print(f'saved {args.model_path}', flush=True)
