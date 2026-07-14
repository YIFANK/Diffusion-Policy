"""Train a text-conditioned state-based diffusion policy on paired StackThree
demos. State 22-d (eef, grip, 3x cube pos+rgb), action 7-d, CLIP text.

Usage:
    python train_stack.py --data ../robosuite_stack_data/train.npz --epochs 150
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
from robosuite_stack import desc_stack, rel22

OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8


class StackDS(torch.utils.data.Dataset):
    def __init__(self, npz_path, relative=False):
        d = np.load(npz_path)
        meta = json.load(open(npz_path.replace('.npz', '_meta.json')))
        self.state, self.action = d['state'], d['action']
        if relative:
            self.state = rel22(self.state)
        self.episode_ends = d['episode_ends']
        texts = np.empty(len(self.state), dtype=object)
        start = 0
        for m, end in zip(meta, self.episode_ends):
            texts[start:end] = desc_stack(m['src'], m['tgt'])
            start = end
        self.text = texts
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
    parser.add_argument('--data', default='../robosuite_stack_data/train.npz')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--text_lr', type=float, default=1e-4)
    parser.add_argument('--model_path', default='../trained_models/stack_v1.pth')
    parser.add_argument('--relative', action='store_true',
                        help='object-centric obs: cube positions eef-relative')
    parser.add_argument('--constrained', action='store_true',
                        help='mark model as trained on zone-constrained layouts '
                             '(eval will use the same placement)')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = StackDS(args.data, relative=args.relative)
    print(f'{len(ds)} samples, {len(ds.episode_ends)} episodes')
    dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=2,
                                     shuffle=True, persistent_workers=True)

    policy = DiffusionPolicy(obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
                             lowdim_obs_dim=22, action_dim=7, num_diffusion_iters=100,
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
        out = {k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()}
               for k, v in ds.stats.items()}
        out['relative'] = bool(args.relative)
        out['constrained'] = bool(args.constrained)
        json.dump(out, f)
    print(f'STACKTRAIN_DONE {args.model_path}', flush=True)
