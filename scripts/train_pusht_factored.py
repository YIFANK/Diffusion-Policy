"""Train a text-conditioned STATE-based diffusion policy on the factored
(rotation x speed) PushT dataset.

The 5-d state carries no goal information, so the rotation variant is a true
conditioning factor. Descriptions: "push the T to the goal rotated {0,90,180,270}
degrees, {quickly,slowly}".

Usage:
    python train_pusht_factored.py --data_dir ../pusht_data/factored --epochs 100
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from tqdm import tqdm
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusion_policy.models.diffusion_policy import DiffusionPolicy
from diffusion_policy.data.dataset import (create_sample_indices, sample_sequence,
                                           get_data_stats, normalize_data)

OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8

ROT_DEG = {0: '0', 1: '90', 2: '180', 3: '270'}
SPEED_WORD = {'fast': 'quickly', 'slow': 'slowly'}


def desc_for(k, speed):
    return (f'push the T to the goal rotated {ROT_DEG[k]} degrees, '
            f'{SPEED_WORD[speed]}')


class FactoredPushT(torch.utils.data.Dataset):
    def __init__(self, data_dir, combos, held_out=()):
        S, A, E, texts = [], [], [], []
        off = 0
        for (k, speed) in combos:
            if (k, speed) in held_out:
                continue
            d = np.load(os.path.join(data_dir, f'rot{k}_{speed}.npz'))
            S.append(d['state']); A.append(d['action'])
            E.append(d['episode_ends'] + off)
            texts += [desc_for(k, speed)] * len(d['state'])
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
        # dummy image (policy is state-based; forward() reads shape[0] only)
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


ALL_COMBOS = [(k, sp) for k in range(4) for sp in ['fast', 'slow']]
HELD_OUT_PUSHT = [(2, 'slow'), (1, 'fast')]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../pusht_data/factored')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exclude_held_out', action='store_true')
    parser.add_argument('--text_lr', type=float, default=1e-4)
    parser.add_argument('--model_path', default='../trained_models/pusht_factored_v1.pth')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    held = HELD_OUT_PUSHT if args.exclude_held_out else ()
    ds = FactoredPushT(args.data_dir, ALL_COMBOS, held_out=held)
    print(f'{len(ds)} samples, held out: {held}')
    dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=2,
                                     shuffle=True, persistent_workers=True)

    policy = DiffusionPolicy(obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
                             lowdim_obs_dim=5, action_dim=2, num_diffusion_iters=100,
                             vision=False, cached_labels_path='../output/cached_labels.pkl',
                             noise_pred_net_type='unet').to(device)
    ema = EMAModel(parameters=policy.parameters(), power=0.75)
    text_params = list(policy.text_encoder.parameters())
    tp_ids = set(id(p) for p in text_params)
    other = [p for p in policy.parameters() if id(p) not in tp_ids]
    opt = torch.optim.AdamW([{'params': other, 'lr': 1e-4},
                             {'params': text_params, 'lr': args.text_lr}],
                            weight_decay=1e-6)
    sched = get_scheduler('cosine', optimizer=opt, num_warmup_steps=500,
                          num_training_steps=len(dl) * args.epochs)

    with tqdm(range(args.epochs), desc='Epoch') as tg:
        for ep in tg:
            losses = []
            for b in dl:
                nimage = b['image'].to(device, dtype=torch.float32)
                nagent = b['agent_pos'].to(device, dtype=torch.float32)
                naction = b['action'].to(device, dtype=torch.float32)
                loss = policy(nimage, nagent, naction, list(b['text']), p_uncond=0.1)
                loss.backward(); opt.step(); opt.zero_grad(); sched.step()
                ema.step(policy.parameters())
                losses.append(loss.item())
            tg.set_postfix(loss=float(np.mean(losses)))
    ema.copy_to(policy.parameters())
    torch.save(policy.state_dict(), args.model_path)
    import json
    with open(args.model_path.replace('.pth', '_stats.json'), 'w') as f:
        json.dump({k: {kk: vv.tolist() for kk, vv in v.items()}
                   for k, v in ds.stats.items()}, f)
    print(f'saved {args.model_path}')
