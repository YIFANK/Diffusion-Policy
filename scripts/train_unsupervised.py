"""v3 — unsupervised behavior-mode discovery (proposal Variant B, auto-decoder).

No labels: every training episode gets a learnable 64-d latent z (an embedding
table optimized jointly with the policy). The policy is conditioned on z exactly
where the text embedding used to go, so ALL downstream machinery (delta probe,
factor probes, MAP inversion, score composition) applies unchanged.

After training, the z-table is probed against the ground-truth factor labels the
model never saw: does the unsupervised latent space organize by corner/side/speed,
and in the action-relevance order the supervised probes predict?

Usage:
    python train_unsupervised.py --lowdim --data_dir ../dataset_modes_60 --epochs 100
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusion_policy.models.diffusion_policy import DiffusionPolicy
from diffusion_policy.data.dataset import PushTImageDataset
from train_modes import build_split

Z_DIM = 64  # matches text_feature_dim so conditioning plumbing is unchanged

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_dir', default='../dataset_modes_60')
    parser.add_argument('--lowdim', action='store_true')
    parser.add_argument('--z_lr', type=float, default=1e-3)
    parser.add_argument('--z_reg', type=float, default=1e-4,
                        help='L2 penalty on z (proposal: penalize high-norm z)')
    parser.add_argument('--model_path', default='../trained_models/dp_unsup_v3.pth')
    parser.add_argument('--exclude_held_out', action='store_true',
                        help='train on the 14 non-held-out combos only (composition test)')
    parser.add_argument('--correlated', action='store_true',
                        help='control: only cw+gentle and ccw+fast combos (side and '
                             'speed perfectly correlated) — tests whether factorized '
                             'geometry requires factorial data')
    args = parser.parse_args()

    vision = not args.lowdim
    state_keys = ('agent', 'o1') if args.lowdim else ('agent',)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    train_paths, train_descs, held_paths, held_descs = build_split()
    if args.exclude_held_out:
        src_paths, src_descs = train_paths, train_descs
    else:
        src_paths, src_descs = train_paths + held_paths, train_descs + held_descs
    if args.correlated:
        keep = [i for i, p in enumerate(src_paths)
                if ('_cw_gentle' in p) or ('_ccw_fast' in p)]
        src_paths = [src_paths[i] for i in keep]
        src_descs = [src_descs[i] for i in keep]
        print(f'correlated control: {len(src_paths)} combos '
              f'(side and speed perfectly confounded)')
    paths = [p.replace('../dataset_modes/', args.data_dir.rstrip('/') + '/')
             for p in src_paths]
    descs = src_descs  # only used for post-hoc labels, not training

    dataset = PushTImageDataset(paths, descs, pred_horizon=16, obs_horizon=2,
                                action_horizon=8, state_keys=state_keys)
    n_eps = dataset.n_episodes
    print(f'{n_eps} episodes, {len(dataset)} samples')

    # ground-truth factor label per episode (for post-hoc probing ONLY):
    # each sample knows its episode index and its frame's task description
    ep_text = {}
    for i in range(len(dataset.episode_idx)):
        e = int(dataset.episode_idx[i])
        if e not in ep_text:
            ep_text[e] = str(dataset.normalized_train_data['text'][dataset.indices[i][0]])
    # episodes shorter than the sampling window produce no samples: their z
    # never receives gradients — label None and exclude from probes
    ep_desc = [ep_text.get(e) for e in range(n_eps)]
    print(f'{sum(d is None for d in ep_desc)} episodes with no samples (excluded from probes)')
    with open(args.model_path.replace('.pth', '_ep_labels.json'), 'w') as f:
        json.dump(ep_desc, f)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=2,
                                             shuffle=True, persistent_workers=True)

    policy = DiffusionPolicy(obs_horizon=2, pred_horizon=16,
                             lowdim_obs_dim=2 * len(state_keys), action_dim=2,
                             num_diffusion_iters=100, vision=vision,
                             cached_labels_path='../output/cached_labels.pkl',
                             noise_pred_net_type='unet').to(device)
    z_table = nn.Embedding(n_eps, Z_DIM).to(device)
    nn.init.normal_(z_table.weight, std=0.1)

    ema = EMAModel(parameters=policy.parameters(), power=0.75)
    text_params = set(id(p) for p in policy.text_encoder.parameters())
    other = [p for p in policy.parameters() if id(p) not in text_params]
    optimizer = torch.optim.AdamW([
        {'params': other, 'lr': 1e-4},
        {'params': z_table.parameters(), 'lr': args.z_lr},
    ], weight_decay=1e-6)
    lr_sched = get_scheduler('cosine', optimizer=optimizer, num_warmup_steps=500,
                             num_training_steps=len(dataloader) * args.epochs)

    with tqdm(range(args.epochs), desc='Epoch') as tglobal:
        for epoch in tglobal:
            losses = []
            for nbatch in dataloader:
                nimage = nbatch['image'][:, :2].to(device)
                nagent = nbatch['agent_pos'][:, :2].to(device)
                naction = nbatch['action'].to(device)
                z = z_table(nbatch['episode_idx'].to(device))
                loss = policy(nimage, nagent, naction, z, p_uncond=0.1)
                loss = loss + args.z_reg * z.pow(2).mean()
                loss.backward()
                optimizer.step(); optimizer.zero_grad(); lr_sched.step()
                ema.step(policy.parameters())
                losses.append(loss.item())
            tglobal.set_postfix(loss=float(np.mean(losses)))

    ema.copy_to(policy.parameters())
    torch.save(policy.state_dict(), args.model_path)
    np.save(args.model_path.replace('.pth', '_ztable.npy'),
            z_table.weight.detach().cpu().numpy())
    print(f'saved {args.model_path} and z-table ({n_eps} x {Z_DIM})')
