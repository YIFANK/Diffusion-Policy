"""E3: causal intervention — same initial state, swap only the conditioning z.

For each fixed env seed (= identical initial layout), roll out once per corner
description. If z causally controls behavior, the achieved corner should follow
the commanded corner across the whole grid. Reports the 4x4 confusion matrix
(commanded corner x achieved corner) and intervention accuracy.

Uses trained (non-held-out) mode descriptions on the healthy lowdim-60 host.

Usage:
    python eval_e3.py --model_path ../trained_models/dp_lowdim_modes_60ep.pth --lowdim
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import collections
import json
import numpy as np
import torch
from omegaconf import OmegaConf
from tiny_embodied_reasoning.environment import env as ter_env
from tiny_embodied_reasoning.observers.observer import StateObserver, ImageObserver
from diffusion_policy.models.diffusion_policy import DiffusionPolicy
from diffusion_policy.data.dataset import PushTImageDataset, normalize_data, unnormalize_data
from generate_demos import ENV_YAML, CORNER_NAMES, WIN_CONDITION
from train_modes import build_split
from cache_text_embeddings import SIDE_PHRASES, SPEED_PHRASES

OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8
# per corner, a (side, speed) that is in the TRAINING split for that corner
TRAINED_MODE = {0: ('cw', 'gentle'), 1: ('cw', 'fast'),
                2: ('cw', 'fast'), 3: ('cw', 'fast')}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def corner_desc(corner):
    side, speed = TRAINED_MODE[corner]
    return (f'push the blue block to the {CORNER_NAMES[corner]} corner, '
            f'{SIDE_PHRASES[side]}, {SPEED_PHRASES[speed]}')


def achieved_corner(block_pos):
    for c, win in enumerate(WIN_CONDITION):
        if win(block_pos):
            return c
    return None


def rollout(policy, stats, commanded_corner, seed, state_keys, max_steps=500):
    cfg = OmegaConf.create(ENV_YAML.format(color='Blue'))
    env = ter_env.TEREnv(**cfg.info, scene_info=cfg.scene_info,
                         agent_info=cfg.agent_info, verbose=False)
    env.set_seed(seed)
    env._setup()
    imgo = ImageObserver(env, render_size=96, verbose=False)
    sto = StateObserver(env, verbose=False)
    desc = corner_desc(commanded_corner)

    def observe():
        st = sto.observe()
        lowdim = np.concatenate([
            np.asarray(env.agent.position, dtype=np.float32) if k == 'agent'
            else np.asarray(st[k]['position'], dtype=np.float32)[:2]
            for k in state_keys])
        return {'image': imgo.observe(), 'agent_pos': lowdim}

    dq = collections.deque([observe()] * OBS_HORIZON, maxlen=OBS_HORIZON)
    for step in range(0, max_steps, ACTION_HORIZON):
        images = np.stack([x['image'] for x in dq]).astype(np.float32) / 255.0
        nagent = normalize_data(np.stack([x['agent_pos'] for x in dq]),
                                stats=stats['agent_pos'])
        ni = torch.from_numpy(images).permute(0, 3, 1, 2).unsqueeze(0).to(device, dtype=torch.float32)
        na = torch.from_numpy(nagent).unsqueeze(0).to(device, dtype=torch.float32)
        with torch.no_grad():
            act = policy.sample(nimages=ni, nagent_poses=na, ntexts=[desc],
                                num_diffusion_iters=100, n_samples=1)
        acts = unnormalize_data(act.cpu().numpy()[0], stats=stats['action'])
        for a in acts[OBS_HORIZON - 1: OBS_HORIZON - 1 + ACTION_HORIZON]:
            env.step(a)
            dq.append(observe())
            st = sto.observe()
            got = achieved_corner(np.array(st['o1']['position']))
            if got is not None:
                return got
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/dp_lowdim_modes_60ep.pth')
    parser.add_argument('--lowdim', action='store_true')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--data_dir', default='../dataset_modes_60')
    args = parser.parse_args()
    state_keys = ('agent', 'o1') if args.lowdim else ('agent',)

    train_paths, train_descs, _, _ = build_split()
    train_paths = [p.replace('../dataset_modes/', args.data_dir.rstrip('/') + '/')
                   for p in train_paths]
    stats = PushTImageDataset(train_paths, train_descs, pred_horizon=PRED_HORIZON,
                              obs_horizon=OBS_HORIZON, action_horizon=ACTION_HORIZON,
                              state_keys=state_keys).stats

    policy = DiffusionPolicy(obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
                             lowdim_obs_dim=2 * len(state_keys), action_dim=2,
                             num_diffusion_iters=100, vision=not args.lowdim,
                             cached_labels_path='../output/cached_labels.pkl',
                             noise_pred_net_type='unet').to(device)
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    policy.eval()

    confusion = np.zeros((4, 5), dtype=int)  # commanded x (achieved 0-3 | none)
    correct = 0
    for seed in range(9000, 9000 + args.seeds):
        for commanded in range(4):
            got = rollout(policy, stats, commanded, seed, state_keys)
            confusion[commanded, 4 if got is None else got] += 1
            correct += int(got == commanded)
            print(f'E3_RESULT seed={seed} commanded={CORNER_NAMES[commanded]} '
                  f'achieved={CORNER_NAMES[got] if got is not None else "none"}', flush=True)

    total = 4 * args.seeds
    out = {'model': os.path.basename(args.model_path),
           'accuracy': correct / total,
           'confusion_rows_commanded': confusion.tolist(),
           'columns': CORNER_NAMES + ['none']}
    with open(args.model_path.replace('.pth', '_e3_eval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('E3_DONE', json.dumps(out), flush=True)
