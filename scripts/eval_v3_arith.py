"""v3 arithmetic steering: does z-space vector arithmetic causally steer behavior?

For each corner: take the centroid z of (corner, cw, fast) episodes, then add the
"gentle − fast" direction computed from OTHER corners only. Roll out both.
If the arithmetic works behaviorally, the shifted z should (a) keep reaching the
corner and (b) take substantially more control steps (the gentle signature —
gentle demos run ~2x longer than fast ones).

Usage:
    python eval_v3_arith.py --model_path ../trained_models/dp_unsup_v3.pth --lowdim
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

OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rollout(policy, stats, z, corner, seed, state_keys, max_steps=600):
    cfg = OmegaConf.create(ENV_YAML.format(color='Blue'))
    env = ter_env.TEREnv(**cfg.info, scene_info=cfg.scene_info,
                         agent_info=cfg.agent_info, verbose=False)
    env.set_seed(seed)
    env._setup()
    imgo = ImageObserver(env, render_size=96, verbose=False)
    sto = StateObserver(env, verbose=False)
    zt = torch.from_numpy(z[None]).to(device, dtype=torch.float32)

    def observe():
        st = sto.observe()
        lowdim = np.concatenate([
            np.asarray(env.agent.position, dtype=np.float32) if k == 'agent'
            else np.asarray(st[k]['position'], dtype=np.float32)[:2]
            for k in state_keys])
        return {'image': imgo.observe(), 'agent_pos': lowdim}

    dq = collections.deque([observe()] * OBS_HORIZON, maxlen=OBS_HORIZON)
    steps = 0
    win = WIN_CONDITION[corner]
    for _ in range(0, max_steps, ACTION_HORIZON):
        images = np.stack([x['image'] for x in dq]).astype(np.float32) / 255.0
        nagent = normalize_data(np.stack([x['agent_pos'] for x in dq]),
                                stats=stats['agent_pos'])
        ni = torch.from_numpy(images).permute(0, 3, 1, 2).unsqueeze(0).to(device, dtype=torch.float32)
        na = torch.from_numpy(nagent).unsqueeze(0).to(device, dtype=torch.float32)
        with torch.no_grad():
            act = policy.sample(nimages=ni, nagent_poses=na, ntexts=zt,
                                num_diffusion_iters=100, n_samples=1)
        acts = unnormalize_data(act.cpu().numpy()[0], stats=stats['action'])
        for a in acts[OBS_HORIZON - 1: OBS_HORIZON - 1 + ACTION_HORIZON]:
            env.step(a)
            dq.append(observe())
            steps += 1
            st = sto.observe()
            if win(np.array(st['o1']['position'])):
                return True, steps
    return False, steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/dp_unsup_v3.pth')
    parser.add_argument('--lowdim', action='store_true')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--data_dir', default='../dataset_modes_60')
    args = parser.parse_args()
    state_keys = ('agent', 'o1') if args.lowdim else ('agent',)

    # z centroids per combo from the trained table
    Z = np.load(args.model_path.replace('.pth', '_ztable.npy'))
    descs = json.load(open(args.model_path.replace('.pth', '_ep_labels.json')))
    def parse(d):
        if d is None: return None
        c = next(i for i, x in enumerate(CORNER_NAMES) if x in d)
        s = 'ccw' if 'counterclockwise' in d else 'cw'
        sp = 'gentle' if 'gently' in d else 'fast'
        return (c, s, sp)
    from collections import defaultdict
    groups = defaultdict(list)
    for z, d in zip(Z, descs):
        p = parse(d)
        if p: groups[p].append(z)
    cent = {k: np.mean(v, axis=0) for k, v in groups.items()}

    # normalization stats (same data the model trained on)
    tp, td, hp, hd = build_split()
    paths = [p.replace('../dataset_modes/', args.data_dir.rstrip('/') + '/')
             for p in tp + hp]
    stats = PushTImageDataset(paths, td + hd, pred_horizon=PRED_HORIZON,
                              obs_horizon=OBS_HORIZON, action_horizon=ACTION_HORIZON,
                              state_keys=state_keys).stats

    policy = DiffusionPolicy(obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
                             lowdim_obs_dim=2 * len(state_keys), action_dim=2,
                             num_diffusion_iters=100, vision=not args.lowdim,
                             cached_labels_path='../output/cached_labels.pkl',
                             noise_pred_net_type='unet').to(device)
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    policy.eval()

    results = {}
    for corner in range(4):
        z_fast = cent[(corner, 'cw', 'fast')]
        # gentle-fast direction from OTHER corners only
        dirs = [cent[(c, s, 'gentle')] - cent[(c, s, 'fast')]
                for c in range(4) if c != corner for s in ['cw', 'ccw']
                if (c, s, 'gentle') in cent and (c, s, 'fast') in cent]
        gentle_dir = np.mean(dirs, axis=0)
        z_arith = z_fast + gentle_dir

        for name, z in [('fast_centroid', z_fast), ('plus_gentle_dir', z_arith)]:
            outs = [rollout(policy, stats, z.astype(np.float32), corner, 5000 + i, state_keys)
                    for i in range(args.episodes)]
            succ = sum(o[0] for o in outs)
            steps = [o[1] for o in outs if o[0]]
            results[f'{CORNER_NAMES[corner]}/{name}'] = {
                'success': succ, 'mean_steps': float(np.mean(steps)) if steps else None}
            print(f'V3A_RESULT {CORNER_NAMES[corner]}/{name}: {succ}/{args.episodes} '
                  f'mean_steps={np.mean(steps):.0f}' if steps else
                  f'V3A_RESULT {CORNER_NAMES[corner]}/{name}: {succ}/{args.episodes}', flush=True)

    with open(args.model_path.replace('.pth', '_arith_eval.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('V3A_DONE', json.dumps(results), flush=True)
