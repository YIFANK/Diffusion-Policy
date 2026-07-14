"""Few-shot adaptation with a geometry prior (K x {cold, warm} matrix).

Question: direct held-out inference fails, but the unsupervised space gives us
a strong prior (arithmetic composition point). Does warm-starting few-shot MAP
inversion at that prior fix adaptation -- especially at K=1, where cold
inversion is known to be unstable?

Arms per held-out combo: K in {1,4} x {cold (z=0 init), warm (z_arith init +
proximal 0.01)}. Metrics: goal success, steps signature.

Usage:
    python eval_v3_prior.py --model_path ../trained_models/dp_unsup_v3_holdout.pth --lowdim
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import numpy as np
import torch
from collections import defaultdict
from diffusion_policy.models.diffusion_policy import DiffusionPolicy
from diffusion_policy.data.dataset import PushTImageDataset
from generate_demos import CORNER_NAMES
from train_modes import build_split
from eval_v3_arith import rollout
from eval_v3_holdout import map_invert, HELD

OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/dp_unsup_v3_holdout.pth')
    parser.add_argument('--lowdim', action='store_true')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--data_dir', default='../dataset_modes_60')
    parser.add_argument('--prior_weight', type=float, default=0.01)
    args = parser.parse_args()
    state_keys = ('agent', 'o1') if args.lowdim else ('agent',)

    Z = np.load(args.model_path.replace('.pth', '_ztable.npy'))
    descs = json.load(open(args.model_path.replace('.pth', '_ep_labels.json')))
    def parse(d):
        if d is None: return None
        return (next(i for i, x in enumerate(CORNER_NAMES) if x in d),
                'ccw' if 'counterclockwise' in d else 'cw',
                'gentle' if 'gently' in d else 'fast')
    groups = defaultdict(list)
    for zz, d in zip(Z, descs):
        p = parse(d)
        if p: groups[p].append(zz)
    cent = {k: np.mean(v, axis=0) for k, v in groups.items()}

    def gentle_dir(exclude_corner):
        ds_ = [cent[(c, s, 'gentle')] - cent[(c, s, 'fast')]
               for c in range(4) if c != exclude_corner for s in ['cw', 'ccw']
               if (c, s, 'gentle') in cent and (c, s, 'fast') in cent]
        return np.mean(ds_, axis=0)

    cA, cB = HELD['A'][0], HELD['B'][0]
    z_arith = {'A': cent[(cA, 'cw', 'fast')] + gentle_dir(cA),
               'B': cent[(cB, 'ccw', 'gentle')] - gentle_dir(cB)}
    demo = {'A': f'{args.data_dir}/Blue_{cA}_cw_gentle.pkl',
            'B': f'{args.data_dir}/Blue_{cB}_ccw_fast.pkl'}

    tp, td, _, _ = build_split()
    paths = [p.replace('../dataset_modes/', args.data_dir.rstrip('/') + '/') for p in tp]
    stats = PushTImageDataset(paths, td, pred_horizon=PRED_HORIZON,
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
    for tag in ['A', 'B']:
        combo = HELD[tag]
        for k in [1, 4]:
            for init_name in ['cold', 'warm']:
                z = map_invert(policy, demo[tag], state_keys, k_shot=k,
                               z_init=z_arith[tag] if init_name == 'warm' else None,
                               prior_weight=args.prior_weight if init_name == 'warm' else 0.0)
                outs = [rollout(policy, stats, z.astype(np.float32), combo[0],
                                7500 + i, state_keys) for i in range(args.episodes)]
                succ = sum(o[0] for o in outs)
                steps = [o[1] for o in outs if o[0]]
                key = f'{tag}/K{k}_{init_name}'
                results[key] = {'success': succ,
                                'mean_steps': float(np.mean(steps)) if steps else None}
                ms = f' mean_steps={np.mean(steps):.0f}' if steps else ''
                print(f'V3P_RESULT {key} ({CORNER_NAMES[combo[0]]},{combo[1]},{combo[2]}): '
                      f'{succ}/{args.episodes}{ms}', flush=True)

    with open(args.model_path.replace('.pth', '_prior_eval.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('V3P_DONE', json.dumps(results), flush=True)
