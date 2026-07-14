"""Text-pathway atrophy metric: ||eps(x,o,z) - eps(x,o,none)|| across conditions.

For a healthy conditional policy the delta should be (a) clearly nonzero and
(b) different across z. An atrophied text pathway gives delta ~ 0 for every z
(the mechanistic signature behind CFG-scale invariance).

Reports, over sampled (state, diffusion-step) pairs:
  - mean ||delta_z|| per condition z
  - mean pairwise ||eps_z1 - eps_z2|| between different conditions
  - reference scale: mean ||eps|| itself

Usage:
    python measure_cond_delta.py --model_path ../trained_models/dp_unet_clip_twoobj_v4.pth \
        --descs "push the blue block to the lower-right corner" \
                "push the red block to the lower-right corner" \
        --data ../dataset_twoobj/TwoObj_blue.pkl
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import json
import numpy as np
import torch
from diffusion_policy.models.diffusion_policy import DiffusionPolicy
from diffusion_policy.data.dataset import PushTImageDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--descs', nargs='+', required=True)
    parser.add_argument('--data', required=True,
                        help='a demo pkl supplying observation states to probe at')
    parser.add_argument('--n_states', type=int, default=64)
    parser.add_argument('--lowdim', action='store_true')
    parser.add_argument('--keys', nargs='+', default=None,
                        help="state keys for lowdim obs, e.g. --keys agent o1 o2")
    parser.add_argument('--text_gate', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    vision = not args.lowdim
    if args.keys:
        state_keys = tuple(args.keys)
    else:
        state_keys = ('agent', 'o1') if args.lowdim else ('agent',)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    policy = DiffusionPolicy(obs_horizon=2, pred_horizon=16,
                             lowdim_obs_dim=2 * len(state_keys), action_dim=2,
                             num_diffusion_iters=100, vision=vision,
                             cached_labels_path='../output/cached_labels.pkl',
                             noise_pred_net_type='unet',
                             text_gate=args.text_gate).to(device)
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    policy.eval()

    ds = PushTImageDataset([args.data], [args.descs[0]], pred_horizon=16,
                           obs_horizon=2, action_horizon=8, state_keys=state_keys)
    idx = np.random.RandomState(args.seed).choice(len(ds), args.n_states, replace=False)

    deltas = {d: [] for d in args.descs}
    pair = []
    eps_norm = []
    with torch.no_grad():
        for i in idx:
            s = ds[int(i)]
            nimage = torch.from_numpy(s['image'][None]).to(device, dtype=torch.float32)
            nagent = torch.from_numpy(s['agent_pos'][None]).to(device, dtype=torch.float32)
            naction = torch.from_numpy(s['action'][None]).to(device, dtype=torch.float32)
            noise = torch.randn_like(naction)
            t = torch.randint(0, 100, (1,), device=device).long()
            x_t = policy.noise_scheduler.add_noise(naction, noise, t)

            cond_un = policy.get_cond(nimage, nagent, None, uncond=True)
            eps_un = policy.noise_pred_net(x_t, t, global_cond=cond_un)
            eps_norm.append(eps_un.norm().item())

            eps_z = {}
            for d in args.descs:
                c = policy.get_cond(nimage, nagent, [d], uncond=False)
                e = policy.noise_pred_net(x_t, t, global_cond=c)
                eps_z[d] = e
                deltas[d].append((e - eps_un).norm().item())
            ds_list = list(args.descs)
            for a in range(len(ds_list)):
                for b in range(a + 1, len(ds_list)):
                    pair.append((eps_z[ds_list[a]] - eps_z[ds_list[b]]).norm().item())

    report = {
        'model': os.path.basename(args.model_path),
        'mean_eps_norm': float(np.mean(eps_norm)),
        'mean_delta_per_desc': {d: float(np.mean(v)) for d, v in deltas.items()},
        'mean_pairwise_between_desc': float(np.mean(pair)) if pair else None,
    }
    print('COND_DELTA', json.dumps(report, indent=2), flush=True)
