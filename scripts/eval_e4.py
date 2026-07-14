"""E4: held-out composition of behavior-mode concepts on the frozen v2 policy.

The v2 policy never saw the held-out combos (see train_modes.HELD_OUT). For each
held-out combo (corner, side, speed) we:
  1. infer one concept per factor, each from demos that share ONLY that factor
     (e.g. z_side from cw demos across *other* corners/speeds),
  2. compose the factor concepts at sampling time (score-space sum, the same
     mechanism as training: eps = eps_c0 + sum_k w_k (eps_ck - eps_c0)),
  3. evaluate closed-loop on the held-out combo's mode-consistency proxy:
     task success at the held-out corner.

Baselines run alongside:
  - text oracle: the held-out combo's own description (policy never trained on
    it, but the description is composable text),
  - single-factor ablations: corner concept alone.

Usage:
    python eval_e4.py --model_path ../trained_models/dp_unet_clip_modes_v2.pth --episodes 10
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import numpy as np
from concept_inference import infer_new_concepts
from inference import evaluate
from train_modes import HELD_OUT, CORNER_NAMES
from cache_text_embeddings import SIDE_PHRASES, SPEED_PHRASES

SIDES = ['cw', 'ccw']
SPEEDS = ['gentle', 'fast']
COLOR = 'Blue'


def factor_paths(corner, side, speed):
    """Demo sets sharing exactly one factor with the held-out combo, drawn only
    from TRAINING combos (never the held-out one)."""
    held = (corner, side, speed)

    def ok(c, si, sp):
        return (c, si, sp) not in HELD_OUT

    corner_paths = [f'../dataset_modes/{COLOR}_{corner}_{si}_{sp}.pkl'
                    for si in SIDES for sp in SPEEDS
                    if (si, sp) != (side, speed) and ok(corner, si, sp)]
    side_paths = [f'../dataset_modes/{COLOR}_{c}_{side}_{sp}.pkl'
                  for c in range(4) for sp in SPEEDS
                  if c != corner and ok(c, side, sp)]
    speed_paths = [f'../dataset_modes/{COLOR}_{c}_{si}_{speed}.pkl'
                   for c in range(4) for si in SIDES
                   if c != corner and ok(c, si, speed)]
    return {'corner': corner_paths, 'side': side_paths, 'speed': speed_paths}


def infer_factor(model_path, tag, paths, epochs, n_traj_per_set,
                 vision=True, state_keys=('agent',)):
    learned = infer_new_concepts(
        model_path=model_path, policy_type='diffusion',
        new_concept_dataset_path=paths,
        weights_output_path=f'../output/concepts/e4_{tag}_weights.npy',
        embeddings_output_path=f'../output/concepts/e4_{tag}_embeddings.npy',
        num_epochs=epochs, learning_rate=1e-2, K=1, logging=False,
        init_type='rand', num_trajectories=n_traj_per_set,
        vision=vision, state_keys=state_keys)
    return learned['embeddings'][0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/dp_unet_clip_modes_v2.pth')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--inversion_epochs', type=int, default=200)
    parser.add_argument('--n_traj_per_set', type=int, default=4)
    parser.add_argument('--w', type=float, default=0.8,
                        help='per-concept composition weight')
    parser.add_argument('--lowdim', action='store_true')
    parser.add_argument('--data_dir', default='../dataset_modes')
    parser.add_argument('--out_suffix', default='')
    args = parser.parse_args()
    vision = not args.lowdim
    state_keys = ('agent', 'o1') if args.lowdim else ('agent',)

    results = {}
    for corner, side, speed in HELD_OUT:
        combo = f'{CORNER_NAMES[corner]}_{side}_{speed}'
        print(f'==== E4 held-out combo: {combo} ====', flush=True)
        paths = factor_paths(corner, side, speed)
        paths = {f: [p.replace('../dataset_modes/', args.data_dir.rstrip('/') + '/')
                     for p in ps] for f, ps in paths.items()}
        z = {}
        for factor, ps in paths.items():
            print(f'--- inferring z_{factor} from {len(ps)} demo sets ---', flush=True)
            z[factor] = infer_factor(args.model_path, f'{combo}_{factor}', ps,
                                     args.inversion_epochs, args.n_traj_per_set,
                                     vision=vision, state_keys=state_keys)

        # text oracle: encode the held-out combo's composable description with
        # the frozen policy's own text encoder (never trained on this combo)
        import torch
        from diffusion_policy.models.diffusion_policy import DiffusionPolicy
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        enc = DiffusionPolicy(obs_horizon=2, pred_horizon=16,
                              lowdim_obs_dim=2 * len(state_keys),
                              action_dim=2, num_diffusion_iters=100, vision=vision,
                              cached_labels_path='../output/cached_labels.pkl',
                              noise_pred_net_type='unet').to(device)
        enc.load_state_dict(torch.load(args.model_path, map_location=device))
        enc.eval()
        desc = (f'push the blue block to the {CORNER_NAMES[corner]} corner, '
                f'{SIDE_PHRASES[side]}, {SPEED_PHRASES[speed]}')
        with torch.no_grad():
            text_emb = enc.encode_text([desc]).cpu().numpy()
        del enc

        settings = {
            'composed_3': (np.stack([z['corner'], z['side'], z['speed']]),
                           np.array([args.w] * 3)),
            'corner_only': (z['corner'][None], np.array([1.5])),
            'text_oracle': (text_emb, np.array([1.5])),
        }
        for name, (emb, w) in settings.items():
            score = evaluate(max_steps=500, num_episodes=args.episodes,
                             model_path=args.model_path, render=False,
                             policy_type='diffusion', task=['blue', corner],
                             concept_embeddings=emb, concept_weights=w,
                             init_type=f'e4_{name}',
                             vision=vision, state_keys=state_keys)
            results[f'{combo}/{name}'] = score
            print(f'E4_RESULT {combo}/{name}: {score}/{args.episodes}', flush=True)

    out = args.model_path.replace('.pth', f'_e4_eval{args.out_suffix}.json')
    with open(out, 'w') as f:
        json.dump({'held_out': [f'{CORNER_NAMES[c]}_{si}_{sp}' for c, si, sp in HELD_OUT],
                   'episodes': args.episodes, 'per_setting': results}, f, indent=2)
    print('E4_DONE', json.dumps(results), flush=True)
