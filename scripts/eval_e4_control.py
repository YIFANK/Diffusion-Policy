"""E4 control: v2 with text conditioning on TRAINED (in-distribution) mode combos.

Completes the E4 interpretation: if v2 scores high here but failed on the
held-out combos even with text_oracle conditioning, the compositional
generalization gap is real (and not just "v2 is weak at these corners").

Picks, for each held-out combo, sibling TRAINED combos at the same corner
(so corner difficulty is matched) and evaluates with the combo's own trained
description.

Usage:
    python eval_e4_control.py --model_path ../trained_models/dp_unet_clip_modes_v2.pth
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
from inference import evaluate
from diffusion_policy.models.diffusion_policy import DiffusionPolicy
from train_modes import HELD_OUT, CORNER_NAMES
from cache_text_embeddings import SIDE_PHRASES, SPEED_PHRASES

SIDES = ['cw', 'ccw']
SPEEDS = ['gentle', 'fast']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/dp_unet_clip_modes_v2.pth')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--lowdim', action='store_true')
    parser.add_argument('--out_suffix', default='')
    args = parser.parse_args()
    vision = not args.lowdim
    state_keys = ('agent', 'o1') if args.lowdim else ('agent',)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = DiffusionPolicy(obs_horizon=2, pred_horizon=16,
                          lowdim_obs_dim=2 * len(state_keys),
                          action_dim=2, num_diffusion_iters=100, vision=vision,
                          cached_labels_path='../output/cached_labels.pkl',
                          noise_pred_net_type='unet').to(device)
    enc.load_state_dict(torch.load(args.model_path, map_location=device))
    enc.eval()

    # trained siblings: same corner as each held-out combo, different mode
    combos = []
    for corner, h_side, h_speed in HELD_OUT:
        for side in SIDES:
            for speed in SPEEDS:
                if (corner, side, speed) not in HELD_OUT:
                    combos.append((corner, side, speed))

    results = {}
    for corner, side, speed in combos:
        tag = f'{CORNER_NAMES[corner]}_{side}_{speed}'
        desc = (f'push the blue block to the {CORNER_NAMES[corner]} corner, '
                f'{SIDE_PHRASES[side]}, {SPEED_PHRASES[speed]}')
        with torch.no_grad():
            emb = enc.encode_text([desc]).cpu().numpy()
        score = evaluate(max_steps=500, num_episodes=args.episodes,
                         model_path=args.model_path, render=False,
                         policy_type='diffusion', task=['blue', corner],
                         concept_embeddings=emb, concept_weights=np.array([1.5]),
                         init_type='e4ctrl', vision=vision, state_keys=state_keys)
        results[tag] = score
        print(f'E4CTRL_RESULT {tag}: {score}/{args.episodes}', flush=True)

    out = args.model_path.replace('.pth', f'_e4_control{args.out_suffix}.json')
    with open(out, 'w') as f:
        json.dump({'episodes': args.episodes, 'per_combo': results}, f, indent=2)
    print('E4CTRL_DONE', json.dumps(results), flush=True)
