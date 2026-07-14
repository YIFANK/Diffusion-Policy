"""E5: object selection in two-object scenes.

Closed-loop evaluation with per-episode metrics:
  - success: the TARGET object reaches the fixed goal corner
  - wrong_object: the DISTRACTOR reaches the goal corner instead (selection error)

Evaluates any checkpoint on all 4 selection tasks; use --model_path with v1
(single-object training) for the zero-shot control, v4 for the multi-object
policy.

Usage:
    python eval_e5_objects.py --model_path ../trained_models/dp_unet_clip_twoobj_v4.pth
    python eval_e5_objects.py --model_path ../trained_models/dp_unet_clip_scripted_v1.pth
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
from generate_demos import (TWO_OBJ_YAML, TWO_OBJ_SCENES, TWO_OBJ_TARGETS,
                            TWO_OBJ_GOAL_CORNER, WIN_CONDITION)
from train_objects import build_lists

OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')


def run_episode(policy, stats, target_name, seed, max_steps=250, guidance=1.5,
                guidance_first=None, state_keys=('agent',)):
    """guidance_first: stronger CFG for the FIRST plan only (the selection
    moment), reverting to `guidance` for execution — exploits the fact that
    text matters at the initial commitment and over-guidance destabilizes
    the push phase."""
    scene_key, target_key = TWO_OBJ_TARGETS[target_name]
    g1, c1, g2, c2 = TWO_OBJ_SCENES[scene_key]
    cfg = OmegaConf.create(TWO_OBJ_YAML.format(o1_geom=g1, o1_color=c1,
                                               o2_geom=g2, o2_color=c2))
    env = ter_env.TEREnv(**cfg.info, scene_info=cfg.scene_info,
                         agent_info=cfg.agent_info, verbose=False)
    env.set_seed(seed)
    env._setup()
    image_observer = ImageObserver(env, render_size=96, verbose=False)
    state_observer = StateObserver(env, verbose=False)
    distractor_key = 'o2' if target_key == 'o1' else 'o1'
    desc = f'push the {target_name} block to the lower-right corner'

    def observe():
        st = state_observer.observe()
        lowdim = np.concatenate([
            np.asarray(env.agent.position, dtype=np.float32) if k == 'agent'
            else np.asarray(st[k]['position'], dtype=np.float32)[:2]
            for k in state_keys])
        return {'image': image_observer.observe(), 'agent_pos': lowdim}

    obs = observe()
    obs_deque = collections.deque([obs] * OBS_HORIZON, maxlen=OBS_HORIZON)
    win = WIN_CONDITION[TWO_OBJ_GOAL_CORNER]
    for step in range(0, max_steps, ACTION_HORIZON):
        images = np.stack([x['image'] for x in obs_deque]).astype(np.float32) / 255.0
        agent_poses = np.stack([x['agent_pos'] for x in obs_deque])
        nagent = normalize_data(agent_poses, stats=stats['agent_pos'])
        # policy.sample/get_cond expects batched input: (B, obs_horizon, C, H, W)
        nimages = torch.from_numpy(images).permute(0, 3, 1, 2).unsqueeze(0).to(device, dtype=torch.float32)
        nagent = torch.from_numpy(nagent).unsqueeze(0).to(device, dtype=torch.float32)
        with torch.no_grad():
            gs = guidance_first if (guidance_first is not None and step == 0) else guidance
            naction = policy.sample(nimages=nimages, nagent_poses=nagent,
                                    ntexts=[desc], num_diffusion_iters=100, n_samples=1,
                                    guidance_scale=gs)
        acts = unnormalize_data(naction.cpu().numpy()[0], stats=stats['action'])
        for a in acts[OBS_HORIZON - 1: OBS_HORIZON - 1 + ACTION_HORIZON]:
            env.step(a)
            obs_deque.append(observe())
            st = state_observer.observe()
            if win(np.array(st[target_key]['position'])):
                return 'success'
            if win(np.array(st[distractor_key]['position'])):
                return 'wrong_object'
    return 'timeout'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--guidance', type=float, default=1.5)
    parser.add_argument('--guidance_first', type=float, default=None,
                        help='stronger CFG for the first plan only')
    parser.add_argument('--targets', nargs='+', default=list(TWO_OBJ_TARGETS))
    parser.add_argument('--text_gate', action='store_true')
    parser.add_argument('--lowdim', action='store_true')
    args = parser.parse_args()
    state_keys = ('agent', 'o1', 'o2') if args.lowdim else ('agent',)

    # normalization stats from the two-object training data
    paths, descs = build_lists()
    stats = PushTImageDataset(paths, descs, pred_horizon=PRED_HORIZON,
                              obs_horizon=OBS_HORIZON, action_horizon=ACTION_HORIZON,
                              state_keys=state_keys).stats

    policy = DiffusionPolicy(obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
                             lowdim_obs_dim=2 * len(state_keys), action_dim=2,
                             num_diffusion_iters=100,
                             vision=not args.lowdim,
                             cached_labels_path='../output/cached_labels.pkl',
                             noise_pred_net_type='unet',
                             text_gate=args.text_gate).to(device)
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    policy.eval()

    results = {}
    for target in args.targets:
        outcomes = collections.Counter(
            run_episode(policy, stats, target, seed=1000 + i, guidance=args.guidance,
                        guidance_first=args.guidance_first, state_keys=state_keys)
            for i in range(args.episodes))
        results[target] = dict(outcomes)
        print(f'E5_RESULT {target}: {dict(outcomes)}', flush=True)

    gf = f'_first{args.guidance_first}' if args.guidance_first else ''
    out = args.model_path.replace('.pth', f'_e5_eval_gs{args.guidance}{gf}.json')
    with open(out, 'w') as f:
        json.dump({'model': os.path.basename(args.model_path),
                   'episodes': args.episodes, 'per_target': results}, f, indent=2)
    print('E5_DONE', json.dumps(results), flush=True)
