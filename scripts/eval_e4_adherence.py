"""E4 with combination-level adherence (reviewer MC1/MC2/MC3).

For each held-out combo and each conditioning arm, rollouts log the full
agent/block trajectory; we report
  goal   = commanded corner reached,
  side   = approach-winding sign matches commanded side,
  speed  = steps-to-success vs per-corner demo-calibrated threshold,
  combo  = goal AND side AND speed.
Arms: text_oracle, corner_only, composed_3, and an unconditional baseline
(z=0, w=0 -- the random-policy corner base rate the reviewer asked for).

Usage:
    python eval_e4_adherence.py --model_path ../trained_models/dp_lowdim_modes_60ep.pth \
        --lowdim --data_dir ../dataset_modes_60 --episodes 10
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
from train_modes import HELD_OUT
from cache_text_embeddings import SIDE_PHRASES, SPEED_PHRASES
from eval_e4 import factor_paths, infer_factor
from inference import sample_with_concept_composition
from mode_adherence import approach_winding, side_label, speed_label, calibrate_from_demos

OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rollout_concept(policy, stats, embs, weights, corner, seed, state_keys,
                    max_steps=500):
    cfg = OmegaConf.create(ENV_YAML.format(color='Blue'))
    env = ter_env.TEREnv(**cfg.info, scene_info=cfg.scene_info,
                         agent_info=cfg.agent_info, verbose=False)
    env.set_seed(seed)
    env._setup()
    imgo = ImageObserver(env, render_size=96, verbose=False)
    sto = StateObserver(env, verbose=False)

    def observe():
        st = sto.observe()
        lowdim = np.concatenate([
            np.asarray(env.agent.position, dtype=np.float32) if k == 'agent'
            else np.asarray(st[k]['position'], dtype=np.float32)[:2]
            for k in state_keys])
        return {'image': imgo.observe(), 'agent_pos': lowdim,
                'agent_xy': np.asarray(env.agent.position, dtype=np.float64),
                'block_xy': np.asarray(st['o1']['position'], dtype=np.float64)[:2]}

    dq = collections.deque([observe()] * OBS_HORIZON, maxlen=OBS_HORIZON)
    agent_pts, block_pts = [], []
    steps = 0
    win = WIN_CONDITION[corner]
    c0 = np.zeros(64, dtype=np.float32)
    for _ in range(0, max_steps, ACTION_HORIZON):
        images = np.stack([x['image'] for x in dq]).astype(np.float32) / 255.0
        nagent = normalize_data(np.stack([x['agent_pos'] for x in dq]),
                                stats=stats['agent_pos'])
        ni = torch.from_numpy(images).permute(0, 3, 1, 2).to(device, dtype=torch.float32)
        na = torch.from_numpy(nagent).to(device, dtype=torch.float32)
        with torch.no_grad():
            if weights is None:  # unconditional baseline
                act = policy.sample(nimages=ni.unsqueeze(0), nagent_poses=na.unsqueeze(0),
                                    ntexts=None, num_diffusion_iters=100, n_samples=1,
                                    guidance_scale=1.0)
            else:
                act = sample_with_concept_composition(
                    policy=policy, nimages=ni, nagent_poses=na,
                    concept_embeddings=embs, concept_weights=weights,
                    c0_embedding=c0, num_diffusion_iters=100, n_samples=1,
                    policy_type='diffusion')
        acts = unnormalize_data(act.cpu().numpy()[0], stats=stats['action'])
        for a in acts[OBS_HORIZON - 1: OBS_HORIZON - 1 + ACTION_HORIZON]:
            env.step(a)
            o = observe()
            dq.append(o)
            agent_pts.append(o['agent_xy']); block_pts.append(o['block_xy'])
            steps += 1
            if win(o['block_xy']):
                return True, steps, agent_pts, block_pts
    return False, steps, agent_pts, block_pts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--lowdim', action='store_true')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--data_dir', default='../dataset_modes_60')
    parser.add_argument('--inversion_epochs', type=int, default=200)
    parser.add_argument('--w', type=float, default=0.8)
    parser.add_argument('--out_suffix', default='')
    args = parser.parse_args()
    state_keys = ('agent', 'o1') if args.lowdim else ('agent',)
    vision = not args.lowdim

    speed_thresh = calibrate_from_demos(args.data_dir)

    from train_modes import build_split
    tp, td, _, _ = build_split()
    paths = [p.replace('../dataset_modes/', args.data_dir.rstrip('/') + '/') for p in tp]
    stats = PushTImageDataset(paths, td, pred_horizon=PRED_HORIZON,
                              obs_horizon=OBS_HORIZON, action_horizon=ACTION_HORIZON,
                              state_keys=state_keys).stats

    policy = DiffusionPolicy(obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
                             lowdim_obs_dim=2 * len(state_keys), action_dim=2,
                             num_diffusion_iters=100, vision=vision,
                             cached_labels_path='../output/cached_labels.pkl',
                             noise_pred_net_type='unet').to(device)
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    policy.eval()

    results = {}
    for corner, side, speed in HELD_OUT:
        combo = f'{CORNER_NAMES[corner]}_{side}_{speed}'
        print(f'==== {combo} ====', flush=True)
        fpaths = factor_paths(corner, side, speed)
        fpaths = {f: [p.replace('../dataset_modes/', args.data_dir.rstrip('/') + '/')
                      for p in ps] for f, ps in fpaths.items()}
        z = {}
        for factor, ps in fpaths.items():
            z[factor] = infer_factor(args.model_path, f'adh_{combo}_{factor}', ps,
                                     args.inversion_epochs, 4,
                                     vision=vision, state_keys=state_keys)
        desc = (f'push the blue block to the {CORNER_NAMES[corner]} corner, '
                f'{SIDE_PHRASES[side]}, {SPEED_PHRASES[speed]}')
        with torch.no_grad():
            text_emb = policy.encode_text([desc]).cpu().numpy()

        arms = {
            'unconditional': (None, None),
            'text_oracle': (text_emb, np.array([1.5])),
            'corner_only': (z['corner'][None], np.array([1.5])),
            'composed_3': (np.stack([z['corner'], z['side'], z['speed']]),
                           np.array([args.w] * 3)),
        }
        for name, (embs, w) in arms.items():
            goal_n = side_n = speed_n = combo_n = 0
            steps_list = []
            for i in range(args.episodes):
                ok, steps, ap, bp = rollout_concept(policy, stats, embs, w,
                                                    corner, 8000 + i, state_keys)
                s_lab = side_label(approach_winding(ap, bp))
                sp_lab = speed_label(steps, speed_thresh[corner]) if ok else None
                goal_n += ok
                side_n += (s_lab == side)
                if ok:
                    speed_n += (sp_lab == speed)
                    steps_list.append(steps)
                    combo_n += (s_lab == side and sp_lab == speed)
            results[f'{combo}/{name}'] = {
                'goal': goal_n, 'side_adh': side_n, 'speed_adh_given_goal': speed_n,
                'combo': combo_n,
                'mean_steps': float(np.mean(steps_list)) if steps_list else None}
            print(f'E4ADH_RESULT {combo}/{name}: goal {goal_n}/{args.episodes} '
                  f'side {side_n}/{args.episodes} speed|goal {speed_n}/{goal_n if goal_n else 0} '
                  f'COMBO {combo_n}/{args.episodes}', flush=True)

    out = args.model_path.replace('.pth', f'_e4_adherence{args.out_suffix}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print('E4ADH_DONE', json.dumps(results), flush=True)
