"""Closed-loop evaluation of a factored robosuite policy with mode-adherence
scoring. Works for both the text-conditioned model (cond = description string)
and direct-z conditioning (cond = 64-d latent), so held-out arms reuse it.

Usage (smoke all trained combos of v2):
    MUJOCO_GL=egl python eval_rs_full.py --model_path ../trained_models/rs_factored_v2.pth \
        --episodes 10 --combos all
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import collections
import json
import numpy as np
import torch
from diffusion_policy.models.diffusion_policy import DiffusionPolicy
from diffusion_policy.data.dataset import normalize_data, unnormalize_data
from robosuite_expert import make_env
from train_robosuite_factored import desc_for, ALL_COMBOS, HELD_OUT_RS
from rs_adherence import labels_for

OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def s7(o):
    return np.concatenate([o['robot0_eef_pos'], o['Can_pos'],
                           [o['robot0_gripper_qpos'][0]]]).astype(np.float32)


def load_policy(model_path):
    stats_raw = json.load(open(model_path.replace('.pth', '_stats.json')))
    stats = {k: {kk: np.array(vv) for kk, vv in v.items()}
             for k, v in stats_raw.items()}
    policy = DiffusionPolicy(obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
                             lowdim_obs_dim=7, action_dim=7, num_diffusion_iters=100,
                             vision=False, cached_labels_path='../output/cached_labels.pkl',
                             noise_pred_net_type='unet').to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    return policy, stats


def rollout_rs(policy, stats, cond, seed, max_steps=900, guidance_scale=1.5):
    """cond: task-description string OR a 64-d latent (np.ndarray/tensor)."""
    env = make_env(seed)
    obs = env.reset()
    dq = collections.deque([s7(obs)] * OBS_HORIZON, maxlen=OBS_HORIZON)
    dummy = torch.zeros(1, OBS_HORIZON, 3, 8, 8, device=device)
    if isinstance(cond, str):
        ntexts = [cond]
    else:
        z = torch.as_tensor(np.asarray(cond), dtype=torch.float32, device=device)
        ntexts = z.reshape(1, -1)
    traj = [s7(obs)]
    steps = 0
    success = False
    for _ in range(0, max_steps, ACTION_HORIZON):
        nag = normalize_data(np.stack(list(dq)), stats=stats['agent_pos'])
        na = torch.from_numpy(nag).unsqueeze(0).to(device, dtype=torch.float32)
        with torch.no_grad():
            act = policy.sample(nimages=dummy, nagent_poses=na, ntexts=ntexts,
                                num_diffusion_iters=100, n_samples=1,
                                guidance_scale=guidance_scale)
        acts = unnormalize_data(act.cpu().numpy()[0], stats=stats['action'])
        for a in acts[OBS_HORIZON - 1: OBS_HORIZON - 1 + ACTION_HORIZON]:
            obs, r, d, i = env.step(a)
            dq.append(s7(obs))
            traj.append(s7(obs))
            steps += 1
            if env._check_success():
                success = True
                break
        if success:
            break
    env.close()
    return success, steps, np.array(traj)


def eval_arm(policy, stats, cond, combo, episodes, thresholds, seed_base=500,
             max_steps=900, tag=''):
    """Roll out `episodes` times; report success, steps, and per-factor
    adherence measured on ALL episodes (adherence is style, not success)."""
    side, arc, speed = combo
    n_s = 0
    steps_ok = []
    adh = {'side': 0, 'arc': 0, 'speed': 0}
    combo_hits = 0
    for i in range(episodes):
        ok, t, traj = rollout_rs(policy, stats, cond, seed_base + i, max_steps)
        lab = labels_for(traj, thresholds)
        m_side = lab['side'] == side
        m_arc = lab['arc'] == arc
        m_speed = lab['speed'] == speed
        adh['side'] += m_side
        adh['arc'] += m_arc
        adh['speed'] += m_speed
        n_s += ok
        if ok:
            steps_ok.append(t)
        combo_hits += ok and m_side and m_arc and m_speed
    ms = f' steps={np.mean(steps_ok):.0f}' if steps_ok else ''
    print(f'RSEVAL {tag} {"_".join(combo)}: goal {n_s}/{episodes}{ms} | '
          f'side {adh["side"]} arc {adh["arc"]} speed {adh["speed"]} | '
          f'COMBO {combo_hits}/{episodes}', flush=True)
    return {'goal': n_s, 'steps': float(np.mean(steps_ok)) if steps_ok else None,
            'side': adh['side'], 'arc': adh['arc'], 'speed': adh['speed'],
            'combo': combo_hits, 'episodes': episodes}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/rs_factored_v2.pth')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--combos', nargs='+', default=['all'])
    parser.add_argument('--thresholds', default='../output/rs_adherence_thresholds.json')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    thr = json.load(open(args.thresholds))
    policy, stats = load_policy(args.model_path)
    combos = (ALL_COMBOS if args.combos == ['all']
              else [tuple(c.split('_')) for c in args.combos])

    results = {}
    for combo in combos:
        desc = desc_for(*combo)
        results['_'.join(combo)] = eval_arm(policy, stats, desc, combo,
                                            args.episodes, thr, tag='text')
    out = args.out or args.model_path.replace('.pth', '_eval.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print('RSEVAL_DONE', flush=True)
