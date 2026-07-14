"""Color-grounding eval for the VISION StackThree host: does a pretrained
visual encoder's continuous color space ground a novel color where the
state-RGB prototype lookup failed?

Arms: trained / zero_purple_src / zero_purple_tgt / purple_red_text —
directly comparable to the state host (v4c: 18/20, 8/20, 3/20, 4/20).

Usage:
    MUJOCO_GL=egl python eval_stack_vis.py --model_path ../trained_models/stack_v6vis.pth
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
from robosuite_stack import make_stack_env, desc_stack, TRAIN_COLORS, HELD_COLOR
from eval_stack import achieved_pair

OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8
PROPRIO = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_vis_policy(model_path):
    raw = json.load(open(model_path.replace('.pth', '_stats.json')))
    raw.pop('vision', None); raw.pop('relative', None)
    constrained = bool(raw.pop('constrained', False))
    stats = {k: {kk: np.array(vv) for kk, vv in v.items()}
             for k, v in raw.items()}
    stats['constrained'] = constrained
    policy = DiffusionPolicy(obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
                             lowdim_obs_dim=PROPRIO, action_dim=7,
                             num_diffusion_iters=100, vision=True,
                             cached_labels_path='../output/cached_labels.pkl',
                             noise_pred_net_type='unet').to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    return policy, stats


def s4(obs):
    return np.concatenate([obs['robot0_eef_pos'],
                           [obs['robot0_gripper_qpos'][0]]]).astype(np.float32)


def rollout_vis(policy, stats, desc, seed, colors, pair, max_steps=450):
    torch.manual_seed(seed)
    env = make_stack_env(seed, colors, constrained=stats.get('constrained', False))
    obs = env.reset()
    env.set_target_pair(*pair)
    im0 = obs['agentview_image'][::-1].astype(np.float32).transpose(2, 0, 1) / 255.0
    imq = collections.deque([im0] * OBS_HORIZON, maxlen=OBS_HORIZON)
    sq = collections.deque([s4(obs)] * OBS_HORIZON, maxlen=OBS_HORIZON)
    steps, success = 0, False
    for _ in range(0, max_steps, ACTION_HORIZON):
        nag = normalize_data(np.stack(list(sq)), stats=stats['agent_pos'])
        na = torch.from_numpy(nag).unsqueeze(0).to(device, torch.float32)
        ni = torch.from_numpy(np.stack(list(imq))).unsqueeze(0).to(device, torch.float32)
        with torch.no_grad():
            act = policy.sample(nimages=ni, nagent_poses=na, ntexts=[desc],
                                num_diffusion_iters=100, n_samples=1)
        acts = unnormalize_data(act.cpu().numpy()[0], stats=stats['action'])
        for a in acts[OBS_HORIZON - 1: OBS_HORIZON - 1 + ACTION_HORIZON]:
            obs, r, d, i = env.step(a)
            sq.append(s4(obs))
            imq.append(obs['agentview_image'][::-1].astype(np.float32)
                       .transpose(2, 0, 1) / 255.0)
            steps += 1
            if env._check_success():
                success = True
                break
        if success:
            break
    ach = pair if success else achieved_pair(env)
    env.close()
    return success, steps, ach


def arm(policy, stats, episodes, seed_base, tag, palette='train',
        purple_role=None, fixed_pair=None):
    n_ok, wrong = 0, 0
    for i in range(episodes):
        seed = seed_base + i
        r2 = np.random.default_rng(seed)
        if palette == 'train':
            colors = list(r2.choice(TRAIN_COLORS, size=3, replace=False))
        elif palette == 'purple_red':
            third = str(r2.choice([c for c in TRAIN_COLORS if c != 'red']))
            colors = [HELD_COLOR, 'red', third]
            r2.shuffle(colors)
        else:
            two = list(r2.choice(TRAIN_COLORS, size=2, replace=False))
            colors = two + [HELD_COLOR]
            r2.shuffle(colors)
        if palette == 'purple_red':
            si, ti = colors.index(HELD_COLOR), colors.index('red')
        elif purple_role is None:
            si, ti = [int(x) for x in r2.choice(3, size=2, replace=False)]
        else:
            p = colors.index(HELD_COLOR)
            other = int(r2.choice([j for j in range(3) if j != p]))
            si, ti = (p, other) if purple_role == 'src' else (other, p)
        desc = desc_stack(colors[si], colors[ti])
        ok, steps, ach = rollout_vis(policy, stats, desc, seed, colors, (si, ti))
        n_ok += ok
        wrong += (not ok) and (ach is not None)
    print(f'STACKVIS {tag}: {n_ok}/{episodes} (wrong-pair {wrong})', flush=True)
    return {'success': n_ok, 'episodes': episodes, 'wrong_pair': wrong}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/stack_v6vis.pth')
    parser.add_argument('--episodes', type=int, default=20)
    args = parser.parse_args()
    policy, stats = load_vis_policy(args.model_path)
    results = {}
    results['trained'] = arm(policy, stats, args.episodes, 5000, 'trained')
    results['zero_purple_src'] = arm(policy, stats, args.episodes, 6000,
                                     'zero_purple_src', 'held', 'src')
    results['zero_purple_tgt'] = arm(policy, stats, args.episodes, 6500,
                                     'zero_purple_tgt', 'held', 'tgt')
    results['purple_red_text'] = arm(policy, stats, args.episodes, 7000,
                                     'purple_red_text', 'purple_red')
    with open(args.model_path.replace('.pth', '_color_eval.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('STACKVIS_DONE', json.dumps(results), flush=True)
