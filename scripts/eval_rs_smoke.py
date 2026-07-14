"""Closed-loop smoke test for the factored robosuite policy: condition on a
trained combo's description, roll out, report success + steps (speed signature).

Usage:
    MUJOCO_GL=egl python eval_rs_smoke.py --episodes 10 \
        --combos left_direct_fast left_high_slow
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
from diffusion_policy.models.diffusion_policy import DiffusionPolicy
from diffusion_policy.data.dataset import normalize_data, unnormalize_data
from robosuite_expert import make_env
from train_robosuite_factored import desc_for, FactoredRS, HELD_OUT_RS

OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rollout(policy, stats, desc, seed, max_steps=900):
    env = make_env(seed)
    obs = env.reset()
    import collections
    def s7(o):
        return np.concatenate([o['robot0_eef_pos'], o['Can_pos'],
                               [o['robot0_gripper_qpos'][0]]]).astype(np.float32)
    dq = collections.deque([s7(obs)] * OBS_HORIZON, maxlen=OBS_HORIZON)
    dummy = torch.zeros(1, OBS_HORIZON, 3, 8, 8, device=device)
    steps = 0
    for _ in range(0, max_steps, ACTION_HORIZON):
        nag = normalize_data(np.stack(list(dq)), stats=stats['agent_pos'])
        na = torch.from_numpy(nag).unsqueeze(0).to(device, dtype=torch.float32)
        with torch.no_grad():
            act = policy.sample(nimages=dummy, nagent_poses=na, ntexts=[desc],
                                num_diffusion_iters=100, n_samples=1)
        acts = unnormalize_data(act.cpu().numpy()[0], stats=stats['action'])
        for a in acts[OBS_HORIZON - 1: OBS_HORIZON - 1 + ACTION_HORIZON]:
            obs, r, d, i = env.step(a)
            dq.append(s7(obs))
            steps += 1
            if env._check_success():
                env.close()
                return True, steps
    env.close()
    return False, steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/rs_factored_v1.pth')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--combos', nargs='+',
                        default=['left_direct_fast', 'left_high_slow'])
    args = parser.parse_args()

    stats_raw = json.load(open(args.model_path.replace('.pth', '_stats.json')))
    stats = {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in stats_raw.items()}

    policy = DiffusionPolicy(obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
                             lowdim_obs_dim=7, action_dim=7, num_diffusion_iters=100,
                             vision=False, cached_labels_path='../output/cached_labels.pkl',
                             noise_pred_net_type='unet').to(device)
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    policy.eval()

    for combo in args.combos:
        s, a, sp = combo.split('_')
        desc = desc_for(s, a, sp)
        outs = [rollout(policy, stats, desc, 500 + i) for i in range(args.episodes)]
        succ = sum(o[0] for o in outs)
        steps = [o[1] for o in outs if o[0]]
        ms = f' mean_steps={np.mean(steps):.0f}' if steps else ''
        print(f'RSSMOKE {combo}: {succ}/{args.episodes}{ms}', flush=True)
    print('RSSMOKE_DONE', flush=True)
