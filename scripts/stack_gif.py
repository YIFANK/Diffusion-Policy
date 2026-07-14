"""Render policy rollouts of StackThree as GIFs (256x256 agentview).

Usage:
    MUJOCO_GL=egl python stack_gif.py --model_path ../trained_models/stack_v3rel.pth \
        --seeds 5000 5001 5002 5003 --out_dir ../output/gifs_v3rel
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import collections
import numpy as np
import torch
from PIL import Image
from diffusion_policy.data.dataset import normalize_data, unnormalize_data
from robosuite_stack import (make_stack_env, state22, desc_stack, rel22,
                             TRAIN_COLORS)
from eval_stack import load_policy, achieved_pair, OBS_HORIZON, ACTION_HORIZON, device


def rollout_gif(policy, stats, seed, out_path, max_steps=450, every=3):
    torch.manual_seed(seed)  # reproducible diffusion sampling: the GIF shows
    # THIS rollout; note eval rollouts are separately sampled unless also seeded
    r2 = np.random.default_rng(seed)
    colors = list(r2.choice(TRAIN_COLORS, size=3, replace=False))
    si, ti = [int(x) for x in r2.choice(3, size=2, replace=False)]
    desc = desc_stack(colors[si], colors[ti])
    tf = rel22 if stats.get('relative') else (lambda s: s)
    env = make_stack_env(seed, colors,
                         constrained=stats.get('constrained', False), cam=256)
    obs = env.reset()
    env.set_target_pair(si, ti)
    dq = collections.deque([tf(state22(obs, env))] * OBS_HORIZON,
                           maxlen=OBS_HORIZON)
    dummy = torch.zeros(1, OBS_HORIZON, 3, 8, 8, device=device)
    frames = [obs['agentview_image'][::-1]]
    steps, success = 0, False
    min_xy, min_dz_at, grip_opened_after_carry, carried = 99.0, None, False, False
    src_z0 = env.cube_pos(si)[2]
    for _ in range(0, max_steps, ACTION_HORIZON):
        nag = normalize_data(np.stack(list(dq)), stats=stats['agent_pos'])
        na = torch.from_numpy(nag).unsqueeze(0).to(device, torch.float32)
        with torch.no_grad():
            act = policy.sample(nimages=dummy, nagent_poses=na, ntexts=[desc],
                                num_diffusion_iters=100, n_samples=1)
        acts = unnormalize_data(act.cpu().numpy()[0], stats=stats['action'])
        for a in acts[OBS_HORIZON - 1: OBS_HORIZON - 1 + ACTION_HORIZON]:
            obs, r, d, i = env.step(a)
            dq.append(tf(state22(obs, env)))
            steps += 1
            # placement anatomy: while the src cube is carried, track its
            # xy distance to the commanded target and release events
            src_p, tgt_p = env.cube_pos(si), env.cube_pos(ti)
            if src_p[2] > src_z0 + 0.05:
                carried = True
                xy = float(np.linalg.norm(src_p[:2] - tgt_p[:2]))
                if xy < min_xy:
                    min_xy, min_dz_at = xy, float(src_p[2] - tgt_p[2])
            if carried and a[-1] < 0:
                grip_opened_after_carry = True
            if steps % every == 0:
                frames.append(obs['agentview_image'][::-1])
            if env._check_success():
                success = True
                break
        if success:
            break
    frames.append(obs['agentview_image'][::-1])
    ach = (si, ti) if success else achieved_pair(env)
    env.close()

    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=60, loop=0)
    tag = 'SUCCESS' if success else 'FAIL'
    print(f'GIF {os.path.basename(out_path)}: [{desc}] {tag} steps={steps} '
          f'achieved={ach} min_xy={min_xy:.3f} dz_at_min={min_dz_at} '
          f'released={grip_opened_after_carry}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    policy, stats = load_policy(args.model_path)
    for s in args.seeds:
        rollout_gif(policy, stats, s,
                    os.path.join(args.out_dir, f'seed{s}.gif'))
    print('GIF_DONE', flush=True)
