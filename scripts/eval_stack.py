"""Color-grounding evaluation for the StackThree policy.

Arms:
  trained    — text on fresh scenes with TRAIN colors (grounding + confusion)
  zero_purple — text naming the held-out color the policy never saw as pixels/RGB
               (CLIP knows the word; the policy's MLP and denoiser do not)
  fewshot_cold / fewshot_warm — K=4 purple demos -> MAP inversion of a 64-d z
               (warm = init at the text embedding of the purple instruction)

Every rollout also records the ACHIEVED pair (which cube ended stacked on
which, if any) for a commanded-vs-achieved confusion analysis.

Usage:
    MUJOCO_GL=egl python eval_stack.py --model_path ../trained_models/stack_v1.pth
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import collections
import itertools
import json
import numpy as np
import torch
from diffusion_policy.models.diffusion_policy import DiffusionPolicy
from diffusion_policy.data.dataset import normalize_data, unnormalize_data
from robosuite_stack import (make_stack_env, state22, desc_stack,
                             TRAIN_COLORS, HELD_COLOR)
from eval_rs_holdout import KShotRS, map_invert

OBS_HORIZON, PRED_HORIZON, ACTION_HORIZON = 2, 16, 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_policy(model_path):
    stats_raw = json.load(open(model_path.replace('.pth', '_stats.json')))
    relative = bool(stats_raw.pop('relative', False))
    constrained = bool(stats_raw.pop('constrained', False))
    stats = {k: {kk: np.array(vv) for kk, vv in v.items()}
             for k, v in stats_raw.items()}
    stats['relative'] = relative
    stats['constrained'] = constrained
    policy = DiffusionPolicy(obs_horizon=OBS_HORIZON, pred_horizon=PRED_HORIZON,
                             lowdim_obs_dim=22, action_dim=7, num_diffusion_iters=100,
                             vision=False, cached_labels_path='../output/cached_labels.pkl',
                             noise_pred_net_type='unet').to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    return policy, stats


def achieved_pair(env):
    for si, ti in itertools.permutations(range(3), 2):
        env.set_target_pair(si, ti)
        if env._check_success():
            return (si, ti)
    return None


def rollout_stack(policy, stats, cond, seed, colors, pair, max_steps=450):
    from robosuite_stack import rel22
    tf = rel22 if stats.get('relative') else (lambda s: s)
    env = make_stack_env(seed, colors, constrained=stats.get('constrained', False))
    obs = env.reset()
    env.set_target_pair(*pair)
    dq = collections.deque([tf(state22(obs, env))] * OBS_HORIZON, maxlen=OBS_HORIZON)
    dummy = torch.zeros(1, OBS_HORIZON, 3, 8, 8, device=device)
    if isinstance(cond, str):
        ntexts = [cond]
    else:
        ntexts = torch.as_tensor(np.asarray(cond), dtype=torch.float32,
                                 device=device).reshape(1, -1)
    steps, success = 0, False
    for _ in range(0, max_steps, ACTION_HORIZON):
        nag = normalize_data(np.stack(list(dq)), stats=stats['agent_pos'])
        na = torch.from_numpy(nag).unsqueeze(0).to(device, dtype=torch.float32)
        with torch.no_grad():
            act = policy.sample(nimages=dummy, nagent_poses=na, ntexts=ntexts,
                                num_diffusion_iters=100, n_samples=1)
        acts = unnormalize_data(act.cpu().numpy()[0], stats=stats['action'])
        for a in acts[OBS_HORIZON - 1: OBS_HORIZON - 1 + ACTION_HORIZON]:
            obs, r, d, i = env.step(a)
            dq.append(tf(state22(obs, env)))
            steps += 1
            env.set_target_pair(*pair)
            if env._check_success():
                success = True
                break
        if success:
            break
    ach = (pair if success else achieved_pair(env))
    env.close()
    return success, steps, ach


def eval_color_arm(policy, stats, episodes, seed_base, tag, cond_fn,
                   palette='train', force_purple_role=None):
    """cond_fn(colors, si, ti) -> conditioning (str or z). Reports commanded
    success and achieved-pair confusion."""
    rng = np.random.default_rng(seed_base)
    n_ok, wrong_pair, no_stack = 0, 0, 0
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
        elif force_purple_role is None:
            si, ti = [int(x) for x in r2.choice(3, size=2, replace=False)]
        else:
            p = colors.index(HELD_COLOR)
            other = int(r2.choice([j for j in range(3) if j != p]))
            si, ti = (p, other) if force_purple_role == 'src' else (other, p)
        cond = cond_fn(colors, si, ti)
        ok, steps, ach = rollout_stack(policy, stats, cond, seed, colors, (si, ti))
        n_ok += ok
        if not ok:
            if ach is None:
                no_stack += 1
            else:
                wrong_pair += 1
    print(f'STACKEVAL {tag}: {n_ok}/{episodes} '
          f'(wrong-pair {wrong_pair}, no-stack {no_stack})', flush=True)
    return {'success': n_ok, 'episodes': episodes,
            'wrong_pair': wrong_pair, 'no_stack': no_stack}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/stack_v1.pth')
    parser.add_argument('--held_data', default='../robosuite_stack_data/held_purple.npz')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--k_shot', type=int, default=4)
    parser.add_argument('--arms', nargs='+',
                        default=['trained', 'zero_purple', 'fewshot'])
    args = parser.parse_args()

    policy, stats = load_policy(args.model_path)
    results = {}
    text_cond = lambda colors, si, ti: desc_stack(colors[si], colors[ti])

    if 'trained' in args.arms:
        results['trained'] = eval_color_arm(
            policy, stats, args.episodes, 5000, 'trained_colors', text_cond)

    if 'zero_purple' in args.arms:
        for role in ['src', 'tgt']:
            results[f'zero_purple_{role}'] = eval_color_arm(
                policy, stats, args.episodes, 6000 + 500 * (role == 'tgt'),
                f'zero_purple_{role}', text_cond,
                palette='held', force_purple_role=role)

    if 'fewshot' in args.arms:
        # K demos of one FIXED held-out pair (purple -> red): a single z can
        # only carry one instruction, so the inversion set must be
        # instruction-homogeneous (mixed-tgt demos would invert to mush)
        meta = json.load(open(args.held_data.replace('.npz', '_meta.json')))
        d = np.load(args.held_data)
        starts = np.concatenate([[0], d['episode_ends'][:-1]])
        idx = [i for i, m in enumerate(meta)
               if m['src'] == HELD_COLOR and m['tgt'] == 'red'][:args.k_shot]
        assert len(idx) == args.k_shot, f'only {len(idx)} purple->red demos'
        S = np.concatenate([d['state'][starts[i]:d['episode_ends'][i]] for i in idx])
        A = np.concatenate([d['action'][starts[i]:d['episode_ends'][i]] for i in idx])
        if stats.get('relative'):
            from robosuite_stack import rel22
            S = rel22(S)
        ends, tot = [], 0
        for i in idx:
            tot += int(d['episode_ends'][i] - starts[i])
            ends.append(tot)
        ktmp = '../robosuite_stack_data/_kshot_tmp.npz'
        np.savez(ktmp, state=S, action=A,
                 episode_ends=np.array(ends, dtype=np.int64))
        ds = KShotRS(ktmp, args.k_shot, stats)
        print(f'STACKEVAL inverting on {len(ds)} samples '
              f'({args.k_shot} purple->red demos)', flush=True)
        colds = [map_invert(policy, ds, seed=r) for r in range(3)]
        z_cold, _ = min(colds, key=lambda x: x[1])
        z_warm_init = policy.encode_text(
            [desc_stack(HELD_COLOR, 'red')]).detach().cpu().numpy()[0]
        z_warm, _ = map_invert(policy, ds, z_init=z_warm_init, prior_weight=0.01)

        # all three arms on IDENTICAL purple->red scenes (same seeds)
        results['purple_red_text'] = eval_color_arm(
            policy, stats, args.episodes, 7000, 'purple_red_text', text_cond,
            palette='purple_red')
        for tag, z in [('fewshot_cold', z_cold), ('fewshot_warm', z_warm)]:
            results[tag] = eval_color_arm(
                policy, stats, args.episodes, 7000, tag,
                lambda colors, si, ti, z=z: z, palette='purple_red')

    out = args.model_path.replace('.pth', '_color_eval.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print('STACKEVAL_DONE', json.dumps(results), flush=True)
