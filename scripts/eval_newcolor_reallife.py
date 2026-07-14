"""Real-life routes to grounding a NOVEL color, compared on identical scenes.

Arms (all use only K demos of purple->red; no observation editing):
  clip_inv     invert a 512-d CLIP-space pseudo-embedding through the FROZEN
               text MLP (textual-inversion style, Gal et al.)
  mlp_ft       unfreeze ONLY the text-projection MLP (~100k params), finetune
               on the K demos; also measure forgetting on trained-color scenes

Baselines for reference (from eval_stack): text 4/20, z64 cold 5/20, warm 3/20.

Usage:
    MUJOCO_GL=egl python eval_newcolor_reallife.py \
        --model_path ../trained_models/stack_v4c.pth --episodes 20
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import json
import numpy as np
import torch
from robosuite_stack import desc_stack, HELD_COLOR
from eval_stack import (load_policy, eval_color_arm, rollout_stack,
                        OBS_HORIZON, device)
from eval_rs_holdout import KShotRS

Z_CLIP = 512


def build_kshot(held_data, stats, k):
    meta = json.load(open(held_data.replace('.npz', '_meta.json')))
    d = np.load(held_data)
    starts = np.concatenate([[0], d['episode_ends'][:-1]])
    idx = [i for i, m in enumerate(meta)
           if m['src'] == HELD_COLOR and m['tgt'] == 'red'][:k]
    assert len(idx) == k
    S = np.concatenate([d['state'][starts[i]:d['episode_ends'][i]] for i in idx])
    A = np.concatenate([d['action'][starts[i]:d['episode_ends'][i]] for i in idx])
    ends, tot = [], 0
    for i in idx:
        tot += int(d['episode_ends'][i] - starts[i])
        ends.append(tot)
    tmp = '../robosuite_stack_data/_rl_kshot.npz'
    np.savez(tmp, state=S, action=A, episode_ends=np.array(ends, dtype=np.int64))
    return KShotRS(tmp, k, stats)


def clip_inversion(policy, ds, epochs=200, lr=1e-2):
    """Optimize a 512-d CLIP-space embedding through the frozen text MLP."""
    init = policy.cached_labels[desc_stack(HELD_COLOR, 'red')].float().to(device)
    e512 = init.clone().reshape(1, -1).requires_grad_(True)
    opt = torch.optim.Adam([e512], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    for ep in range(epochs):
        for b in dl:
            nagent = b['agent_pos'].to(device, torch.float32)
            naction = b['action'].to(device, torch.float32)
            B = nagent.shape[0]
            dummy = torch.zeros(B, OBS_HORIZON, 3, 8, 8, device=device)
            z64 = policy.text_encoder(e512).expand(B, -1)
            loss = policy(dummy, nagent, naction, z64, p_uncond=0.0)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    with torch.no_grad():
        return policy.text_encoder(e512).detach().cpu().numpy()[0]


def mlp_finetune(policy, ds, steps=600, lr=1e-4):
    """Clone the policy; unfreeze ONLY the text MLP; finetune on K demos."""
    ft = copy.deepcopy(policy)
    for p in ft.parameters():
        p.requires_grad_(False)
    for p in ft.text_encoder.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(ft.text_encoder.parameters(), lr=lr)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    desc = desc_stack(HELD_COLOR, 'red')
    it, done = iter(dl), 0
    while done < steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl)
            continue
        nagent = b['agent_pos'].to(device, torch.float32)
        naction = b['action'].to(device, torch.float32)
        B = nagent.shape[0]
        dummy = torch.zeros(B, OBS_HORIZON, 3, 8, 8, device=device)
        loss = ft(dummy, nagent, naction, [desc] * B, p_uncond=0.0)
        opt.zero_grad(); loss.backward(); opt.step()
        done += 1
    ft.eval()
    return ft


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../trained_models/stack_v4c.pth')
    parser.add_argument('--held_data', default='../robosuite_stack_data/held_c.npz')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--k_shot', type=int, default=4)
    args = parser.parse_args()

    policy, stats = load_policy(args.model_path)
    ds = build_kshot(args.held_data, stats, args.k_shot)
    text_cond = lambda colors, si, ti: desc_stack(colors[si], colors[ti])
    results = {}

    # (c) CLIP-space textual inversion
    z64 = clip_inversion(policy, ds)
    results['clip_inv'] = eval_color_arm(
        policy, stats, args.episodes, 7000, 'clip_inv',
        lambda colors, si, ti, z=z64: z, palette='purple_red')

    # (b) text-MLP-only finetune + forgetting
    ft = mlp_finetune(policy, ds)
    results['mlp_ft_purple'] = eval_color_arm(
        ft, stats, args.episodes, 7000, 'mlp_ft_purple', text_cond,
        palette='purple_red')
    results['mlp_ft_forgetting'] = eval_color_arm(
        ft, stats, args.episodes, 5000, 'mlp_ft_trained(forgetting)', text_cond,
        palette='train')

    with open('../output/newcolor_reallife.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('REALLIFE_DONE', json.dumps(results), flush=True)
