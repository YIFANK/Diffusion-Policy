"""Interface head-to-head on HELD-OUT robosuite combos, on a policy trained
WITHOUT them (rs_factored_v2_holdout). Five routes to each held-out combo:

  text        — zero-shot language (the held-out description through the MLP)
  z_arith     — zero-shot composition at the 64-d interface:
                z(s,a,sp) = z(s,a',sp) + z(s,a,sp') - z(s,a',sp')
  inv_cold    — K=4 MAP inversion, zero init (3 restarts, best final loss)
  inv_warm_text  — K=4 inversion warm-started at the language point (+proximal)
  inv_warm_arith — K=4 inversion warm-started at the arithmetic point (+proximal)

Reports goal success, steps, side/arc/speed adherence, combination-level
success, and the conditional delta of every arm's conditioning vector.

Usage:
    MUJOCO_GL=egl python eval_rs_holdout.py \
        --model_path ../trained_models/rs_factored_v2_holdout.pth --episodes 10
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch
from diffusion_policy.data.dataset import (create_sample_indices, sample_sequence,
                                           normalize_data)
from train_robosuite_factored import desc_for, HELD_OUT_RS
from eval_rs_full import load_policy, eval_arm, OBS_HORIZON, PRED_HORIZON, \
    ACTION_HORIZON, device

Z_DIM = 64


class KShotRS(torch.utils.data.Dataset):
    """First K episodes of a held-out combo's npz, normalized with the
    TRAINING stats of the holdout model (mirrors deployment)."""

    def __init__(self, npz_path, k, stats):
        d = np.load(npz_path)
        ends = d['episode_ends'][:k]
        T = int(ends[-1])
        self.train_data = {
            'agent_pos': normalize_data(d['state'][:T], stats['agent_pos']),
            'action': normalize_data(d['action'][:T], stats['action'])}
        self.indices = create_sample_indices(ends, PRED_HORIZON,
                                             OBS_HORIZON - 1, ACTION_HORIZON - 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        b0, b1, s0, s1 = self.indices[idx]
        ns = sample_sequence(self.train_data, PRED_HORIZON, b0, b1, s0, s1)
        return {'agent_pos': ns['agent_pos'][:OBS_HORIZON], 'action': ns['action']}


def map_invert(policy, ds, epochs=200, lr=1e-2, z_init=None, prior_weight=0.0,
               seed=0):
    torch.manual_seed(seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    if z_init is None:
        z = torch.zeros(1, Z_DIM, device=device, requires_grad=True)
        anchor = None
    else:
        z = torch.tensor(np.asarray(z_init, dtype=np.float32)[None],
                         device=device, requires_grad=True)
        anchor = z.detach().clone()
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    final = None
    for ep in range(epochs):
        tot, n = 0.0, 0
        for b in dl:
            nagent = b['agent_pos'].to(device, dtype=torch.float32)
            naction = b['action'].to(device, dtype=torch.float32)
            B = nagent.shape[0]
            dummy = torch.zeros(B, OBS_HORIZON, 3, 8, 8, device=device)
            loss = policy(dummy, nagent, naction, z.expand(B, -1), p_uncond=0.0)
            if anchor is not None and prior_weight > 0:
                loss = loss + prior_weight * (z - anchor).pow(2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * B; n += B
        sched.step()
        final = tot / n
    return z.detach().cpu().numpy()[0], final


def cond_delta(policy, ds, cond, n_batch=256, t_mid=50):
    """Mean ||eps(x,o,z) - eps(x,o,0)|| on held-out demo tuples at mid noise."""
    dl = torch.utils.data.DataLoader(ds, batch_size=n_batch, shuffle=True)
    b = next(iter(dl))
    nagent = b['agent_pos'].to(device, dtype=torch.float32)
    naction = b['action'].to(device, dtype=torch.float32)
    B = nagent.shape[0]
    dummy = torch.zeros(B, OBS_HORIZON, 3, 8, 8, device=device)
    if isinstance(cond, str):
        ntext = [cond] * B
    else:
        ntext = torch.as_tensor(np.asarray(cond), dtype=torch.float32,
                                device=device).reshape(1, -1).expand(B, -1)
    with torch.no_grad():
        c = policy.get_cond(dummy, nagent, ntext, uncond=False)
        u = policy.get_cond(dummy, nagent, ntext, uncond=True)
        noise = torch.randn_like(naction)
        t = torch.full((B,), t_mid, device=device, dtype=torch.long)
        noisy = policy.noise_scheduler.add_noise(naction, noise, t)
        e_c = policy.noise_pred_net(noisy, t, global_cond=c)
        e_u = policy.noise_pred_net(noisy, t, global_cond=u)
    return float((e_c - e_u).flatten(1).norm(dim=1).mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        default='../trained_models/rs_factored_v2_holdout.pth')
    parser.add_argument('--data_dir', default='../robosuite_data')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--k_shot', type=int, default=4)
    parser.add_argument('--thresholds', default='../output/rs_adherence_thresholds.json')
    args = parser.parse_args()

    thr = json.load(open(args.thresholds))
    policy, stats = load_policy(args.model_path)

    enc = lambda s, a, sp: policy.encode_text([desc_for(s, a, sp)]).detach().cpu().numpy()[0]

    results = {}
    for combo in HELD_OUT_RS:
        s, a, sp = combo
        name = '_'.join(combo)
        desc = desc_for(s, a, sp)
        a2 = 'high' if a == 'direct' else 'direct'
        sp2 = 'slow' if sp == 'fast' else 'fast'
        z_arith = enc(s, a2, sp) + enc(s, a, sp2) - enc(s, a2, sp2)
        z_lang = enc(s, a, sp)

        ds = KShotRS(os.path.join(args.data_dir, f'{name}.npz'),
                     args.k_shot, stats)
        print(f'RSHOLD {name}: inverting ({len(ds)} samples from K={args.k_shot})',
              flush=True)
        colds = [map_invert(policy, ds, seed=r) for r in range(3)]
        z_cold, l_cold = min(colds, key=lambda x: x[1])
        print(f'RSHOLD {name}: cold restarts losses '
              f'{[f"{l:.4f}" for _, l in colds]}', flush=True)
        z_wt, _ = map_invert(policy, ds, z_init=z_lang, prior_weight=0.01)
        z_wa, _ = map_invert(policy, ds, z_init=z_arith, prior_weight=0.01)

        arms = {'text': desc, 'z_arith': z_arith, 'inv_cold': z_cold,
                'inv_warm_text': z_wt, 'inv_warm_arith': z_wa}
        results[name] = {}
        for tag, cond in arms.items():
            delta = cond_delta(policy, ds, cond)
            print(f'RSHOLD_DELTA {name}/{tag}: {delta:.3f}', flush=True)
            r = eval_arm(policy, stats, cond, combo, args.episodes, thr,
                         tag=f'{name}/{tag}')
            r['delta'] = delta
            results[name][tag] = r

    out = args.model_path.replace('.pth', '_holdout_eval.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print('RSHOLD_DONE', json.dumps(results), flush=True)
