"""Regenerate paper/figures/factor_contrast.pdf — Fig 1(a), decluttered.

Two trajectories per condition (was ~8), cross-condition pairs matched by
nearest block start position, start/end markers, direction arrows; speed
panel keeps equal-time dots (the visual IS the dot density).
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
DIR = '../dataset_modes_60'


def load(corner, side, speed):
    d = pickle.load(open(f'{DIR}/Blue_{corner}_{side}_{speed}.pkl', 'rb'))
    out = []
    for t in d.trajectories:
        ag = np.array([[f.state['agent']['position'][0],
                        f.state['agent']['position'][1]] for f in t.data])
        o1 = np.array([t.data[0].state['o1']['position'][0],
                       t.data[0].state['o1']['position'][1]])
        o1_end = np.array([t.data[-1].state['o1']['position'][0],
                           t.data[-1].state['o1']['position'][1]])
        out.append((ag, o1, o1_end))
    return out, d.config['corner']


def pathlen(ag):
    return np.linalg.norm(np.diff(ag, axis=0), axis=1).sum()


def resample(ag, n=50):
    d = np.concatenate([[0], np.cumsum(
        np.linalg.norm(np.diff(ag, axis=0), axis=1))])
    if d[-1] == 0:
        return np.repeat(ag[:1], n, axis=0)
    s = np.linspace(0, d[-1], n)
    return np.stack([np.interp(s, d, ag[:, k]) for k in range(2)], axis=1)


def match_route(A, B):
    """pick the (a, b) pair whose ROUTES are most similar (speed panel:
    same path, different timing) among reasonably long episodes."""
    best, pair = None, None
    for a in A:
        if pathlen(a[0]) < 400:
            continue
        ra = resample(a[0])
        for b in B:
            if pathlen(b[0]) < 400:
                continue
            score = (np.linalg.norm(resample(b[0]) - ra, axis=1).mean()
                     + 0.5 * np.linalg.norm(a[1] - b[1]))
            if best is None or score < best:
                best, pair = score, (a, b)
    return [pair[0]], [pair[1]]


def match_pairs(A, B, n=1):
    """pick n episode pairs: similar block starts AND short, clean paths."""
    ia, ib = [], []
    used_b = set()
    order = sorted(((np.linalg.norm(a[1] - b[1])
                     + 0.08 * (pathlen(a[0]) + pathlen(b[0])), i, j)
                    for i, a in enumerate(A) for j, b in enumerate(B)))
    for score, i, j in order:
        if i in ia or j in used_b:
            continue
        ia.append(i); used_b.add(j); ib.append(j)
        if len(ia) == n:
            break
    return [A[i] for i in ia], [B[j] for j in ib]


def draw(ax, trajs, color, label, dots=False, stride=15):
    for k, (ag, o1, _) in enumerate(trajs):
        ax.plot(ag[:, 0], ag[:, 1], color=color, lw=2.4, alpha=0.95,
                label=label if k == 0 else None, zorder=3,
                solid_capstyle='round')
        if dots:
            ax.plot(ag[::stride, 0], ag[::stride, 1], 'o', ms=5,
                    mfc='white', mec=color, mew=1.4, alpha=0.95, zorder=4)
        ax.plot(*ag[0], 'o', color=color, ms=8, mfc='white',
                mew=2.0, zorder=5)
        ax.plot(*ag[-1], '*', color=color, ms=14, zorder=5)
        ax.plot(*o1, 's', color='#455a64', ms=10, alpha=0.6, zorder=2)


def goal_patch(ax, trajs):
    ends = np.array([t[2] for t in trajs])
    c = ends.mean(axis=0)
    ax.add_patch(plt.Circle(c, 36, color='#bbdefb', zorder=1))
    ax.text(*c, 'goal', ha='center', va='center', fontsize=8,
            color='#1565c0', zorder=2)


# find the upper-left-goal corner index
corner_idx = None
for i in range(4):
    _, cname = load(i, 'cw', 'fast')
    if cname == 'upper-left':
        corner_idx = i
        break

cw, _ = load(corner_idx, 'cw', 'fast')
ccw, _ = load(corner_idx, 'ccw', 'fast')
fast, _ = load(corner_idx, 'cw', 'fast')
gentle, _ = load(corner_idx, 'cw', 'gentle')

fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.1))
for ax in axes:
    ax.set_xlim(0, 512); ax.set_ylim(0, 512)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#263238')

a, b = match_pairs(cw, ccw)
goal_patch(axes[0], a + b)
draw(axes[0], a, '#1f6fd6', 'cw')
draw(axes[0], b, '#e53935', 'ccw')
axes[0].set_title('side contrast (goal: upper-left, fast)', fontsize=10.5)
axes[0].legend(loc='lower right', fontsize=9, framealpha=0.9)

a, b = match_route(fast, gentle)
goal_patch(axes[1], a + b)
draw(axes[1], a, '#1f6fd6', 'fast', dots=True, stride=10)
draw(axes[1], b, '#2e7d32', 'gentle', dots=True, stride=10)
axes[1].set_title('speed contrast (goal: upper-left, cw)', fontsize=10.5)
axes[1].legend(loc='lower right', fontsize=9, framealpha=0.9)
axes[1].text(10, 12, 'dots mark equal time intervals', fontsize=7.5,
             color='#546e7a')
axes[0].text(10, 12, r'$\circ$ start   $\star$ end   $\blacksquare$ block',
             fontsize=7.5, color='#546e7a')

plt.tight_layout()
fig.savefig('../paper_figs_out/factor_contrast.pdf', bbox_inches='tight')
print('FCONTRAST_DONE')
