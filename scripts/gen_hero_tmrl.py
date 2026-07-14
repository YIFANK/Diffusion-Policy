"""Fig 1(b): TMRL-structured hero — problem (left, red) | mechanism (center
panel, real frames on top + latent-space minis below) | outcome (right,
green). Real frames from output/hero/*.npz (expert demos + actual policy
rollouts); latent maps from output/centroids2d.json. No bar charts.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
H = {k: dict(np.load(f'../output/hero/{k}.npz'))
     for k in ['ing1', 'ing2', 'held_demo', 'warm_roll', 'fail_roll']}
D = json.load(open('../output/centroids2d.json'))
xy = np.array(D['xy']); arith = np.array(D['arith_xy'])
origin = np.array(D['origin_xy'])
allp = np.vstack([xy, arith[None], origin[None]])
lo, hi = allp.min(0), allp.max(0)
xy = (xy - lo) / (hi - lo); arith = (arith - lo) / (hi - lo)
origin = (origin - lo) / (hi - lo)
target = arith + np.array([0.13, -0.11])
wrong = xy[int(np.argmin(np.linalg.norm(xy - origin, axis=1)))]

CREAM = '#fdfaf4'
fig = plt.figure(figsize=(9.4, 3.35))
fig.patch.set_facecolor(CREAM)


def traj_img(ax, dat, color, frac=1.0, lw=1.1):
    fr = dat['frames'][-1]
    ax.imshow(fr, extent=[0, 512, 0, 512], origin='lower')
    ag = dat['agent'][:int(len(dat['agent']) * frac)]
    ax.plot(ag[:, 0], ag[:, 1], color=color, lw=lw, alpha=0.55)
    ax.plot(*ag[0], 'o', color=color, ms=4, mfc='white', mew=1.2)
    ax.plot(*ag[-1], 'o', color=color, ms=4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#b0bec5'); s.set_linewidth(1.0)


def pill(ax, text, color='#546e7a', y=-0.16):
    ax.text(0.5, y, text, transform=ax.transAxes, ha='center', va='top',
            fontsize=6.6, color=color,
            bbox=dict(boxstyle='round,pad=0.32', fc='white', ec=color,
                      lw=0.8))


# ================= LEFT: the problem =================
fig.text(0.012, 0.955, 'the held-out combination', fontsize=9.5,
         color='#c62828', fontweight='bold')
fig.text(0.012, 0.875, '“...upper-left, clockwise, gently”',
         fontsize=7.6, style='italic', color='#37474f')
axL = fig.add_axes([0.035, 0.50, 0.115, 0.33])
traj_img(axL, H['held_demo'], '#c62828')
pill(axL, 'never trained', '#c62828')
axL.text(0.94, 0.92, '✗', transform=axL.transAxes, fontsize=11,
         color='white', ha='center', va='center', family='DejaVu Sans',
         bbox=dict(boxstyle='circle,pad=0.18', fc='#c62828', ec='none'))
axL1 = fig.add_axes([0.012, 0.145, 0.082, 0.24])
traj_img(axL1, H['ing1'], '#546e7a', lw=1.2)
pill(axL1, 'trained: ul·cw·fast')
axL2 = fig.add_axes([0.108, 0.145, 0.082, 0.24])
traj_img(axL2, H['ing2'], '#546e7a', lw=1.2)
pill(axL2, 'trained: ul·ccw·gentle')

# ================= CENTER: the mechanism =================
panel = fig.add_axes([0.215, 0.04, 0.55, 0.86])
panel.set_facecolor('#f2f0e9')
panel.set_xticks([]); panel.set_yticks([])
for s in panel.spines.values():
    s.set_visible(False)
fig.text(0.49, 0.955, 'Composition as Posterior Inference', fontsize=11.5,
         fontweight='bold', ha='center')

# top row: real frames
axC1 = fig.add_axes([0.235, 0.56, 0.14, 0.30])
traj_img(axC1, H['fail_roll'], '#e65100')
pill(axC1, 'cold inversion: misses the goal', '#e65100', y=-0.10)
axC2 = fig.add_axes([0.61, 0.56, 0.14, 0.30])
traj_img(axC2, H['warm_roll'], '#2e7d32')
pill(axC2, 'prior + 4 demos: the mode', '#2e7d32', y=-0.10)
fig.text(0.492, 0.73, 'warm-start the\ninversion at the\ncompositional prior',
         fontsize=7.2, ha='center', color='#37474f')
ar = FancyArrowPatch((0.40, 0.66), (0.585, 0.66),
                     transform=fig.transFigure, arrowstyle='-|>',
                     mutation_scale=11, color='#2e7d32', lw=1.6)
fig.add_artist(ar)

# bottom row: three latent-space minis (mechanism)
def mini(ax, stage):
    for (x, y) in xy:
        ax.scatter(x, y, s=12, c='#cfd8dc', edgecolors='#90a4ae',
                   linewidths=0.4, zorder=3)
    ax.add_patch(Circle(target, 0.075, fill=False, ls=':',
                        ec='#2e7d32', lw=1.0, zorder=4))
    if stage == 0:
        ax.scatter(*origin, marker='s', s=22, c='#e65100', zorder=6)
        ax.add_patch(FancyArrowPatch(origin, wrong, arrowstyle='-|>',
                                     mutation_scale=7, color='#e65100',
                                     lw=1.2, linestyle='--', zorder=5))
    if stage >= 1:
        ax.scatter(*arith, marker='*', s=110, c='#2e7d32',
                   edgecolors='white', linewidths=0.5, zorder=6)
    if stage == 2:
        ax.add_patch(FancyArrowPatch(arith, target, arrowstyle='-|>',
                                     mutation_scale=7, color='#2e7d32',
                                     lw=1.4, zorder=6))
        ax.scatter(*target, marker='*', s=110, c='#1b5e20',
                   edgecolors='white', linewidths=0.5, zorder=7)
    ax.set_xlim(-0.2, 1.25); ax.set_ylim(-0.2, 1.25)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#b0bec5'); s.set_linewidth(0.8)

labels = ['cold start: wrong basin', 'arithmetic prior: right neighborhood',
          '+ 4 demos: the mode']
for i, lab in enumerate(labels):
    ax = fig.add_axes([0.235 + i * 0.185, 0.10, 0.15, 0.27])
    ax.set_facecolor('white')
    mini(ax, i)
    ax.set_title(lab, fontsize=6.8, color='#37474f', pad=3)
fig.text(0.49, 0.035, 'the latent mode space (PCA of the real combination '
         'centroids)', fontsize=6.6, color='#78909c', ha='center')

# ================= RIGHT: the outcome =================
fig.text(0.79, 0.955, 'executed, never demonstrated', fontsize=9.5,
         color='#2e7d32', fontweight='bold')
w = H['warm_roll']
n = len(w['frames'])
for i, (fi, tag) in enumerate([(int(n * 0.25), 'approach: clockwise'),
                               (int(n * 0.7), 'pace: gentle'),
                               (n - 1, 'goal: upper-left')]):
    ax = fig.add_axes([0.81, 0.645 - i * 0.30, 0.095, 0.215])
    ax.imshow(w['frames'][fi], extent=[0, 512, 0, 512], origin='lower')
    k = int(len(w['agent']) * (fi + 1) / n)
    ag = w['agent'][:k]
    ax.plot(ag[:, 0], ag[:, 1], color='#2e7d32', lw=1.3, alpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#a5d6a7' if i < 2 else '#2e7d32')
        s.set_linewidth(1.0 if i < 2 else 1.6)
    pill(ax, tag, '#2e7d32', y=-0.115)
ax.text(0.93, 0.9, '✓', transform=ax.transAxes, fontsize=11,
        color='white', ha='center', va='center', family='DejaVu Sans',
        bbox=dict(boxstyle='circle,pad=0.18', fc='#2e7d32', ec='none'))

fig.savefig('../paper/figures/hero_bc.pdf', bbox_inches='tight',
            facecolor=CREAM)
print('wrote TMRL-style hero')
