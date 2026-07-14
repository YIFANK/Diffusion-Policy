"""Generate paper/figures/hero_bc.pdf — Fig 1(b): the story triptych.

Prompting fails (X) -> cold inversion lands in the wrong basin (the REAL
latent geometry: PCA of the holdout policy's combo centroids) -> prior + 4
demos finds the correct mode (vertical result bars). Reads left to right
like a sentence; center panel is real data (output/centroids2d.json from
make_page_assets.py), not a cartoon.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
D = json.load(open('../output/centroids2d.json'))
xy = np.array(D['xy'])
combos = [tuple(c) for c in D['combos']]
arith = np.array(D['arith_xy'])
origin = np.array(D['origin_xy'])
# normalize everything into a unit box so offsets/radii are stable
allp = np.vstack([xy, arith[None], origin[None]])
lo, hi = allp.min(0), allp.max(0)
norm = lambda p: (p - lo) / (hi - lo)
xy = norm(xy); arith = norm(arith); origin = norm(origin)

fig = plt.figure(figsize=(8.8, 3.0))

# ---------------- panel 1: prompting fails ----------------
ax1 = fig.add_axes([0.015, 0.06, 0.24, 0.80])
ax1.axis('off')
ax1.set_title('prompting fails', fontsize=10, color='#c62828',
              fontweight='bold', loc='left')
ax1.text(0.5, 0.76, '“push to the upper-left,\nclockwise, gently”',
         ha='center', va='center', fontsize=8.6, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', fc='#fbe9e7', ec='#c62828'),
         transform=ax1.transAxes)
ax1.annotate('', xy=(0.5, 0.42), xytext=(0.5, 0.58),
             xycoords='axes fraction',
             arrowprops=dict(arrowstyle='-|>', color='#78909c', lw=1.2))
ax1.text(0.5, 0.30, 'frozen diffusion policy', ha='center', fontsize=8.6,
         bbox=dict(boxstyle='round,pad=0.45', fc='#e3f2fd', ec='#1565c0'),
         transform=ax1.transAxes)
ax1.text(0.5, 0.10, 'held-out combination:  0.5 / 20  [x]',
         ha='center', fontsize=9, color='#c62828', fontweight='bold',
         transform=ax1.transAxes)

# ---------------- panel 2: the latent space (real PCA) ----------------
ax2 = fig.add_axes([0.30, 0.10, 0.40, 0.76])
ax2.set_title('reaching the mode needs a prior',
              fontsize=10, fontweight='bold', loc='left')
col = {('gentle',): '#2e7d32', ('fast',): '#546e7a'}
for (c, s, sp), (x, y) in zip(combos, xy):
    ax2.scatter(x, y, s=42, c='#b0bec5' if sp == 'fast' else '#a5d6a7',
                edgecolors='#546e7a', linewidths=0.6, zorder=3)
ax2.scatter(*arith, marker='*', s=260, c='#2e7d32', edgecolors='white',
            linewidths=0.8, zorder=6)
ax2.annotate('compositional prior\n(latent arithmetic)', arith,
             xytext=(arith[0] - 0.52, arith[1] + 0.16), fontsize=7.6,
             color='#2e7d32',
             arrowprops=dict(arrowstyle='-', color='#2e7d32', lw=0.7))
target = arith + np.array([0.13, -0.11])
ax2.add_patch(Circle(target, 0.075, fill=False, ls=':', ec='#2e7d32',
                     lw=1.4, zorder=5))
ax2.annotate('held-out mode', target,
             xytext=(target[0] + 0.10, target[1] - 0.16), fontsize=7.6,
             color='#2e7d32',
             arrowprops=dict(arrowstyle='-', color='#2e7d32', lw=0.7))
# cold path: origin -> nearest trained centroid (wrong basin)
d = np.linalg.norm(xy - origin, axis=1)
wrong = xy[int(np.argmin(d))]
ax2.scatter(*origin, marker='s', s=44, c='#e65100', zorder=6)
ax2.add_patch(FancyArrowPatch(origin, wrong, arrowstyle='-|>',
                              mutation_scale=10, color='#e65100', lw=1.4,
                              linestyle='--', zorder=5))
ax2.annotate('cold inversion:\nwrong basin (9/20)',
             (origin + (np.array(wrong) - origin) * 0.5),
             xytext=(min(origin[0], wrong[0]) - 0.05,
                     max(origin[1], wrong[1]) + 0.13), fontsize=7.6,
             color='#e65100')
ax2.add_patch(FancyArrowPatch(arith, target, arrowstyle='-|>',
                              mutation_scale=10, color='#2e7d32', lw=1.8,
                              zorder=6))
ax2.text(0.985, 0.03, 'PCA of the policy’s real combo centroids',
         fontsize=7, color='#78909c', ha='right',
         transform=ax2.transAxes)
ax2.set_xlim(-0.22, 1.30); ax2.set_ylim(-0.22, 1.30)
ax2.set_aspect('equal')
ax2.set_xticks([]); ax2.set_yticks([])
for s in ax2.spines.values():
    s.set_color('#cfd8dc')

# ---------------- panel 3: results (vertical bars) ----------------
ax3 = fig.add_axes([0.775, 0.16, 0.21, 0.70])
arms = ['prompt', 'demos\nalone', 'prior\n+ demos']
means = [0.5, 9.0, 17.9]
errs = [0.5, 2.4, 1.2]
colors = ['#c62828', '#e65100', '#2e7d32']
ax3.bar(range(3), means, yerr=errs, capsize=3, color=colors, width=0.6,
        error_kw={'lw': 1.0, 'ecolor': '#37474f'})
ax3.set_title('finding the mode', fontsize=10, color='#2e7d32',
              fontweight='bold', loc='left')
ax3.text(2, means[2] + errs[2] + 0.6, '17.9', fontsize=9,
         color='#2e7d32', ha='center', fontweight='bold')
ax3.set_xticks(range(3))
ax3.set_xticklabels(arms, fontsize=7.6)
ax3.set_ylim(0, 21.5)
ax3.set_yticks([0, 5, 10, 15, 20])
ax3.set_ylabel('held-out success / 20', fontsize=7.6)
ax3.axhline(20, color='#90a4ae', lw=0.7, ls=':')
ax3.tick_params(labelsize=7.2)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

fig.savefig('../paper/figures/hero_bc.pdf', bbox_inches='tight')
print('wrote hero story figure')
