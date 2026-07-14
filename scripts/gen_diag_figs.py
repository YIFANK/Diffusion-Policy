"""Redraw the two diagnostic figures as grouped dot-strips with
reader-facing labels (internal codenames removed).

fig:delta  -> figures/delta_ladder.pdf
fig:ladder -> figures/intervention_ladder.pdf
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 9})

# ---------------- delta ladder ----------------
fig, ax = plt.subplots(figsize=(7.4, 2.1))
groups = [
    ('collapsed text encoder', [1.2e-3], '#c62828', '= 0 (exact)'),
    ('image policies, text channel\n(before/after partial fixes)',
     [0.019, 0.086, 0.24], '#e65100', '0.02 - 0.24'),
    ('healthy channels (state)', [1.09, 2.15], '#2e7d32', '1.1 - 2.2'),
]
for gi, (lab, vals, col, note) in enumerate(groups):
    y = 2 - gi
    ax.scatter(vals, [y] * len(vals), s=64, c=col, zorder=5,
               edgecolors='white', linewidths=0.8)
    ax.text(6.5e-4, y + 0.28, lab, fontsize=8.6, color=col,
            va='bottom', fontweight='bold')
    ax.text(max(vals) * 1.5, y, note, fontsize=8.6, color=col,
            va='center', fontweight='bold')
ax.axvline(5.7, color='#90a4ae', lw=0.9, ls=':')
ax.text(5.7, 2.62, r'$\Vert\epsilon\Vert\approx5.7$', fontsize=7.6,
        color='#78909c', ha='center')
ax.set_xscale('log')
ax.set_xlim(5e-4, 12)
ax.set_ylim(-0.55, 2.95)
ax.set_yticks([])
ax.set_xlabel(r'conditional delta '
              r'$\Vert\epsilon_\theta(x,o,z)-\epsilon_\theta(x,o,\varnothing)'
              r'\Vert$  (log scale)', fontsize=8.6)
for s in ['top', 'right', 'left']:
    ax.spines[s].set_visible(False)
fig.savefig('../paper/figures/delta_ladder.pdf', bbox_inches='tight')

# ---------------- intervention strip ----------------
fig2, ax2 = plt.subplots(figsize=(7.4, 1.7))
fixes = [55, 57.5, 57.5, 52.5, 55]
ax2.scatter(fixes, [1] * len(fixes), s=64, c='#e65100', zorder=5,
            edgecolors='white', linewidths=0.8)
ax2.scatter([85], [0], s=80, c='#2e7d32', zorder=5,
            edgecolors='white', linewidths=0.8)
ax2.text(54.5, 1.42, 'image observations: every objective-level fix',
         fontsize=8.6, color='#e65100', fontweight='bold', ha='center')
ax2.text(85, 0.42, 'state observations,\nsame recipe', fontsize=8.6,
         color='#2e7d32', fontweight='bold', ha='center')
ax2.text(60.5, 1, '55 &pm; 3%'.replace('&pm;', '±'), fontsize=8.6,
         color='#e65100', va='center', fontweight='bold')
ax2.text(88.5, 0, '85%', fontsize=8.6, color='#2e7d32', va='center',
         fontweight='bold')
ax2.axvline(50, color='#90a4ae', lw=0.9, ls='--')
ax2.text(50, 1.95, 'chance', fontsize=7.6, color='#78909c', ha='center')
ax2.set_xlim(38, 100)
ax2.set_ylim(-0.7, 2.2)
ax2.set_yticks([])
ax2.set_xlabel('object-selection success (%)', fontsize=8.6)
for s in ['top', 'right', 'left']:
    ax2.spines[s].set_visible(False)
fig2.savefig('../paper/figures/intervention_ladder.pdf',
             bbox_inches='tight')
print('wrote both diagnostic figures')
