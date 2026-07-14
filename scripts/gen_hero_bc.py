"""Generate paper/figures/hero_bc.pdf — Fig 1 panels (b) and (c).

(b) three interfaces into the frozen policy's mode space, annotated with
    headline numbers; (c) the composition ladder: prior alone / likelihood
    alone / posterior (warm-started inversion), pooled over held-out combos
    A and B at 20 eps x 5 reps (output/power_push_prior.json numbers).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'serif', 'font.size': 9})

fig = plt.figure(figsize=(8.6, 3.1))

# ---------------- (b) three interfaces ----------------
axb = fig.add_axes([0.01, 0.02, 0.55, 0.96])
axb.set_xlim(0, 10)
axb.set_ylim(0, 10)
axb.axis('off')
axb.set_title('(b) Three interfaces to the same latent modes',
              fontsize=9.5, fontweight='bold', loc='left')

routes = [
    (8.0, '#fbe9e7', '#c62828', 'compositional language\n'
     '"...upper-left, clockwise, gently"', 'held-out: 0.25/10'),
    (5.0, '#fff3e0', '#e65100', 'few-shot demo inversion\n'
     r'$z^*=\arg\max_z \log p(D|z)+\log p(z)$',
     'warm-started: 17.9/20'),
    (2.0, '#e8f5e9', '#2e7d32', 'unsupervised episode latents\n'
     '+ vector arithmetic', 'steered: 31/40, slows 27-131%'),
]
for y, fc, ec, label, note in routes:
    axb.add_patch(FancyBboxPatch((0.2, y - 0.9), 4.6, 1.8,
                                 boxstyle='round,pad=0.12',
                                 fc=fc, ec=ec, lw=1.2))
    axb.text(2.5, y, label, ha='center', va='center', fontsize=7.3)
    axb.add_patch(FancyArrowPatch((4.95, y), (6.75, 5.0),
                                  arrowstyle='-|>', mutation_scale=11,
                                  color=ec, lw=1.3,
                                  connectionstyle='arc3,rad=0.08'))
    tx, ty = 5.65, y + (0.75 if y >= 5 else -0.62)
    axb.text(tx, ty, note, fontsize=6.8, color=ec, fontweight='bold',
             ha='center')
axb.add_patch(FancyBboxPatch((6.85, 3.6), 2.9, 2.8,
                             boxstyle='round,pad=0.12',
                             fc='#e3f2fd', ec='#1565c0', lw=1.4))
axb.text(8.3, 5.35, 'frozen diffusion\npolicy', ha='center', va='center',
         fontsize=8.2, fontweight='bold', color='#0d47a1')
axb.text(8.3, 4.35, r'$\epsilon_\theta(x, o, z)$', ha='center',
         va='center', fontsize=8.5, color='#0d47a1')
axb.text(8.3, 3.1, 'steering toward\na target behavior', ha='center',
         va='top', fontsize=6.6, color='#455a64')

# ---------------- (c) the composition ladder ----------------
axc = fig.add_axes([0.63, 0.15, 0.36, 0.75])
arms = ['language\n(held-out)', 'prior alone\n(arithmetic)',
        'demos alone\n(K=4 cold)', 'prior + demos\n(K=4 warm)']
means = [0.5, 4.8, 9.0, 17.9]
errs = [0.5, 1.5, 2.4, 1.2]
colors = ['#c62828', '#6d4c41', '#e65100', '#2e7d32']
bars = axc.bar(range(4), means, yerr=errs, capsize=3,
               color=colors, width=0.62,
               error_kw={'lw': 1.0, 'ecolor': '#37474f'})
for i, (m, e) in enumerate(zip(means, errs)):
    axc.text(i, m + e + 0.45, f'{m:.1f}', ha='center', fontsize=8.5,
             fontweight='bold')
axc.set_xticks(range(4))
axc.set_xticklabels(arms, fontsize=6.8)
axc.set_ylim(0, 21.5)
axc.axhline(20, color='#90a4ae', lw=0.8, ls=':')
axc.text(3.45, 20.25, 'max', fontsize=6.3, color='#78909c', ha='right')
axc.set_ylabel('held-out combination success / 20', fontsize=7.5)
axc.set_title('(c) Composition = posterior inference', fontsize=9.5,
              fontweight='bold', loc='left')
axc.spines['top'].set_visible(False)
axc.spines['right'].set_visible(False)
axc.tick_params(labelsize=7.5)

fig.savefig('../paper/figures/hero_bc.pdf', bbox_inches='tight')
print('wrote paper/figures/hero_bc.pdf')
