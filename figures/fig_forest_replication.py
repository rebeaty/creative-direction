"""
Forest plot showing CD -> Creativity across all 4 samples.
Internal meta-analysis with fixed-effects summary.
Output: figures/figure_forest_replication.png
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import bootstrap_env  # noqa: F401

sys.stdout.reconfigure(encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

basedir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(basedir, 'scripts'))
from analysis_common import ensure_figure_dir, fisher_z_ci, fixed_effects_meta, get_reproduction_results

outpath = os.path.join(ensure_figure_dir(), 'figure_forest_replication.png')

C_MAIN = '#2E86C1'
C_SUMMARY = '#D64541'
C_GRAY = '#555555'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

result_rows = get_reproduction_results()
studies = []
for row in result_rows:
    label = row['forest_label'].replace(': ', ':\n', 1)
    studies.append((label, row['r'], row['n'], row['context']))

meta = fixed_effects_meta([
    {'label': row['forest_label'], 'r': row['r'], 'n': row['n'], 'context': row['context']}
    for row in result_rows
])

print('\n--- CD -> Creativity: Internal Meta-Analysis ---')
for label, r, n, ctx in studies:
    lo, hi, _, _ = fisher_z_ci(r, n)
    print(f'{label.split(chr(10))[0]:12s}  r = {r:+.3f} [{lo:+.3f}, {hi:+.3f}], N = {n:5d}, {ctx}')
print(
    f'{"Summary":12s}  r = {meta["r"]:+.3f} [{meta["ci_lo"]:+.3f}, {meta["ci_hi"]:+.3f}], '
    f'N = {meta["n_total"]}, p = {meta["p"]:.2e}'
)

fig, ax = plt.subplots(figsize=(6.5, 3.5))
y_positions = [4, 3, 2, 1]
labels = []

for i, (label, r, n, _ctx) in enumerate(studies):
    lo, hi, _, _ = fisher_z_ci(r, n)
    y = y_positions[i]
    ax.plot([lo, hi], [y, y], color=C_MAIN, linewidth=1.5, solid_capstyle='round')
    marker_size = 4 + n**0.5 / 8
    ax.plot(r, y, 'o', color=C_MAIN, markersize=marker_size, zorder=5)
    ax.text(0.52, y, f'r = {r:.2f} [{lo:.2f}, {hi:.2f}]', va='center', fontsize=8, color=C_GRAY)
    ax.text(0.72, y, f'N = {n:,}', va='center', fontsize=8, color=C_GRAY)
    label_short = label.split('\n')[0]
    label_detail = '\n'.join(label.split('\n')[1:])
    labels.append(f'{label_short}\n{label_detail}')

diamond_x = [meta['ci_lo'], meta['r'], meta['ci_hi'], meta['r']]
diamond_y = [0, 0.25, 0, -0.25]
ax.fill(diamond_x, diamond_y, color=C_SUMMARY, alpha=0.9, zorder=5)
ax.text(0.52, 0, f"r = {meta['r']:.2f} [{meta['ci_lo']:.2f}, {meta['ci_hi']:.2f}]",
        va='center', fontsize=8, fontweight='bold', color=C_SUMMARY)
ax.text(0.72, 0, f"N = {meta['n_total']:,}", va='center', fontsize=8,
        fontweight='bold', color=C_SUMMARY)

ax.axhline(0.55, color='#cccccc', linewidth=0.5, linestyle='-')
ax.axvline(0, color='#cccccc', linewidth=0.5, linestyle='--')
ax.set_yticks(y_positions + [0])
ax.set_yticklabels(labels + ['Fixed-effects\nsummary'], fontsize=8)
ax.set_xlabel('Pearson r (person-level)', fontsize=10)
ax.set_xlim(-0.1, 0.82)
ax.set_ylim(-0.7, 5)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)

plt.tight_layout()
fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
print(f'\nSaved to {outpath}')
plt.close()

