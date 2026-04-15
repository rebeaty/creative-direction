"""
Figure 1: Human validation - two-panel scatter (Sample 2).
  Panel A: Process (CD) - human vs LLM ratings (item-level)
  Panel B: Product (creativity) - human vs LLM ratings (item-level)
Output: figures/figure_validation.png
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import bootstrap_env  # noqa: F401

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

basedir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(basedir, 'scripts'))
from analysis_common import SEED, ensure_figure_dir, get_sample2_validation_data

outdir = ensure_figure_dir()

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.facecolor': '#F0F0F0',
    'axes.edgecolor': 'none',
    'axes.grid': True,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'grid.color': 'white',
    'grid.linewidth': 1.0,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.facecolor': 'white',
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.major.size': 0,
    'ytick.major.size': 0,
})

print('Loading Sample 2 validation data...')
validation = get_sample2_validation_data()
cd_merged = validation['process']
r_cd, p_cd = validation['process_r']
crea_merged = validation['product']
r_crea, p_crea = validation['product_r']
print(f'  Process: N = {len(cd_merged)} responses, r = {r_cd:.3f}, p = {p_cd:.2e}')
print(f'  Product: N = {len(crea_merged)} responses, r = {r_crea:.3f}, p = {p_crea:.2e}')

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.5))
rng = np.random.default_rng(SEED)

jx = cd_merged['human_cd'].to_numpy() + rng.normal(0, 0.04, len(cd_merged))
jy = cd_merged['ai_cd'].to_numpy() + rng.normal(0, 0.04, len(cd_merged))
ax_a.scatter(jx, jy, s=18, alpha=0.35, color='#4C72B0', edgecolors='none', zorder=3)
slope, intercept = np.polyfit(cd_merged['human_cd'], cd_merged['ai_cd'], 1)
xl = np.linspace(0.8, 5.2, 100)
ax_a.plot(xl, slope * xl + intercept, color='#C44E52', linewidth=1.8, zorder=4)
ax_a.text(0.05, 0.95, f'r = {r_cd:.2f}', transform=ax_a.transAxes,
          fontsize=10, va='top', ha='left', fontstyle='italic', color='#333333')
ax_a.text(0.05, 0.87, f'N = {len(cd_merged)} responses', transform=ax_a.transAxes,
          fontsize=8, va='top', ha='left', color='#666666')
ax_a.set_xlabel('Human CD rating (mean of 5 raters)')
ax_a.set_ylabel('LLM CD rating (mean of 3 models)')
ax_a.set_xlim(0.7, 5.3)
ax_a.set_ylim(0.7, 5.3)
ax_a.set_xticks([1, 2, 3, 4, 5])
ax_a.set_yticks([1, 2, 3, 4, 5])
ax_a.set_aspect('equal')
ax_a.text(-0.05, 1.08, 'A', transform=ax_a.transAxes,
          fontsize=14, fontweight='bold', va='top', ha='right')

jx2 = crea_merged['human_creativity'].to_numpy() + rng.normal(0, 0.04, len(crea_merged))
jy2 = crea_merged['avg_creativity'].to_numpy() + rng.normal(0, 0.4, len(crea_merged))
ax_b.scatter(jx2, jy2, s=18, alpha=0.35, color='#4C72B0', edgecolors='none', zorder=3)
slope2, intercept2 = np.polyfit(crea_merged['human_creativity'], crea_merged['avg_creativity'], 1)
xl2 = np.linspace(0.8, 5.2, 100)
ax_b.plot(xl2, slope2 * xl2 + intercept2, color='#C44E52', linewidth=1.8, zorder=4)
ax_b.text(0.05, 0.95, f'r = {r_crea:.2f}', transform=ax_b.transAxes,
          fontsize=10, va='top', ha='left', fontstyle='italic', color='#333333')
ax_b.text(0.05, 0.87, f'N = {len(crea_merged)} responses', transform=ax_b.transAxes,
          fontsize=8, va='top', ha='left', color='#666666')
ax_b.set_xlabel('Human creativity rating (mean of 5 raters)')
ax_b.set_ylabel('LLM creativity score (mean of 3 models)')
ax_b.set_xlim(0.7, 5.3)
ax_b.set_xticks([1, 2, 3, 4, 5])
ax_b.text(-0.05, 1.08, 'B', transform=ax_b.transAxes,
          fontsize=14, fontweight='bold', va='top', ha='right')

plt.tight_layout()
fig.savefig(os.path.join(outdir, 'figure_validation.png'),
            bbox_inches='tight', facecolor='white', dpi=300)
plt.close()
print('Saved figure_validation.png')
