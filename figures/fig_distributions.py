"""
Figure: CD distributions across samples - density curves only.
Output: figures/figure_distributions.png
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import bootstrap_env  # noqa: F401

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

basedir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(basedir, 'scripts'))
from analysis_common import (
    ensure_figure_dir,
    load_gemini_person_level,
    load_sample1_person_level,
    load_sample2_person_level,
    load_wildchat_person_level,
)

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

COLORS = ['#E69F00', '#D55E00', '#009E73', '#0072B2']

print('Loading data...')
s1_person = load_sample1_person_level()['cd'].to_numpy()
s2_person = load_sample2_person_level()['cd'].to_numpy()
s4_person = load_gemini_person_level()['cd'].to_numpy()
s3_person = load_wildchat_person_level()['cd'].to_numpy()

samples = [
    {'data': s1_person, 'label': 'S1: ChatGPT, idea generation', 'color': COLORS[0]},
    {'data': s2_person, 'label': 'S2: ChatGPT, personal writing', 'color': COLORS[1]},
    {'data': s4_person, 'label': 'S3: Gemini, preregistered', 'color': COLORS[2]},
    {'data': s3_person, 'label': 'S4: WildChat, naturalistic', 'color': COLORS[3]},
]

for sample in samples:
    sample['mean'] = np.mean(sample['data'])
    sample['n'] = len(sample['data'])
    print(f"  {sample['label']}: M = {sample['mean']:.2f}, N = {sample['n']}")

fig, ax = plt.subplots(figsize=(4.5, 3.0))
ax.set_facecolor('white')
ax.grid(False)

xs = np.linspace(0.8, 5.2, 500)
for sample in samples:
    kde = gaussian_kde(sample['data'], bw_method=0.25)
    ys = kde(xs)
    ax.fill_between(xs, ys, alpha=0.15, color=sample['color'])
    ax.plot(xs, ys, color=sample['color'], alpha=0.9, lw=1.5,
            label=f"{sample['label']} (M = {sample['mean']:.2f}, N = {sample['n']})")
    ax.axvline(sample['mean'], color=sample['color'], ls='--', lw=0.7, alpha=0.6)

ax.legend(frameon=True, fancybox=False, edgecolor='#DDDDDD',
          fontsize=6.5, loc='upper center', handlelength=1.5,
          borderpad=0.6, labelspacing=0.4,
          bbox_to_anchor=(0.5, 1.22), ncol=2)
ax.set_xlabel('Creative Direction (person-level mean)')
ax.set_ylabel('Density')
ax.set_xlim(0.8, 5.2)

plt.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(os.path.join(outdir, 'figure_distributions.png'),
            bbox_inches='tight', facecolor='white', dpi=300)
plt.close()
print('Saved figure_distributions.png')
