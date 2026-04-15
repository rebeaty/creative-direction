"""
CFA of Human Ratings: Two-Factor Measurement Model
====================================================
Beaty & DiStefano

Fits a two-factor confirmatory factor analysis to person-level human
ratings from Sample 2. Process CD (5 raters) and Product Creativity
(5 raters) are modeled as separate latent factors rated by independent
panels, eliminating shared method variance.

Reports: model fit indices, standardized loadings, latent correlation,
and inter-rater reliability (ICC(3,k)).

    python scripts/cfa_human_ratings.py

Requirements: pandas, numpy, semopy, pingouin (optional, for ICC)
"""
import bootstrap_env  # noqa: F401
import os, sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from semopy import Model
from semopy.stats import calc_stats

basedir = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(basedir, 'data', 'sample2')

# ── Load data ──
ratings = pd.read_csv(os.path.join(DATA, 'human_ratings.csv'))

# Separate process and product ratings
proc = ratings[ratings['condition'] == 'process']
prod = ratings[ratings['condition'] == 'product']

# Pivot to wide format: one column per rater
proc_wide = proc.pivot_table(
    index=['participant_id', 'task_num'], columns='rater_id', values='rating'
)
proc_wide.columns = [f'h{i+1}' for i in range(len(proc_wide.columns))]
proc_wide = proc_wide.reset_index()

prod_wide = prod.pivot_table(
    index=['participant_id', 'task_num'], columns='rater_id', values='rating'
)
prod_wide.columns = [f'r{i+1}' for i in range(len(prod_wide.columns))]
prod_wide = prod_wide.reset_index()

# Merge and aggregate to person level
sem_data = proc_wide.merge(prod_wide, on=['participant_id', 'task_num'], how='inner')
score_cols = [c for c in sem_data.columns if c not in ['participant_id', 'task_num']]
person = sem_data.groupby('participant_id')[score_cols].mean().reset_index().dropna()

print(f'N = {len(person)} participants')
print(f'Process raters: 5 (h1-h5), Product raters: 5 (r1-r5)')
print()

# ── Inter-rater reliability: ICC(3,k) ──
print('=' * 60)
print('INTER-RATER RELIABILITY — ICC(3,k)')
print('=' * 60)

try:
    import pingouin as pg

    for label, condition, prefix in [('Process CD', 'process', 'h'), ('Product Creativity', 'product', 'r')]:
        df_long = ratings[ratings['condition'] == condition].copy()
        icc = pg.intraclass_corr(
            data=df_long, targets='item_id', raters='rater_id', ratings='rating'
        )
        row = icc[icc['Type'] == 'ICC3k']
        val = float(row['ICC'].values[0])
        ci_lo = float(row['CI95%'].values[0][0])
        ci_hi = float(row['CI95%'].values[0][1])
        print(f'  {label}: ICC(3,k) = {val:.2f}, 95% CI [{ci_lo:.2f}, {ci_hi:.2f}]')
except ImportError:
    print('  (pingouin not installed — skipping ICC)')

print()

# ── Two-factor CFA ──
print('=' * 60)
print('TWO-FACTOR CFA')
print('=' * 60)

model_spec = """
ProcessCD =~ h1 + h2 + h3 + h4 + h5
ProductCreativity =~ r1 + r2 + r3 + r4 + r5
ProcessCD ~~ ProductCreativity
"""

m = Model(model_spec)
m.fit(person)
est = m.inspect()

# Fit indices
fit = calc_stats(m)
chi2 = float(fit['chi2'].values[0])
chi2_df = int(float(fit['DoF'].values[0]))
cfi = float(fit['CFI'].values[0])
tli = float(fit['TLI'].values[0])
rmsea = float(fit['RMSEA'].values[0])

print(f'  chi-squared({chi2_df}) = {chi2:.2f}')
print(f'  CFI = {cfi:.3f}')
print(f'  TLI = {tli:.3f}')
print(f'  RMSEA = {rmsea:.3f}')
print()

# Standardized loadings
# semopy reports: indicator ~ latent (lval=indicator, rval=latent)
print('Standardized loadings:')
loadings = est[est['op'] == '~']
for _, row in loadings.iterrows():
    ind = row['lval']   # indicator (observed variable)
    lv = row['rval']    # latent variable
    lam = float(row['Estimate'])
    var_resid = float(est[(est['op'] == '~~') & (est['lval'] == ind) & (est['rval'] == ind)]['Estimate'].values[0])
    var_lv = float(est[(est['op'] == '~~') & (est['lval'] == lv) & (est['rval'] == lv)]['Estimate'].values[0])
    std_loading = lam * np.sqrt(var_lv) / np.sqrt(lam**2 * var_lv + var_resid)
    print(f'  {lv} -> {ind}: std = {std_loading:.2f}')

print()

# Latent correlation
cov_row = est[(est['op'] == '~~') & (est['lval'] == 'ProcessCD') & (est['rval'] == 'ProductCreativity')]
if len(cov_row) == 0:
    cov_row = est[(est['op'] == '~~') & (est['lval'] == 'ProductCreativity') & (est['rval'] == 'ProcessCD')]

cov_val = float(cov_row['Estimate'].values[0])
var_proc = float(est[(est['op'] == '~~') & (est['lval'] == 'ProcessCD') & (est['rval'] == 'ProcessCD')]['Estimate'].values[0])
var_prod = float(est[(est['op'] == '~~') & (est['lval'] == 'ProductCreativity') & (est['rval'] == 'ProductCreativity')]['Estimate'].values[0])
lat_r = cov_val / np.sqrt(var_proc * var_prod)

z_val = float(cov_row['z-value'].values[0])
p_val = float(cov_row['p-value'].values[0])

print(f'Latent correlation: r = {lat_r:.3f}, z = {z_val:.3f}, p = {p_val:.6f}')
