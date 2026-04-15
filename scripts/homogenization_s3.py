"""
Compute within-group pairwise cosine similarity for Sample 3 homogenization analysis.

This reproduces the bundled Sample 3 homogenization analysis (median split,
Welch's t-test) and saves item-level within-group similarity values to
`data/sample3/` so the reported d-values can be verified from the distributed
analysis file.

Output columns added to data/sample3/within_group_similarity.csv:
  - cd_group: "Low" or "High" (median split within task)
  - within_group_sim: mean pairwise cosine similarity to others in same CD group

Note: This script operates on the bundled `sample3_analysis.csv` file.
"""

import bootstrap_env  # noqa: F401
import os
import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(ROOT, 'data', 'sample3', 'sample3_analysis.csv')
OUTPUT = os.path.join(ROOT, 'data', 'sample3', 'within_group_similarity.csv')

TASK_LABELS = {1: 'social_media', 3: 'business_pitch', 5: 'interview'}

df = pd.read_csv(ANALYSIS, encoding='utf-8-sig')
print(f"Loaded {len(df)} observations")

# Encode responses
print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
all_embs = encoder.encode(df['response'].astype(str).tolist(), show_progress_bar=False, batch_size=64)
# Normalize
emb_norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
all_embs_n = all_embs / emb_norms

# Compute within-group similarity for the bundled Sample 3 analysis file.
df['cd_group'] = ''
df['within_group_sim'] = np.nan

for tn in sorted(df['task_number'].unique()):
    sub = df[df['task_number'] == tn].copy()
    idx = sub.index.values
    n = len(idx)
    if n < 15:
        continue

    task_embs = all_embs_n[idx]
    sim_mat = task_embs @ task_embs.T
    np.fill_diagonal(sim_mat, np.nan)

    # Median split on CD within task
    med = sub['mean_creative_direction'].median()
    groups = np.where(sub['mean_creative_direction'] <= med, 'Low', 'High')
    df.loc[idx, 'cd_group'] = groups

    label = TASK_LABELS.get(tn, '?')
    print(f"\nTask {tn} ({label}) N={n}, median CD={med:.2f}:")

    results = {}
    for grp in ['Low', 'High']:
        gmask = (groups == grp)
        if gmask.sum() < 3:
            continue
        within = sim_mat[np.ix_(gmask, gmask)].copy()
        np.fill_diagonal(within, np.nan)
        person_sims = np.nanmean(within, axis=1)
        grp_idx = idx[gmask]
        df.loc[grp_idx, 'within_group_sim'] = person_sims
        results[grp] = {'mean': np.nanmean(within), 'person_sims': person_sims, 'n': gmask.sum()}
        print(f"  {grp:4s} CD (N={gmask.sum():3d}): mean within-group sim = {np.nanmean(within):.4f}")

    if 'Low' in results and 'High' in results:
        t_val, t_p = stats.ttest_ind(results['Low']['person_sims'],
                                      results['High']['person_sims'],
                                      equal_var=False)
        diff = results['Low']['mean'] - results['High']['mean']
        d = diff / np.sqrt((np.var(results['Low']['person_sims']) +
                            np.var(results['High']['person_sims'])) / 2)
        sig = '***' if t_p < .001 else '**' if t_p < .01 else '*' if t_p < .05 else 'ns'
        print(f"  Low - High = {diff:+.4f}, d = {d:.3f}, t = {t_val:.3f}, p = {t_p:.4f} {sig}")

# Save the bundled analysis file with within-group similarity columns
df.rename(columns={'user_turns': 'n_user_turns'}, inplace=True)
out_cols = ['participant_id', 'task_number', 'task_label', 'word_count', 'n_user_turns',
            'paste_count', 'self_report_cc', 'mails_total',
            'anthropic_creative_direction', 'google_creative_direction', 'mistral_creative_direction',
            'mean_creative_direction',
            'crea_openai', 'crea_qwen', 'crea_xai', 'avg_creativity',
            'cd_group', 'within_group_sim']
out_df = df[out_cols].copy()
out_df.to_csv(OUTPUT, index=False)
print(f"\nSaved within-group similarity file: {len(out_df)} rows, {out_df.participant_id.nunique()} participants")

# Verify the bundled-data d-values from the saved file
print("\n" + "=" * 60)
print("VERIFICATION: Within-group similarity d-values from bundled data")
print("=" * 60)
osf2 = pd.read_csv(OUTPUT)
for task in ['social_media', 'business_pitch', 'interview']:
    sub = osf2[osf2['task_label'] == task].dropna(subset=['within_group_sim'])
    lo = sub[sub['cd_group'] == 'Low']['within_group_sim']
    hi = sub[sub['cd_group'] == 'High']['within_group_sim']
    t_val, t_p = stats.ttest_ind(lo, hi, equal_var=False)
    # Same d formula as pipeline: diff / sqrt(mean of variances)
    d = (lo.mean() - hi.mean()) / np.sqrt((lo.var() + hi.var()) / 2)
    sig = '***' if t_p < .001 else '**' if t_p < .01 else '*' if t_p < .05 else 'ns'
    print(f"  {task:15s}: d = {d:.2f}, p = {t_p:.4f} {sig}  (N_lo={len(lo)}, N_hi={len(hi)})")

