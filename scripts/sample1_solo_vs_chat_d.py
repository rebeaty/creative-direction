"""Sample 1 solo vs chat Cohen's d, computed under multiple formulations."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy import stats
from analysis_common import DATA_DIR

s1 = pd.read_csv(os.path.join(DATA_DIR, "sample1", "creativity_scored_20260225_063157.csv"))
s1["avg_crea"] = s1[["openai_score","qwen_score","xai_score"]].mean(axis=1)
print(f"Sample 1 creativity_scored rows: {len(s1)}")
print(f"tasks: {sorted(s1['task'].unique())}, conditions: {sorted(s1['condition'].unique())}")
print(f"Unique participants: {s1['user_id'].nunique()}\n")

for task in sorted(s1["task"].unique()):
    sub = s1[s1["task"] == task]
    solo = sub[sub["condition"] == "solo"][["user_id", "avg_crea"]].dropna()
    chat = sub[sub["condition"] == "chat"][["user_id", "avg_crea"]].dropna()
    print(f"\n=== task={task} ===")
    print(f"  solo: N={len(solo)}, M={solo['avg_crea'].mean():.3f}, SD={solo['avg_crea'].std():.3f}")
    print(f"  chat: N={len(chat)}, M={chat['avg_crea'].mean():.3f}, SD={chat['avg_crea'].std():.3f}")

    # Paired data: merge on user_id
    paired = solo.merge(chat, on="user_id", suffixes=("_solo", "_chat"))
    print(f"  paired N={len(paired)}")
    diff = paired["avg_crea_chat"] - paired["avg_crea_solo"]
    mean_diff = diff.mean(); sd_diff = diff.std()
    r_sc = paired[["avg_crea_solo", "avg_crea_chat"]].corr().iloc[0,1]

    # d_z (paired / repeated measures)
    d_z = mean_diff / sd_diff
    # d_av: Cohen's d average (biased)
    s_pool_av = (paired["avg_crea_solo"].std() + paired["avg_crea_chat"].std()) / 2
    d_av = mean_diff / s_pool_av
    # d_rm: repeated-measures corrected
    s_pool = np.sqrt((paired["avg_crea_solo"].var() + paired["avg_crea_chat"].var()) / 2)
    d_rm = (mean_diff / s_pool) * np.sqrt(2*(1-r_sc))
    # d_ind: independent-samples Cohen's d
    s_pool_ind = np.sqrt(((len(solo)-1)*solo["avg_crea"].var() + (len(chat)-1)*chat["avg_crea"].var()) / (len(solo)+len(chat)-2))
    d_ind = mean_diff / s_pool_ind
    t_paired, p_paired = stats.ttest_rel(paired["avg_crea_chat"], paired["avg_crea_solo"])
    print(f"  diff = chat - solo = {mean_diff:+.3f}, sd(diff) = {sd_diff:.3f}, r(solo,chat) = {r_sc:.3f}")
    print(f"  d_z (paired):          {d_z:.3f}")
    print(f"  d_av (avg SD):         {d_av:.3f}")
    print(f"  d_rm (corrected paired): {d_rm:.3f}")
    print(f"  d_ind (independent):   {d_ind:.3f}")
    print(f"  t(paired) = {t_paired:.2f}, p = {p_paired:.2e}")
