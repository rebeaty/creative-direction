"""Sample 1 and Sample 2 homogenization median splits."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer
from analysis_common import DATA_DIR

enc = SentenceTransformer("all-MiniLM-L6-v2")

def median_split_hom(df, cd_col, text_col, task_col, label):
    print(f"\n{label}:")
    for tname, sub in df.groupby(task_col):
        sub = sub.dropna(subset=[text_col, cd_col]).reset_index(drop=True)
        if len(sub) < 20:
            print(f"  task={tname}: N={len(sub)} (skipped, too small)")
            continue
        emb = enc.encode(sub[text_col].astype(str).tolist(), show_progress_bar=False, batch_size=64)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        sim = emb @ emb.T
        np.fill_diagonal(sim, np.nan)
        med = sub[cd_col].median()
        grp = np.where(sub[cd_col] <= med, "Low", "High")
        lows = np.where(grp == "Low")[0]; highs = np.where(grp == "High")[0]
        if len(lows) < 3 or len(highs) < 3:
            continue
        lo_sim = np.nanmean(sim[np.ix_(lows, lows)], axis=1)
        hi_sim = np.nanmean(sim[np.ix_(highs, highs)], axis=1)
        t, p = stats.ttest_ind(lo_sim, hi_sim, equal_var=False)
        d = (lo_sim.mean() - hi_sim.mean()) / np.sqrt((lo_sim.var() + hi_sim.var()) / 2)
        print(f"  task={tname}: N={len(sub)}, N_lo={len(lows)}, N_hi={len(highs)}, Lo sim={lo_sim.mean():.4f}, Hi sim={hi_sim.mean():.4f}, d={d:+.2f}, p={p:.4f}")

# Sample 1
s1_cr = pd.read_csv(os.path.join(DATA_DIR, "sample1", "creativity_scored_20260225_063157.csv"))
s1_proc = pd.read_csv(os.path.join(DATA_DIR, "sample1", "study1_process_20260224_221431.csv"))
s1 = s1_cr[s1_cr["condition"] == "chat"].merge(
    s1_proc[["user_id", "task", "mean_creative_direction"]],
    on=["user_id", "task"],
    how="inner",
)
median_split_hom(s1, "mean_creative_direction", "user_response", "task",
                  "[Sample 1] Homogenization median split (chat only)")

# Sample 2
s2_cr = pd.read_csv(os.path.join(DATA_DIR, "sample2", "creativity_scored_20260316_192451.csv"))
s2_cr = s2_cr.drop_duplicates(subset=["participant_id", "task_num"], keep="first")
s2_proc = pd.read_csv(os.path.join(DATA_DIR, "sample2", "study2_process_20260224_221913.csv"))
s2 = s2_cr.merge(s2_proc[["participant_id", "task_num", "mean_creative_direction"]],
                  on=["participant_id", "task_num"], how="inner")
# map task_num to label for readability
TASK_MAP = {1: "social_media", 2: "product_review", 3: "business_pitch", 5: "personal_challenge"}
s2["task_label"] = s2["task_num"].map(TASK_MAP)
median_split_hom(s2, "mean_creative_direction", "response", "task_label",
                  "[Sample 2] Homogenization median split (all 4 tasks)")
