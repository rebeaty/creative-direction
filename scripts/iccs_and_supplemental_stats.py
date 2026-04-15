"""Per-sample process and product ICC(2,k), Sample 3 self-report and MAILS correlations, and the Sample 1 design-task delegation floor."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from analysis_common import DATA_DIR, load_sample1_items

def icc2k(df, raters):
    d = df.dropna(subset=raters).reset_index(drop=True).copy()
    d["_s"] = d.index
    long = d.melt(id_vars=["_s"], value_vars=raters, var_name="rater", value_name="score")
    tab = pg.intraclass_corr(data=long, targets="_s", raters="rater", ratings="score")
    row = tab[tab["Type"] == "ICC2k"].iloc[0]
    ci = row["CI95%"]
    return row["ICC"], ci[0], ci[1], len(d)

print("=" * 70)
print("PROCESS (CD) ICC(2,k) per sample — manuscript Table S3: .915 .921 .930 .846")
print("=" * 70)
s1_proc = pd.read_csv(os.path.join(DATA_DIR, "sample1", "study1_process_20260224_221431.csv"))
s2_proc = pd.read_csv(os.path.join(DATA_DIR, "sample2", "study2_process_20260224_221913.csv"))
s3 = pd.read_csv(os.path.join(DATA_DIR, "sample3", "sample3_analysis.csv"))
s4_proc = pd.read_csv(os.path.join(DATA_DIR, "sample4", "sample4_process_full_20260225_082835.csv"))

for name, df in [("S1", s1_proc), ("S2", s2_proc), ("S3", s3), ("S4", s4_proc)]:
    cols = [c for c in df.columns if c.endswith("_creative_direction") and c != "mean_creative_direction"]
    if len(cols) < 3:
        print(f"  {name}: ratings cols = {cols} (skipping)")
        continue
    icc, lo, hi, n = icc2k(df, cols)
    print(f"  {name}: ICC(2,k) = {icc:.3f} [{lo:.3f}, {hi:.3f}]  N={n}  raters={cols}")

print()
print("=" * 70)
print("PRODUCT ICC(2,k) per sample — SI §3: .882 .902 .899 .773")
print("=" * 70)
s1_cr = pd.read_csv(os.path.join(DATA_DIR, "sample1", "creativity_scored_20260225_063157.csv"))
s2_cr = pd.read_csv(os.path.join(DATA_DIR, "sample2", "creativity_scored_20260316_192451.csv"))
s4_cr = pd.read_csv(os.path.join(DATA_DIR, "sample4", "sample4_creativity_20260306.csv"))
for name, df in [("S1", s1_cr), ("S2", s2_cr), ("S3", s3), ("S4", s4_cr)]:
    cols = [c for c in ["openai_score", "qwen_score", "xai_score", "crea_openai", "crea_qwen", "crea_xai"] if c in df.columns]
    if len(cols) < 3:
        print(f"  {name}: no product rater cols")
        continue
    icc, lo, hi, n = icc2k(df, cols)
    print(f"  {name}: ICC(2,k) = {icc:.3f} [{lo:.3f}, {hi:.3f}]  N={n}")

print()
print("=" * 70)
print("SAMPLE 3 SELF-REPORT CD + MAILS (SI §8.5)")
print("=" * 70)
print(f"  columns include: self_report_cc={'self_report_cc' in s3.columns}, mails_total={'mails_total' in s3.columns}, paste_count={'paste_count' in s3.columns}")
if {"self_report_cc","mails_total","paste_count"}.issubset(s3.columns):
    person = s3.groupby("participant_id").agg(
        sr=("self_report_cc","mean"),
        mails=("mails_total","mean"),
        cd=("mean_creative_direction","mean"),
        crea=("avg_creativity","mean"),
        paste=("paste_count","mean"),
    ).dropna(subset=["sr", "cd", "mails", "crea"])
    print(f"  Self-report CD composite: M={person['sr'].mean():.3f}, SD={person['sr'].std():.3f}, N={len(person)}  [mss: M=5.70, SD=0.76]")
    print(f"  MAILS: M={person['mails'].mean():.3f}, SD={person['mails'].std():.3f}  [mss: M=5.70, SD=1.55]")
    print(f"  r(self-report CD, LLM CD) = {stats.pearsonr(person['sr'], person['cd'])[0]:+.3f}, p={stats.pearsonr(person['sr'], person['cd'])[1]:.4f}  [mss: .26, .001]")
    print(f"  r(MAILS, LLM CD) = {stats.pearsonr(person['mails'], person['cd'])[0]:+.3f}, p={stats.pearsonr(person['mails'], person['cd'])[1]:.4f}  [mss: .12, .14]")
    print(f"  r(MAILS, creativity) = {stats.pearsonr(person['mails'], person['crea'])[0]:+.3f}, p={stats.pearsonr(person['mails'], person['crea'])[1]:.4f}  [mss: .22, .006]")
    # Copy-paste: manuscript says pasted on 46.0% of tasks, M = 0.74 per task
    paste_pct = (s3["paste_count"] > 0).mean() * 100
    paste_M = s3["paste_count"].mean(); paste_SD = s3["paste_count"].std()
    print(f"  Copy-paste: pasted on {paste_pct:.1f}% of tasks, M={paste_M:.2f}, SD={paste_SD:.2f}  [mss: 46.0%, M=0.74, SD=1.08]")
    # Self-report McDonald's omega via factor loadings from pingouin? skip; just report Cronbach's alpha as proxy
    sr_items = [c for c in s3.columns if c.startswith("self_report") and c != "self_report_cc"]
    print(f"  Self-report items: {sr_items}")

print()
print("=" * 70)
print("SAMPLE 1 — design task 48% floor delegation")
print("=" * 70)
s1_items = load_sample1_items()
for task in sorted(s1_items["task"].unique()):
    sub = s1_items[s1_items["task"] == task]
    cd = sub["mean_creative_direction"]
    floor = (cd <= 1.5).sum(); n = len(sub)
    print(f"  task={task}: N={n}, mean CD = {cd.mean():.2f}, CD <= 1.5: {floor}/{n} = {100*floor/n:.1f}%  [mss for design: 48% at floor]")
