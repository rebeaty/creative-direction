"""
Shared loaders and analysis helpers for the bundled analysis pipeline.

The `data/` directory contains processed files sufficient to reproduce the
reported statistics without rerunning API-based scoring pipelines.
"""

from __future__ import annotations

import bootstrap_env  # noqa: F401
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import lstsq
from scipy import stats

SEED = 99
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")


def ensure_figure_dir() -> str:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    return FIGURE_DIR


def zscore(values: pd.Series | np.ndarray) -> pd.Series:
    series = pd.Series(values, dtype=float)
    sd = series.std(ddof=1)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - series.mean()) / sd


def partial_corr(x, y, covar):
    """Pearson correlation between x and y after residualizing on covar."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    covar = np.asarray(covar, dtype=float)
    design = np.column_stack([np.ones(len(x)), covar])
    bx, _, _, _ = lstsq(design, x, rcond=None)
    by, _, _, _ = lstsq(design, y, rcond=None)
    rx = x - design @ bx
    ry = y - design @ by
    return stats.pearsonr(rx, ry)


def fisher_z_ci(r: float, n: int, alpha: float = 0.05):
    z_val = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z_val - z_crit * se)
    hi = np.tanh(z_val + z_crit * se)
    return lo, hi, z_val, se


def fixed_effects_meta(rows: List[Dict[str, float]]) -> Dict[str, float]:
    z_vals = []
    se_vals = []
    for row in rows:
        _, _, z_val, se = fisher_z_ci(row["r"], row["n"])
        z_vals.append(z_val)
        se_vals.append(se)

    z_arr = np.asarray(z_vals)
    se_arr = np.asarray(se_vals)
    weights = 1 / se_arr**2
    z_summary = np.sum(weights * z_arr) / np.sum(weights)
    se_summary = 1 / np.sqrt(np.sum(weights))
    z_crit = stats.norm.ppf(0.975)
    q_stat = np.sum(weights * (z_arr - z_summary) ** 2)
    df_q = len(z_arr) - 1

    return {
        "r": np.tanh(z_summary),
        "ci_lo": np.tanh(z_summary - z_crit * se_summary),
        "ci_hi": np.tanh(z_summary + z_crit * se_summary),
        "p": 2 * (1 - stats.norm.cdf(abs(z_summary / se_summary))),
        "n_total": int(sum(row["n"] for row in rows)),
        "q": q_stat,
        "df_q": df_q,
        "p_q": 1 - stats.chi2.cdf(q_stat, df_q),
        "i2": max(0.0, (q_stat - df_q) / q_stat * 100) if q_stat > 0 else 0.0,
    }


def format_p_value(p: float) -> str:
    if p < 0.001:
        return "< .001"
    if p < 0.05:
        return f"= {p:.3f}"
    return f"= {p:.3f} (ns)"


def format_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _count_sample1_user_turns(transcript: str) -> int:
    if pd.isna(transcript):
        return 0
    text = str(transcript)
    return text.count("\nUSER:") + int(text.startswith("USER:"))


def _count_sample2_user_turns(transcript: str) -> int:
    if pd.isna(transcript):
        return 0
    return str(transcript).count("\nUSER:")


def load_sample1_items() -> pd.DataFrame:
    process_df = pd.read_csv(os.path.join(DATA_DIR, "sample1", "study1_process_20260224_221431.csv"))
    creativity_df = pd.read_csv(os.path.join(DATA_DIR, "sample1", "creativity_scored_20260225_063157.csv"))
    creativity_df["avg_creativity"] = creativity_df[["openai_score", "qwen_score", "xai_score"]].mean(axis=1)
    merged = process_df.merge(
        creativity_df[["user_id", "task", "condition", "avg_creativity"]],
        on=["user_id", "task", "condition"],
        how="inner",
    )
    merged = merged[merged["condition"] == "chat"].copy()
    merged["n_user_turns"] = merged["chat_transcript"].apply(_count_sample1_user_turns)
    return merged


def load_sample1_person_level() -> pd.DataFrame:
    items = load_sample1_items()
    return (
        items.groupby("user_id")
        .agg(
            cd=("mean_creative_direction", "mean"),
            creativity=("avg_creativity", "mean"),
            n_user_turns=("n_user_turns", "mean"),
        )
        .dropna()
        .reset_index()
    )


def load_sample2_items() -> pd.DataFrame:
    process_df = pd.read_csv(os.path.join(DATA_DIR, "sample2", "study2_process_20260224_221913.csv"))
    creativity_df = pd.read_csv(os.path.join(DATA_DIR, "sample2", "creativity_scored_20260316_192451.csv"))
    creativity_df = creativity_df.drop_duplicates(subset=["participant_id", "task_num"], keep="first").copy()
    creativity_df["avg_creativity"] = creativity_df[["openai_score", "qwen_score", "xai_score"]].mean(axis=1)
    merged = process_df.merge(
        creativity_df[["participant_id", "task_num", "avg_creativity"]],
        on=["participant_id", "task_num"],
        how="inner",
        validate="one_to_one",
    )
    merged["n_user_turns"] = process_df["chat_transcript"].apply(_count_sample2_user_turns)
    return merged


def load_sample2_person_level() -> pd.DataFrame:
    items = load_sample2_items()
    return (
        items.groupby("participant_id")
        .agg(
            cd=("mean_creative_direction", "mean"),
            creativity=("avg_creativity", "mean"),
            n_user_turns=("n_user_turns", "mean"),
        )
        .dropna()
        .reset_index()
    )


def load_wildchat_items() -> pd.DataFrame:
    process_df = pd.read_csv(os.path.join(DATA_DIR, "sample4", "sample4_process_full_20260225_082835.csv"))
    creativity_df = pd.read_csv(os.path.join(DATA_DIR, "sample4", "sample4_creativity_20260306.csv"))
    ip_lookup = pd.read_csv(os.path.join(DATA_DIR, "sample4", "wildchat_hashed_ip_lookup.csv"))
    creativity_df["avg_creativity"] = creativity_df[["openai_score", "qwen_score", "xai_score"]].mean(axis=1)
    analytic_hashes = set(creativity_df["conversation_hash"])
    filtered = process_df[process_df["conversation_hash"].isin(analytic_hashes)].copy()
    filtered = filtered.drop(columns=["hashed_ip"], errors="ignore")
    merged = filtered.merge(
        creativity_df[["conversation_hash", "avg_creativity"]],
        on="conversation_hash",
        how="inner",
    )
    merged = merged.merge(ip_lookup, on="conversation_hash", how="left")
    return merged


def load_wildchat_person_level() -> pd.DataFrame:
    items = load_wildchat_items()
    return (
        items.groupby("hashed_ip")
        .agg(
            cd=("mean_creative_direction", "mean"),
            creativity=("avg_creativity", "mean"),
            n_user_turns=("turn_count", "mean"),
        )
        .dropna()
        .reset_index()
    )


def load_gemini_person_level() -> pd.DataFrame:
    gemini = pd.read_csv(os.path.join(DATA_DIR, "sample3", "sample3_analysis.csv"))
    turns = gemini.groupby("participant_id").agg(n_user_turns=("user_turns_num", "mean")).reset_index()
    person = gemini.groupby("participant_id").agg(
        cd=("mean_creative_direction", "mean"),
        creativity=("avg_creativity", "mean"),
    ).reset_index()
    return person.merge(turns, on="participant_id", how="inner").dropna()


def get_reproduction_results() -> List[Dict[str, object]]:
    samples = [
        ("S1", "Lab idea generation, ChatGPT", "S1: ChatGPT idea generation", "Controlled", load_sample1_person_level()),
        ("S2", "Lab personal writing, ChatGPT", "S2: ChatGPT personal writing", "Controlled", load_sample2_person_level()),
        ("S3", "Preregistered Gemini", "S3: Gemini preregistered", "Controlled", load_gemini_person_level()),
        ("S4", "WildChat naturalistic", "S4: WildChat naturalistic", "Naturalistic", load_wildchat_person_level()),
    ]

    results = []
    for code, heading, forest_label, context, frame in samples:
        r0, p0 = stats.pearsonr(frame["cd"], frame["creativity"])
        r_cd, p_cd = partial_corr(frame["cd"], frame["creativity"], frame["n_user_turns"])
        r_turns, p_turns = partial_corr(frame["n_user_turns"], frame["creativity"], frame["cd"])
        results.append(
            {
                "code": code,
                "heading": heading,
                "forest_label": forest_label,
                "context": context,
                "person": frame,
                "n": len(frame),
                "r": float(r0),
                "p": float(p0),
                "partial_cd": float(r_cd),
                "partial_cd_p": float(p_cd),
                "partial_turns": float(r_turns),
                "partial_turns_p": float(p_turns),
            }
        )
    return results


def get_replication_results() -> List[Dict[str, object]]:
    """Backward-compatible alias."""
    return get_reproduction_results()


def get_sample2_validation_data():
    process_df = pd.read_csv(os.path.join(DATA_DIR, "sample2", "study2_process_20260224_221913.csv"))
    items = load_sample2_items()
    human_ratings = pd.read_csv(os.path.join(DATA_DIR, "sample2", "human_ratings.csv"))

    process_ratings = human_ratings[
        (human_ratings["condition"] == "process") & (human_ratings["no_interaction"] == 0)
    ]
    human_cd = (
        process_ratings.groupby(["participant_id", "task_num"])["rating"]
        .mean()
        .reset_index(name="human_cd")
    )
    ai_cd = (
        process_df[["participant_id", "task_num", "mean_creative_direction"]]
        .drop_duplicates()
        .rename(columns={"mean_creative_direction": "ai_cd"})
    )
    cd_merged = human_cd.merge(ai_cd, on=["participant_id", "task_num"], how="inner").dropna()

    product_ratings = human_ratings[
        (human_ratings["condition"] == "product") & (human_ratings["no_interaction"] == 0)
    ]
    human_creativity = (
        product_ratings.groupby(["participant_id", "task_num"])["rating"]
        .mean()
        .reset_index(name="human_creativity")
    )
    ai_creativity = items[["participant_id", "task_num", "avg_creativity"]].drop_duplicates()
    creativity_merged = human_creativity.merge(
        ai_creativity, on=["participant_id", "task_num"], how="inner"
    ).dropna()

    return {
        "process": cd_merged,
        "process_r": stats.pearsonr(cd_merged["human_cd"], cd_merged["ai_cd"]),
        "product": creativity_merged,
        "product_r": stats.pearsonr(
            creativity_merged["human_creativity"],
            creativity_merged["avg_creativity"],
        ),
    }


def _sample2_mediation_frame() -> pd.DataFrame:
    items = load_sample2_items()
    person = (
        items.groupby("participant_id")
        .agg(
            cd=("mean_creative_direction", "mean"),
            creativity=("avg_creativity", "mean"),
        )
        .reset_index()
    )
    aggregates = pd.read_csv(os.path.join(DATA_DIR, "sample2", "aggregates_master_final.csv"))
    frame = person.merge(aggregates, on="participant_id", how="inner")

    g_cols = ["letter_sets_percentage", "vocab_percentage", "verbal_fluency_total"]
    c_cols = ["sctt", "design", "drawing", "metaphor_avg"]

    for col in g_cols + c_cols:
        frame[col + "_z"] = zscore(pd.to_numeric(frame[col], errors="coerce"))

    frame["g"] = frame[[col + "_z" for col in g_cols]].mean(axis=1, skipna=True)
    frame["c"] = frame[[col + "_z" for col in c_cols]].mean(axis=1, skipna=True)
    return frame.dropna(subset=["g", "c", "cd", "creativity"]).reset_index(drop=True)


def _standardized_mediation_paths(frame: pd.DataFrame, iv_col: str) -> Dict[str, float]:
    data = frame[[iv_col, "cd", "creativity"]].copy()
    for col in [iv_col, "cd", "creativity"]:
        data[col] = zscore(data[col]).to_numpy()

    r_xm = float(data[iv_col].corr(data["cd"]))
    r_xy = float(data[iv_col].corr(data["creativity"]))
    r_my = float(data["cd"].corr(data["creativity"]))
    denom = 1 - r_xm**2
    b = (r_my - r_xy * r_xm) / denom
    cp = (r_xy - r_my * r_xm) / denom

    return {
        "a": r_xm,
        "b": b,
        "cp": cp,
        "total": r_xy,
        "indirect": r_xm * b,
    }


def _bias_corrected_indirect_ci(frame: pd.DataFrame, iv_col: str, n_boot: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    observed = _standardized_mediation_paths(frame, iv_col)["indirect"]
    indices = np.arange(len(frame))
    boot = np.empty(n_boot, dtype=float)

    for idx in range(n_boot):
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        boot[idx] = _standardized_mediation_paths(frame.iloc[sample_idx], iv_col)["indirect"]

    prop_less = np.clip(np.mean(boot < observed), 1e-6, 1 - 1e-6)
    z0 = stats.norm.ppf(prop_less)
    alpha_lo = float(np.clip(stats.norm.cdf(2 * z0 + stats.norm.ppf(0.025)), 0, 1))
    alpha_hi = float(np.clip(stats.norm.cdf(2 * z0 + stats.norm.ppf(0.975)), 0, 1))
    ci_lo, ci_hi = np.quantile(boot, [alpha_lo, alpha_hi])
    return {"ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}


def get_sample2_mediation_results(n_boot: int = 0, seed: int = SEED) -> Dict[str, object]:
    frame = _sample2_mediation_frame()
    results = {"n": len(frame)}

    for iv_col in ["g", "c"]:
        corr_cd = stats.pearsonr(frame[iv_col], frame["cd"])[0]
        corr_creativity = stats.pearsonr(frame[iv_col], frame["creativity"])[0]

        standardized = _standardized_mediation_paths(frame, iv_col)
        data = frame[[iv_col, "cd", "creativity"]].copy()
        for col in [iv_col, "cd", "creativity"]:
            data[col] = zscore(data[col])

        a_model = sm.OLS(data["cd"], sm.add_constant(data[iv_col])).fit()
        bc_model = sm.OLS(data["creativity"], sm.add_constant(data[[iv_col, "cd"]])).fit()

        summary = {
            **standardized,
            "r_iv_cd": float(corr_cd),
            "r_iv_creativity": float(corr_creativity),
            "a_p": float(a_model.pvalues[iv_col]),
            "b_p": float(bc_model.pvalues["cd"]),
            "cp_p": float(bc_model.pvalues[iv_col]),
        }

        if n_boot:
            summary.update(_bias_corrected_indirect_ci(frame, iv_col, n_boot=n_boot, seed=seed))

        results[iv_col] = summary

    return results
