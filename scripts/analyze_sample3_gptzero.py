"""
Run and summarize the preregistered GPTZero analysis for Sample 3.

If `data/sample3/gptzero_predictions.csv` is already present, this script uses
the bundled predictions and reports the corresponding item- and person-level
associations. If the file is missing and `GPTZERO_API_KEY` is set, the script
will query the GPTZero API, save the predictions, and then compute the summary
statistics.
"""

from __future__ import annotations

import bootstrap_env  # noqa: F401
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_PATH = os.path.join(ROOT, "data", "sample3", "sample3_analysis.csv")
PREDICTIONS_PATH = os.path.join(ROOT, "data", "sample3", "gptzero_predictions.csv")
SUMMARY_PATH = os.path.join(ROOT, "data", "sample3", "gptzero_summary.json")
API_URL = "https://api.gptzero.me/v2/predict/text"
MAX_WORKERS = 12


def _load_sample3() -> pd.DataFrame:
    frame = pd.read_csv(ANALYSIS_PATH, encoding="utf-8-sig").copy()
    return frame.reset_index(drop=True)


def _fetch_prediction(text: str, api_key: str) -> dict:
    response = requests.post(
        API_URL,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": api_key,
        },
        json={"document": text},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    document = payload["documents"][0]
    probs = document.get("class_probabilities", {})
    return {
        "version": payload.get("version"),
        "neat_version": payload.get("neatVersion"),
        "document_classification": document.get("document_classification"),
        "predicted_class": document.get("predicted_class"),
        "confidence_score": document.get("confidence_score"),
        "confidence_category": document.get("confidence_category"),
        "human_prob": probs.get("human"),
        "ai_prob": probs.get("ai"),
        "mixed_prob": probs.get("mixed"),
        "detect_prob": (probs.get("ai", 0.0) or 0.0) + (probs.get("mixed", 0.0) or 0.0),
        "completely_generated_prob": document.get("completely_generated_prob"),
        "average_generated_prob": document.get("average_generated_prob"),
        "result_message": document.get("result_message"),
        "raw_json": json.dumps(payload, ensure_ascii=True),
    }


def _generate_predictions(frame: pd.DataFrame, api_key: str) -> pd.DataFrame:
    rows: list[dict] = [None] * len(frame)  # type: ignore[assignment]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_prediction, str(row["response"]), api_key): idx
            for idx, row in frame.iterrows()
        }
        for future in as_completed(futures):
            idx = futures[future]
            rows[idx] = future.result()

    predictions = pd.DataFrame(rows)
    out = pd.concat(
        [
            frame[["participant_id", "task_number", "task_label", "mean_creative_direction", "avg_creativity"]]
            .reset_index(drop=True),
            predictions,
        ],
        axis=1,
    )
    out.to_csv(PREDICTIONS_PATH, index=False)
    return out


def _load_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(PREDICTIONS_PATH):
        preds = pd.read_csv(PREDICTIONS_PATH)
        if len(preds) == len(frame):
            return preds

    api_key = os.environ.get("GPTZERO_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing bundled GPTZero predictions and no GPTZERO_API_KEY provided."
        )

    print(f"Submitting {len(frame)} Sample 3 products to GPTZero...")
    return _generate_predictions(frame, api_key)


def _corr_summary(x: pd.Series, y: pd.Series) -> dict:
    r, p = stats.pearsonr(x, y)
    return {"r": float(r), "p": float(p), "n": int(len(x))}


def main() -> int:
    frame = _load_sample3()
    preds = _load_predictions(frame)
    merged = frame.merge(
        preds[
            [
                "participant_id",
                "task_number",
                "detect_prob",
                "ai_prob",
                "mixed_prob",
                "human_prob",
                "document_classification",
                "predicted_class",
                "confidence_score",
                "confidence_category",
                "completely_generated_prob",
                "average_generated_prob",
                "version",
                "neat_version",
            ]
        ],
        on=["participant_id", "task_number"],
        how="inner",
        validate="one_to_one",
    )

    item_level = _corr_summary(merged["mean_creative_direction"], merged["detect_prob"])

    person = (
        merged.groupby("participant_id")
        .agg(
            cd=("mean_creative_direction", "mean"),
            detect_prob=("detect_prob", "mean"),
            ai_prob=("ai_prob", "mean"),
            mixed_prob=("mixed_prob", "mean"),
            human_prob=("human_prob", "mean"),
        )
        .reset_index()
    )
    person_level = _corr_summary(person["cd"], person["detect_prob"])

    by_task = {}
    for task_number, task_df in merged.groupby("task_number"):
        label = str(task_df["task_label"].iloc[0])
        by_task[label] = _corr_summary(task_df["mean_creative_direction"], task_df["detect_prob"])

    summary = {
        "metric": "GPTZero non-human detection probability (ai + mixed)",
        "api_version": str(preds["version"].dropna().mode().iloc[0]) if preds["version"].notna().any() else None,
        "api_neat_version": str(preds["neat_version"].dropna().mode().iloc[0]) if preds["neat_version"].notna().any() else None,
        "item_level": item_level,
        "person_level": person_level,
        "by_task": by_task,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Sample 3 GPTZero analysis")
    print(f"  Metric: {summary['metric']}")
    print(f"  API version: {summary['api_version']} ({summary['api_neat_version']})")
    print(
        f"  Item level:   r = {item_level['r']:+.3f}, p = {item_level['p']:.4f}, N = {item_level['n']}"
    )
    print(
        f"  Person level: r = {person_level['r']:+.3f}, p = {person_level['p']:.4f}, N = {person_level['n']}"
    )
    for label, result in by_task.items():
        print(
            f"  {label}: r = {result['r']:+.3f}, p = {result['p']:.4f}, N = {result['n']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
