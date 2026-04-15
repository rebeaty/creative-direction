"""
Reproduce the reported results from the processed files in `data/`.

This script prints the main manuscript statistics and then runs the
remaining scripts used for the appendix analyses and figures.

Usage:
  python scripts/reproduce_results.py
"""

from __future__ import annotations

import bootstrap_env  # noqa: F401
import subprocess
import sys
import warnings
from pathlib import Path

from analysis_common import (
    fisher_z_ci,
    fixed_effects_meta,
    format_p_value,
    format_stars,
    get_reproduction_results,
    get_sample2_mediation_results,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

result_rows = get_reproduction_results()

print("=" * 80)
print("Creative Direction -> Creativity (4 samples)")
print("Product scoring ensemble: GPT-4.1, Qwen 3.5, Grok-3")
print("=" * 80)

for row in result_rows:
    suffix = " users" if row["code"] == "S4" else ""
    print(f"\n{row['code']}: {row['heading']} (N={row['n']}{suffix})")
    print(f"  Zero-order r(CD, creativity):  {row['r']:+.3f} (p {format_p_value(row['p'])})")
    print(
        f"  Partial r(CD|turns):           "
        f"{row['partial_cd']:+.3f} (p {format_p_value(row['partial_cd_p'])})"
    )
    print(
        f"  Partial r(turns|CD):           "
        f"{row['partial_turns']:+.3f} (p {format_p_value(row['partial_turns_p'])})"
    )

print("\n" + "=" * 80)
print("INTERNAL META-ANALYSIS (Fixed Effects)")
print("=" * 80)

forest_rows = []
for row in result_rows:
    lo, hi, _, _ = fisher_z_ci(row["r"], row["n"])
    print(f"{row['forest_label']:35s}  r = {row['r']:+.3f} [{lo:+.3f}, {hi:+.3f}], N = {row['n']:5d}")
    forest_rows.append({"label": row["forest_label"], "r": row["r"], "n": row["n"], "context": row["context"]})

meta = fixed_effects_meta(forest_rows)
print(
    f'{"Summary":35s}  r = {meta["r"]:+.3f} [{meta["ci_lo"]:+.3f}, {meta["ci_hi"]:+.3f}], '
    f'N = {meta["n_total"]:5d}, p = {meta["p"]:.2e}'
)
print(f'Q = {meta["q"]:.2f}, df = {meta["df_q"]}, p = {meta["p_q"]:.3f}, I2 = {meta["i2"]:.1f}%')

print("\n" + "=" * 80)
print("MEDIATION ANALYSIS (Sample 2)")
print("=" * 80)

mediation = get_sample2_mediation_results()
print(f"\nMediation N = {mediation['n']}")
for label in ["g", "c"]:
    result = mediation[label]
    print(f"  r({label}, CD) = {result['r_iv_cd']:.3f},  r({label}, creativity) = {result['r_iv_creativity']:.3f}")

for label in ["g", "c"]:
    result = mediation[label]
    print(f"\n  {label} mediation (standardized):")
    print(f"    a ({label}->CD):         {result['a']:.3f}")
    print(f"    b (CD->creativity):     {result['b']:.3f}")
    print(f"    c' (direct):            {result['cp']:.3f}")
    print(f"    c  (total):             {result['total']:.3f}")
    print(f"    ab (indirect):          {result['indirect']:.3f}")

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f'\n{"Sample":<25s} {"N":>5s} {"r(CD,Crea)":>12s} {"pr(CD|turns)":>14s} {"pr(turns|CD)":>14s}')
print("-" * 72)
for row in result_rows:
    print(
        f"{row['code']:<25s} {row['n']:>5d} "
        f"{row['r']:>+.3f}        "
        f"{row['partial_cd']:>+.3f}{format_stars(row['partial_cd_p']):<6s}    "
        f"{row['partial_turns']:>+.3f}{format_stars(row['partial_turns_p']):<6s}"
    )

FOLLOW_UP_COMMANDS = [
    ("CFA validation", [sys.executable, "scripts/cfa_human_ratings.py"]),
    ("Sample 3 homogenization", [sys.executable, "scripts/homogenization_s3.py"]),
    ("Sample 3 GPTZero exploratory analysis", [sys.executable, "scripts/analyze_sample3_gptzero.py"]),
    ("Figure 1 validation", [sys.executable, "figures/fig_validation.py"]),
    ("Figure 2 forest plot", [sys.executable, "figures/fig_forest_replication.py"]),
    ("Figure 3 mediation", [sys.executable, "figures/fig_mediation.py"]),
    ("Figure 4 distributions", [sys.executable, "figures/fig_distributions.py"]),
]

for label, command in FOLLOW_UP_COMMANDS:
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
