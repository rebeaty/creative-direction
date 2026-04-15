# Creative Direction

Processed data and analysis code for:

> Beaty, R. E., & DiStefano, P. V. Human direction drives creativity with large language models.

This repository contains the processed data and analysis code for the manuscript and SI.

## Quick Start

Create a Python environment and install:

```bash
conda env create -f environment.yml
conda activate creative-direction-repro
```

Or install into an existing environment:

```bash
pip install -r requirements.txt
```

For reproducible setup, use a fresh environment rather than an existing base
Python or Conda installation.

Run the full analysis pipeline:

```bash
python scripts/reproduce_results.py
```

This runs:

- the four main CD-creativity correlations
- the fixed-effects meta-analysis
- discriminant-validity analyses controlling for interaction quantity
- the Sample 2 human-rater CFA validation
- the Sample 2 mediation analyses
- the Sample 3 homogenization analysis
- all four main manuscript figures

Generated figures are written to `figures/`.

## Repository Structure

| Directory | Contents |
|-----------|----------|
| `data/` | Bundled processed data for Samples 1-4 plus a data dictionary in `data/README.md` |
| `scripts/` | Scripts for results, appendix analyses, and figure generation |
| `figures/` | Figure-generation scripts and generated figure files |
| `docs/` | Preregistration PDF |

## Optional Individual Commands

`python scripts/reproduce_results.py` runs the full bundled pipeline. The
individual scripts remain available if you want to rerun one component:

```bash
python scripts/cfa_human_ratings.py
python scripts/homogenization_s3.py
python scripts/analyze_sample3_gptzero.py
python figures/fig_validation.py
python figures/fig_forest_replication.py
python figures/fig_mediation.py
python figures/fig_distributions.py
```

## Supplemental Analyses

Statistics reported in the SI but not in the main pipeline:

```bash
python scripts/iccs_and_supplemental_stats.py   # per-sample ICC(2,k), S3 self-report and MAILS, S1 delegation floor
python scripts/sample1_solo_vs_chat_d.py        # Sample 1 solo-vs-chat Cohen's d
python scripts/homogenization_s1_s2.py          # Sample 1 and 2 homogenization median splits
```

## Included Documents

- `docs/preregistration.pdf`

## Notes

- The repository uses processed data files bundled in `data/`.
- All random seeds used in the active analysis code are set to `99`.
- `scripts/homogenization_s3.py` downloads the `all-MiniLM-L6-v2`
  SentenceTransformers model on first run unless it is already cached locally.

## License

Code is released under the MIT License. See `data/README.md` for dataset terms.
