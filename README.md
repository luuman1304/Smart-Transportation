# Smart Transportation: station cold-start evaluation

This repository contains the source code used in the study of station-level
cold-start evaluation for last-mile route sequencing. It compares JPT-AMZ under
two paired protocols:

1. random out-of-fold evaluation, where other historical routes from the target
   station can remain in training;
2. station-held-out evaluation, where the target station is removed before all
   JPT history tables and routing artifacts are built.

The repository also contains the station-grouped risk selector, cross-station
grammar and soft-prior controls, and the LaDe-D external evaluation. Manuscripts,
Word/PDF files, raw datasets, generated routes, and outputs from discarded
experiments are intentionally excluded.

## Repository scope

```text
src/lastmile_data/
  prepare.py                         canonical Amazon data preparation
  reproduce_jpt.py                   author-code JPT reproduction
  build_residual_oof_*.py            leakage-controlled random OOF controls
  evaluate_shift_phase_a.py          five-station pilot
  evaluate_station_shift_expansion.py  twelve-station confirmation
  audit_*.py                         split and prediction integrity checks
  train_sarp_phase_b.py              station-grouped risk stress test
  evaluate_shared_zone_grammar.py    hard cross-station transfer control
  evaluate_soft_zone_prior.py        bounded soft-prior control
  lade_external.py                   primary LaDe-D external evaluation
  lade_pairwise.py                   exploratory pairwise sensitivity model
  build_station_shift_paper.py       statistical aggregation utilities
tests/                               synthetic and unit-level correctness tests
scripts/                             public dataset download helpers
data/metadata/station_city_map.csv   station-to-metropolitan grouping metadata
```

## External source dependencies

The exact third-party implementations are pinned as Git submodules:

- `third_party/jpt-amz`: author implementation of JPT-AMZ;
- `third_party/rc-cli`: official Amazon challenge scorer.

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/luuman1304/Smart-Transportation.git
cd Smart-Transportation
```

If the repository was cloned without them:

```bash
git submodule update --init --recursive
```

## Environment

Python 3.9 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[research,dev]'
```

Build the JPT/LKH dependency according to
`third_party/jpt-amz/README`. The Python pipeline expects the compiled author
implementation under `third_party/jpt-amz` and the scorer at
`third_party/rc-cli/scoring/score.py`.

## Data

Raw data is not redistributed. Download it from the original providers:

```bash
./scripts/download_amazon.sh
./scripts/download_lade_d.sh
```

Expected local roots are ignored by Git:

```text
data/raw/amazon/
data/raw/lade/
```

Users must follow the original dataset licenses and terms of use.

## Validation

The included tests use synthetic fixtures or small deterministic examples. They
check route feasibility, official scoring integration, split integrity,
station exclusion, conformal quantiles, transfer controls, and LaDe ranking
metrics.

```bash
pytest -q
```

## Reproducibility boundaries

- Target routes are excluded from their random OOF training fold.
- The entire target station is excluded before building station-held-out JPT
  history and intermediate artifacts.
- Target sequences are opened only after predicted tours are frozen.
- Pilot stations and confirmation stations are kept separate.
- Statistical uncertainty is aggregated at station level.
- Raw datasets and generated experiment outputs are never committed.

This repository provides research code, not a production dispatch system.
