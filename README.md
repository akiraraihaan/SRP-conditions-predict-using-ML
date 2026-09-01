# extra-deep — journal-revision experiment

Single-label image classification of downhole dynamometer cards into 10 sucker rod
pump operating conditions. 695 images, 10 classes.

This repository is the **revision** of the study in `../FINAL-pipeline`. It replaces
that work's single 80:10:10 split with repeated stratified k-fold on GPU, adds two
baseline architectures, and adds a class-weight ablation. Hyperparameters are **not**
re-searched — the grid-search winners are locked and carried over.

`../FINAL-pipeline` is read-only reference material. Nothing here writes to it.
What was taken from it, and from which file and cell, is recorded in
[MIGRATION_NOTES.md](MIGRATION_NOTES.md).

---

## What changes relative to the old study

| | old | this repository |
| --- | --- | --- |
| Reporting protocol | one stratified 80:10:10 split | RepeatedStratifiedKFold, 5 splits × 3 repeats = 15 folds |
| Role of the 80:10:10 split | selection **and** reporting | selection only (the *development split*) |
| Architectures | yolo26 n / s / m | + `mobilenetv3_small`, + `resnet18` |
| Class weights | computed, **never applied** to the loss | applied, and ablated |
| Compute | CPU | GPU (Kaggle) for training; CPU (Raspberry Pi) for the edge benchmark |
| Hyperparameter search | 46 configurations | none re-run; 8 missing medium configs completed, 6 baseline lr runs |
| Statistics | Friedman over 10 classes, single split | paired across 15 folds |

---

## Layout

```
configs/          data.yaml, folds.yaml, arms.yaml — every path and constant
src/srpcard/      library code, no side effects on import
scripts/          00..07, each runs end to end from one command
notebooks/        kaggle_runner.ipynb — thin; no experiment logic
docs/             KAGGLE_SETUP.md
artifacts/        generated. gitignored EXCEPT folds.json and image_index.csv,
                  which are committed and treated as frozen inputs.
```

## Scripts, in execution order

| script | what it does | runs |
| --- | --- | ---: |
| `00_build_folds.py` | image index, class normalisation, conflict-group exclusion, count asserts, dev split + distribution assert, legacy cross-references, folds.json | — |
| `01_complete_medium_grid.py` | the 8 medium configurations missing from the old grid, on the dev split; recomputes the medium winner; writes it back to `arms.yaml` | 8 |
| `02_lr_sweep_baselines.py` | lr ∈ {1e-4, 1e-3, 1e-2} for the two baselines on the dev split; writes winners back to `arms.yaml` | 6 |
| `03_run_cv.py` | the main experiment: 5 arms × 15 folds | 75 |
| `04_run_ablation.py` | `class_weights: none` on `yolo26n`, same folds/seeds/epochs/batch/lr | 15 |
| `05_learning_curve.py` | 20/40/60/80/100 % of each fold's training set, 3 repeats, under the **final locked** nano config | — |
| `06_export_figures.py` | publication figures, vector PDF + high-resolution PNG | — |
| `07_bench_edge.py` | standalone Raspberry Pi latency benchmark. No training, no CUDA. | — |

Everything except `07` runs on Kaggle. `07` runs on the Pi.

## Resumability

`src/srpcard/registry.py` appends one JSONL record per completed run to
`artifacts/registry.jsonl`. `run_id` is a deterministic hash of the parameters that
define the run, so every script checks the registry before starting and prints how
many runs are complete, skipped and remaining. A session killed at run 40 loses
nothing — rerun the same command.

## Reproducibility

One `set_seed(seed)` covers `random`, `numpy`, `torch`, `torch.cuda` and
`PYTHONHASHSEED`, and is called at the start of every run; the resolved seed is logged
into the registry alongside the installed library versions. All paths come from
`configs/data.yaml` — nothing is hardcoded to `/kaggle` or to a local directory.

## Data

Raw, labelled only by directory:

```
<DATA_ROOT>/<class_dir>/*.png|*.jpg
```

No pre-existing train/val/test structure. Class directory names are normalised by
stripping a leading `^\d+_` before use, because the source data mixes
`10_severe_vibration` with `severe_vibration`; after normalisation there must be
exactly 10 classes, and loading fails loudly with the offending names otherwise. In the
old pipeline that collision silently deflated every metric —
[MIGRATION_NOTES.md §3](MIGRATION_NOTES.md) has the evidence.

See [docs/KAGGLE_SETUP.md](docs/KAGGLE_SETUP.md) for uploading the images as a private
Kaggle Dataset and wiring `DATA_ROOT` to it.

## Status

Built in order, verified at each step:

- [x] (a) repo skeleton + MIGRATION_NOTES.md
- [x] (b) data loading, class normalisation, image_index.csv, count asserts
- [x] (c) legacy_split recovery (Route A) + distribution assert + legacy cross-references
- [x] (d) folds.json (frozen, corpus-fingerprinted)
- [x] (e) registry + models + train + evaluate, smoke test of yolo26n on one fold
- [x] (f) remaining scripts, aggregate, Kaggle runner, docs

Nothing has been run at scale yet: the registry holds a single smoke run. See
HANDOVER.md for the execution order.
