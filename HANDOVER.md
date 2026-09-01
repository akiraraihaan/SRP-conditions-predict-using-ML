# HANDOVER

State of the repository at the end of the scaffolding work, what to run in which
order, and what to carry back out of Kaggle so the next session resumes cleanly.

**Nothing has been run at scale.** `artifacts/registry.jsonl` holds exactly one
record: the `yolo26n` repeat-0 fold-0 smoke run, on CPU, under the final
criterion. Every other number in this repository is still to be produced.

---

## 1. Files

### Created

| path | what |
| --- | --- |
| `README.md` | orientation, layout, status |
| `MIGRATION_NOTES.md` | 15 sections: everything extracted from `../FINAL-pipeline`, with file and cell citations |
| `HANDOVER.md` | this file |
| `requirements.txt` | pinned to the versions found in `../FINAL-pipeline/.venv` |
| `.gitignore` | `artifacts/` ignored except the six committed artefacts |
| `configs/data.yaml` | paths, classes, raw counts, `clean_corpus`, `legacy_reference` |
| `configs/folds.yaml` | CV protocol and the seed scheme |
| `configs/arms.yaml` | five arms, `uniform_protocol`, search spaces, ablation, learning curve |
| `src/srpcard/config.py` | paths, `set_seed`, provenance *(addition to the brief's module list)* |
| `src/srpcard/data.py` | normalisation, index, conflict groups, asserts, letterbox |
| `src/srpcard/legacy_split.py` | dev-split recovery, routes A/B/C, distribution assert |
| `src/srpcard/legacy_audit.py` | the three legacy cross-references *(addition)* |
| `src/srpcard/folds.py` | fold construction, corpus fingerprint, verification, report |
| `src/srpcard/models.py` | one interface for all five arms |
| `src/srpcard/train.py` | the uniform training loop, class weights, class-weight verification |
| `src/srpcard/evaluate.py` | metrics, always 10x10 canonical |
| `src/srpcard/efficiency.py` | params, GFLOPs, size, light latency |
| `src/srpcard/registry.py` | append-only JSONL, `run_id`, resume planning |
| `src/srpcard/aggregate.py` | `summary_cv.csv` and the other manuscript tables |
| `src/srpcard/figures.py` | matplotlib-only figures, PDF + PNG |
| `scripts/00_build_folds.py` … `scripts/07_bench_edge.py` | the eight scripts |
| `notebooks/kaggle_runner.ipynb` | thin runner, 13 cells, no experiment logic |
| `docs/KAGGLE_SETUP.md` | dataset upload, `DATA_ROOT`, session boundary, script order |

### Generated, and committed

| artefact | why it is committed |
| --- | --- |
| `artifacts/image_index.csv` | 695 rows; every other artefact references images by its `idx` |
| `artifacts/folds.json` | **frozen input**; 15 folds + val slices + corpus fingerprint |
| `artifacts/dev_split.json` | recovered by Route A, which cannot run on Kaggle |
| `artifacts/excluded_images.csv` | the 27 excluded files, with size and dimensions |
| `artifacts/folds_report.md` | fold-level content report, for review |
| `artifacts/legacy_contamination.json` | the three cross-reference results |
| `artifacts/legacy_grid_metrics.csv` | the 46 legacy configs; **script 01 needs this on Kaggle** |
| `artifacts/legacy_test_predictions.csv` | per-image predictions behind cross-reference 2 |

### Generated, not committed

`registry.jsonl`, `summary_cv.csv`, `summary_per_class.csv`, `selected_epochs.csv`,
`medium_grid_complete.csv`, `baseline_lr_sweep.csv`, `ablation_paired.csv`,
`ablation_per_class.csv`, `learning_curve.csv`, `edge_benchmark.json`, `figures/`.

**Exception:** `registry.jsonl` *must* be committed between Kaggle sessions — see §4.

### Not modified

`../FINAL-pipeline` — read only, throughout. Verified with `find -newermt`.

---

## 2. Execution order

Strict. Later steps consume what earlier ones write.

| # | command | runs | notes |
| ---: | --- | ---: | --- |
| 1 | `python scripts/00_build_folds.py` | — | idempotent; verifies the committed artefacts rather than rebuilding them |
| 2 | `python scripts/01_complete_medium_grid.py` | 8 | **writes `configs/arms.yaml`** |
| 3 | `python scripts/02_lr_sweep_baselines.py` | 6 | **writes `configs/arms.yaml`** |
| 4 | `python scripts/03_run_cv.py` | 75 | needs 2 and 3 committed first |
| 5 | `python scripts/04_run_ablation.py` | 15 | needs 4's `yolo26n` folds for the paired analysis |
| 6 | `python scripts/05_learning_curve.py` | 225 | needs 2's locked `yolo26n` config |
| 7 | `python scripts/06_export_figures.py` | — | skips figures whose inputs are absent, naming the script that makes them |
| 8 | `python scripts/07_bench_edge.py …` | — | **Raspberry Pi, not Kaggle**; CPU only, refuses to run if CUDA is visible |

Steps 1–7 run on Kaggle. Each is resumable; re-running a finished step prints
`N already complete (skipped), 0 remaining` and exits.

`--dry-run` on steps 2–6 prints the plan without training. Use it first.

### Scripts that write back into `configs/arms.yaml`

Only two:

- **`01_complete_medium_grid.py`** → `yolo26m`: `epochs`, `batch`, `lr`,
  `locked: true`, `provisional: false`, `lr_source`.
- **`02_lr_sweep_baselines.py`** → `mobilenetv3_small` and `resnet18`: `lr`
  (currently `null`), `locked: true`, `lr_source`.

**Commit `configs/arms.yaml` immediately after each.** The next Kaggle session
clones fresh. If you skip this, `03_run_cv.py` trains the provisional `yolo26m`
configuration and refuses the two baselines outright (`lr: null` → an explicit
skip message, not a crash).

---

## 3. What each stage produced, in one line each

- **(b)** 695 images, 10 classes after normalisation, all per-class counts assert clean.
- **conflict groups** 13 sha1 groups spanning >1 class → 27 files excluded → **668**, imbalance 4.16:1.
- **(c)** dev split recovered by **Route A** (695/695 by filename, 0 by sha1), reproduces 556/69/70 exactly. Route B matches the counts but disagrees on 225 images — the count assert alone could not have caught that.
- **cross-refs** 1 contaminated image in the legacy test set (+0.0141 on F1 if removed); 0 of the 6 "similarity" errors are contaminated, so that paragraph survives; the within-nano Friedman significance came from **val**, not test, so that paragraph does not.
- **(d)** 15 folds, corpus-fingerprinted, every image in exactly 3 test partitions, no sha1 across any fold boundary, `gas_influence` 6–7 test images per fold.
- **(e)** one model interface for five arms; class-weight verification passes; determinism verified; smoke run selected epoch 34/50, test macro-F1 0.5484, 125 s CPU.

---

## 4. Carrying results out of Kaggle

Kaggle sessions end after ~12 h. The registry is append-only and fsync'd after
every completed run, so an interrupted session loses nothing.

**After every session, in order:**

1. Run the last notebook cell (§5 of the notebook) even if the run was
   interrupted. It copies `artifacts/` to `/kaggle/working/artifacts_out/` and
   zips it to `artifacts_bundle.zip`.
2. Download `artifacts_bundle.zip` from the output panel.
3. Commit these:

| file | when | why |
| --- | --- | --- |
| `artifacts/registry.jsonl` | **every session** | the only record of completed runs; without it the next session re-runs everything |
| `configs/arms.yaml` | after steps 2 and 3 | the locked hyperparameters |
| `artifacts/medium_grid_complete.csv` | after step 2 | the 18-config table |
| `artifacts/baseline_lr_sweep.csv` | after step 3 | the 6-run sweep |
| `artifacts/ablation_paired.csv`, `ablation_per_class.csv` | after step 5 | manuscript tables |
| `artifacts/learning_curve.csv` | after step 6 | manuscript table |
| `artifacts/summary_cv.csv`, `summary_per_class.csv`, `selected_epochs.csv` | after step 7 | manuscript tables |
| `artifacts/figures/*` | after step 7 | the figures |
| `artifacts/edge_benchmark.json` | after step 8 | Pi results |

4. Next session: the fresh clone brings `registry.jsonl` with it, so resumption is
   automatic. Alternatively upload `artifacts/` as a small private dataset and set
   `RESUME_FROM` in notebook cell 3.

**Never commit** model weights or `runs/` — both are gitignored and large.

---

## 5. Things to look at before writing the methods section

1. **`selected_epoch` distribution.** `artifacts/selected_epochs.csv` and
   `fig_selected_epochs`. If it clusters near the epoch budget, the locked epoch
   counts are too short and the budget needs revisiting. The smoke run picked
   34/50 — comfortable, but that is one fold of one arm.
2. **The two contradicted manuscript claims** in MIGRATION_NOTES §5.4 (class
   weights were never applied in the old work) and §13.3 (the within-nano Friedman
   significance came from the validation partition).
3. **`gas_influence` has 2 validation images per fold** (`folds_report.md`). It
   does not affect reported metrics, but it adds variance to checkpoint selection.
4. **Script 01 is the only script that deliberately reproduces the legacy bug.**
   Everything it produces belongs to the legacy unweighted protocol and must be
   labelled as such wherever it is reported. It is not comparable with anything
   from scripts 02–05.

---

## 6. Local development

```bash
export SRPCARD_DATA_ROOT=/path/to/dataset      # the 10 class directories
export SRPCARD_WEIGHTS_DIR=/path/to/checkpoints # optional; else ultralytics downloads
export SRPCARD_LEGACY_DIR=../FINAL-pipeline     # optional; only for phase 2b

python scripts/00_build_folds.py
python scripts/03_run_cv.py --arms yolo26n --repeat 0 --fold 0   # smoke test
```

Every script takes `--dry-run` (except 00 and 07) and fails with a message naming
the missing file when an input is absent.
