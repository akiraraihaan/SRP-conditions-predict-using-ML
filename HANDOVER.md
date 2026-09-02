# HANDOVER

State of the repository at the end of the scaffolding work, what to run in which
order, and what to carry back out of Kaggle so the next session resumes cleanly.

**Nothing has been run at scale.** `artifacts/registry.jsonl` is committed
**empty**. It previously held one CPU smoke record for `yolo26n` r0f0; that record
predated the current schema, and because script 03 skips by `run_id` it would have
suppressed the GPU run of that fold and left one CPU result on an older schema
sitting in the final 75. It was truncated deliberately — see §4.4.

---

## 1. Files

### Created

| path | what |
| --- | --- |
| `README.md` | orientation, layout, status |
| `MIGRATION_NOTES.md` | 15 sections: everything extracted from `../FINAL-pipeline`, with file and cell citations |
| `HANDOVER.md` | this file |
| `requirements.txt` | pinned to the versions found in `../FINAL-pipeline/.venv` |
| `.gitignore` | `artifacts/` ignored except the nine committed artefacts (which now include `registry.jsonl`) |
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
| `scripts/restore_arms.py` | puts `artifacts/resolved_arms.yaml` back over `configs/arms.yaml` |
| `notebooks/colab_runner.ipynb` | **the primary runner**, 19 cells, no experiment logic |
| `docs/COLAB_SETUP.md` | Drive layout, dataset upload, the `artifacts/` symlink, CPU-runtime fallback |
| `notebooks/kaggle_runner.ipynb` | thin runner, 13 cells, no experiment logic *(alternative platform)* |
| `docs/KAGGLE_SETUP.md` | dataset upload, `DATA_ROOT`, session boundary, script order *(alternative platform)* |

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
| `artifacts/registry.jsonl` | **the resume state**; committed at the end of every session, restored by the next session's clone. Currently empty |
| `artifacts/resolved_arms.yaml` | snapshot of the resolved hyperparameters, written by scripts 01 and 02. Absent until one of them runs. See §4.6 |

### Generated, not committed

`summary_cv.csv`, `summary_per_class.csv`, `selected_epochs.csv`,
`medium_grid_complete.csv`, `baseline_lr_sweep.csv`, `ablation_paired.csv`,
`ablation_per_class.csv`, `learning_curve.csv`, `edge_benchmark.json`, `figures/`.

(`registry.jsonl` used to be in this list. It is now committed, and that commit is
the primary resume path — see §4.)

### Not modified

`../FINAL-pipeline` — read only, throughout. Verified with `find -newermt`.

---

## 2. Execution order

Strict. Later steps consume what earlier ones write.

| # | command | runs | notes |
| ---: | --- | ---: | --- |
| 1 | `python scripts/00_build_folds.py` | — | **phase 0 preflight** (§4.1), then idempotent verification of the committed artefacts rather than rebuilding them |
| 2 | `python scripts/01_complete_medium_grid.py` | 8 | **writes `configs/arms.yaml`** |
| 3 | `python scripts/02_lr_sweep_baselines.py` | 6 | **writes `configs/arms.yaml`** |
| 4 | `python scripts/03_run_cv.py` | 75 | needs 2 and 3 committed first |
| 5 | `python scripts/04_run_ablation.py` | 15 | needs 4's `yolo26n` folds for the paired analysis |
| 6 | `python scripts/05_learning_curve.py` | 75 | needs 2's locked `yolo26n` config |
| 7 | `python scripts/06_export_figures.py` | — | skips figures whose inputs are absent, naming the script that makes them |
| 8 | `python scripts/07_bench_edge.py …` | — | **Raspberry Pi, not Kaggle**; CPU only, refuses to run if CUDA is visible |

Steps 1–7 run on the GPU platform (§4). Each is resumable; re-running a finished
step prints `N already complete (skipped), 0 remaining` and exits.

Step 8 runs on **none** of the GPU platforms: `07_bench_edge.py` is CPU-only by
design and refuses to run when CUDA is visible. It belongs on the Raspberry Pi.

Step 6 is 5 fractions × 15 folds = **75 runs**, one draw each, 15 estimates per
fraction. It was previously planned at 225 because `learning_curve.repeats: 3`
multiplied the 15 folds, which already *are* 3 repeats of 5-fold CV — three times
the size of the main experiment for a supporting analysis, with the extra draws
resampling the same 15 partitions. The key is gone from `configs/arms.yaml`, and
`05_learning_curve.py` raises if it reappears.

`--dry-run` on steps 2–6 prints the plan without training. Use it first.

### Scripts that write back into `configs/arms.yaml`

Only two:

- **`01_complete_medium_grid.py`** → `yolo26m`: `epochs`, `batch`, `lr`,
  `locked: true`, `provisional: false`, `lr_source`.
- **`02_lr_sweep_baselines.py`** → `mobilenetv3_small` and `resnet18`: `lr`
  (currently `null`), `locked: true`, `lr_source`.

**Commit `configs/arms.yaml` immediately after each.** The next session clones
fresh.

If you skip it, you no longer silently get the wrong experiment — §4.5 covers what
happens instead — but you do get a stop, and the fix costs a step. Commit the file.

---

## 3. What each stage produced, in one line each

- **(b)** 695 images, 10 classes after normalisation, all per-class counts assert clean.
- **conflict groups** 13 sha1 groups spanning >1 class → 27 files excluded → **668**, imbalance 4.16:1.
- **(c)** dev split recovered by **Route A** (695/695 by filename, 0 by sha1), reproduces 556/69/70 exactly. Route B matches the counts but disagrees on 225 images — the count assert alone could not have caught that.
- **cross-refs** 1 contaminated image in the legacy test set (+0.0141 on F1 if removed); 0 of the 6 "similarity" errors are contaminated, so that paragraph survives; the within-nano Friedman significance came from **val**, not test, so that paragraph does not.
- **(d)** 15 folds, corpus-fingerprinted, every image in exactly 3 test partitions, no sha1 across any fold boundary, `gas_influence` 6–7 test images per fold.
- **(e)** one model interface for five arms; class-weight verification passes; determinism verified; smoke run selected epoch 34/50, test macro-F1 0.5484, 125 s CPU.

---

## 4. Where this runs, and how results survive

### 4.0 Platform

**Free Google Colab is the primary path.** It needs a Google account and nothing
else. `notebooks/colab_runner.ipynb` is the runner; `docs/COLAB_SETUP.md` is the
setup guide.

The alternatives were evaluated and rejected, and are documented rather than
recommended:

| platform | status | blocker |
| --- | --- | --- |
| **free Colab** | **primary** | — a Google account is enough |
| Kaggle | documented alternative | GPU *and* Internet are gated behind account verification. `notebooks/kaggle_runner.ipynb` and `docs/KAGGLE_SETUP.md` are complete and current, and work as soon as an account is verified |
| Google Cloud Free Trial | rejected | the trial billing account forbids attaching GPUs to VM instances and forbids quota-increase requests; new-account GPU quota is zero. The $300 credit needs a card, an upgrade and an approved quota request first |
| RunPod | rejected | requires a $10 deposit |
| Raspberry Pi | **required, for step 8 only** | `07_bench_edge.py` refuses to run when CUDA is visible. It runs on none of the platforms above |

The two notebooks are the same shape and differ only where the platforms differ.
The one difference that matters is how results survive:

- **Colab** symlinks `artifacts/` into Drive, so `registry.jsonl` is written to
  durable storage *as each run completes*. A dropped session costs at most the run
  in flight, and the zip at the end is for git versioning only.
- **Kaggle** stages everything in the container, so the last cell must be run
  before the session ends or the session's work is lost.

Free Colab's own limits — ~90 min idle disconnect, dynamic session length, and a
dynamic GPU quota that silently drops you to a CPU runtime after heavy use — mean
the two 75-run scripts take several sessions across several days. That is expected,
and the registry resume is what makes it workable. `docs/COLAB_SETUP.md` §4 and §5
cover both.

Everything from §4.1 down is platform-independent.

### 4.1 First, run the preflight

`python scripts/00_build_folds.py` opens with a **phase 0 preflight** that never
trains and never raises — it reports, and the script exits non-zero if anything
failed. `--preflight-only` runs just that part; `--skip-checkpoint-preflight`
skips the downloads. It covers:

| section | what it tells you |
| --- | --- |
| environment | GPU name, torch / torchvision / ultralytics versions, `torch.version.cuda`, and whether `cudnn.deterministic` actually took effect |
| DATA_ROOT | the path it resolved and from where, plus the per-class image counts it actually found against `expected_counts` |
| committed artefacts | `image_index.csv`, `dev_split.json` and `folds.json` verified against their **fingerprints**, not merely present: the `folds.json` corpus block is recomputed from the committed index, and the dev split is checked disjoint, exhaustive and in range |
| pretrained checkpoints | all five arms' checkpoints **loaded**, not just resolved — so a missing YOLO26 file surfaces in minute one instead of after 40 runs, and the table says which arms downloaded and which were already local |

### 4.2 The pretrained-fallback guard

`configs/arms.yaml` gives each YOLO arm a `pretrained_fallback` naming a **YOLO11**
checkpoint. If that were taken silently, the architecture behind every table and
figure would change with nothing downstream able to detect it. So:

- The resolved checkpoint filename is asserted against the arm's declared
  architecture. A mismatch raises `ArchitectureMismatchError` naming both.
- Falling back is refused unless you pass `--allow-pretrained-fallback`
  (scripts 01–05). Even then it prints a full-width banner and flags the runs.
- Every registry record carries `checkpoint_resolved` (the filename actually
  loaded) and `pretrained_fallback_used` (bool). Notebook cell 6 counts them.

Script 01 loads its checkpoint directly rather than through `build_model`, so it
carries the same guard explicitly.

As of the preflight run on this machine, all three YOLO26 classification
checkpoints download successfully, so the fallback should never fire.

### 4.3 The record schema, and why stale records are dangerous

`registry.REQUIRED_RECORD_FIELDS` is the current schema. Presence is what is
checked, not truthiness: a field that legitimately does not apply is `None` with
the reason recorded beside it, which is different from the field being absent
because an older version of the code did not know about it.

That distinction matters because of how resumption works. A stale record still
matches by `run_id`, so the run it stands for is **skipped** — it is never re-run,
and its older numbers are inherited into the final results. So:

- `append_record` **refuses** to write a record missing any required field. That is
  this code's own bug, caught at the moment it would be planted.
- `warn_if_stale` prints a loud block naming every offending line, its `run_id` and
  its missing fields. It runs from `print_plan`, once per script invocation, right
  where the skip count is printed.
- `summarise` reports `n_incomplete_schema` and `n_class_weights_unverified`.

Three groups of field were added because they had to travel *with* the results
rather than live only in a console line or a smoke test:

| field | why |
| --- | --- |
| `checkpoint_resolved`, `pretrained_fallback_used` | which pretrained weights actually loaded (§4.2) |
| `class_weights_verified`, `class_weights_proof` | the measured proof that the balanced weights reach the loss (§4.4) |
| `corpus_fingerprint` | the corpus the run was produced on — verified at load time by every consumer of `folds.json`, now recorded too |
| `selected_epoch`, `epochs_run`, `stopped_early`, `best_val_f1`, `best_val_loss`, `min_val_loss_epoch`, `history` | the uniform-protocol training outcome, promoted from `extra` to top level so there is one source of truth |

`corpus_fingerprint` carries a `kind`: `cv_clean_668` for scripts 03–05 (taken from
the verified `folds.json` corpus block) and `dev_raw_695` for scripts 01 and 02,
which run on the raw index with its original labels and never touch the clean
corpus. Same key names, so the two can be diffed directly.

Script 01 is the one script whose records carry `class_weights_verified: None` and
a `None` training outcome — it reproduces the legacy unweighted protocol through
`ultralytics model.train()`, so there are no class weights to verify and no
uniform-loop history to record. Both carry an explicit reason string.

### 4.4 The class-weight proof travels with the results

`train.verify_class_weights_applied()` existed but nothing called it, so the proof
lived only in a smoke test. It is now called by `require_class_weights_verified()`
**once per invocation of scripts 02, 03, 04 and 05, before the run loop**, and:

- the script **aborts** if it fails — no run is written under weights that do not
  reach the loss;
- the boolean and the measured weighted/unweighted cross-entropy pair go into
  **every record** that invocation writes.

Script 04 runs it too even though the ablation trains *unweighted*: the ablation
only means anything if the weighting it removes demonstrably works.

This is guarding against a failure that already happened once. The legacy pipeline
computed the balanced weights, printed them, charted them — and never passed them
to the trainer (MIGRATION_NOTES.md §5.4). Nothing in its saved output would have
revealed it.

### 4.5 Hyperparameter drift, and how it is prevented

This is the most dangerous failure mode left in the pipeline, because it produces
plausible numbers rather than an error.

`epochs`, `batch` and `lr` are in `registry.RUN_ID_FIELDS`. Change one and the same
fold of the same arm hashes to a **different** `run_id`. Resumption is by `run_id`,
so nothing is skipped — the fold is simply trained again, under the new settings,
and the registry ends up holding **two hyperparameter regimes for one arm**. A mean
across folds then averages both and reports a number describing no configuration
that was ever run.

The realistic route in: scripts 01 and 02 rewrite `configs/arms.yaml`; the session
ends before that file is committed; the next clone carries the **provisional**
`yolo26m` config again. Nothing about that is visibly wrong.

Four checks now close it, at every point where it could do damage:

| where | what it does |
| --- | --- |
| `00_build_folds.py` phase 0 | reports each arm's `epochs`/`batch`/`lr` and whether it is `locked`, `lr: null` or `provisional`, and whether `artifacts/resolved_arms.yaml` exists and **differs** from `configs/arms.yaml`. A difference is the signature of a lost config, and the preflight exits non-zero |
| scripts 03, 04, 05 | before the run loop, `registry.assert_arms_match_registry` compares the loaded config against the `epochs`/`batch`/`lr` recorded in completed runs of the same arm and `split_kind`. On a mismatch it **aborts**, printing both configurations side by side and every affected `run_id`. Never proceeds, never treats it as a new run |
| `aggregate.py` | `assert_hyperparameters_unanimous` refuses to build `summary_cv.csv` or `summary_per_class.csv` when an arm's records are not unanimous, naming the offending `run_ids` |
| scripts 01, 02 | snapshot the resolved config to `artifacts/resolved_arms.yaml` so it can be recovered — §4.6 |

Scripts 01 and 02 are deliberately **not** guarded this way: they *sweep*
hyperparameters, so varying `epochs`/`batch`/`lr` across their records is the point.
The guard applies to the scripts that write cv-kind records.

### 4.6 Recovering a resolved config

Scripts 01 and 02 write `artifacts/resolved_arms.yaml` immediately after they rewrite
`configs/arms.yaml`: a header naming the resolving script, the timestamp and the git
commit, then a verbatim copy of the file.

It lives in `artifacts/`, which on Colab is a symlink into Drive — so it survives a
dropped session even if `configs/arms.yaml` was never downloaded or committed. It is
also committed (it has a `.gitignore` exception), so the same recovery works from a
bare clone on any platform.

In a fresh clone whose `configs/arms.yaml` has reverted:

```bash
python scripts/restore_arms.py --check   # show the difference, change nothing
python scripts/restore_arms.py           # restore, then COMMIT configs/arms.yaml
```

Both print a per-arm before/after of `epochs`/`batch`/`lr`. `--check` exits 1 when a
difference exists, so it is usable as a guard in a shell script.

`restore_arms.py` only ever writes `configs/arms.yaml`. It does not touch the
registry: if runs were already completed under the wrong config, restoring the file
is the first half of the fix and deleting those `run_ids` is the second. The abort
message from §4.5 names them.

### 4.7 Getting the results out

**On Colab**, Drive already holds everything the moment it is written, so this is
about git history rather than survival: run the zip cell when you want to commit,
not before the session ends. Missing it costs a commit, not a session's work.

**On Kaggle**, the zip cell *is* how the work survives, and it must be run before the
session ends even if the script was interrupted.

Either way, in order:

1. Run the zip cell. It copies `artifacts/` plus `configs/arms.yaml` into a bundle.
2. Download `artifacts_bundle.zip` and unpack it over your local checkout.
3. Commit these:

| file | when | why |
| --- | --- | --- |
| `artifacts/registry.jsonl` | **every session** | the only record of completed runs; without it the next session re-runs everything. This commit **is** the resume mechanism |
| `configs/arms.yaml` | after steps 2 and 3 | the locked hyperparameters |
| `artifacts/medium_grid_complete.csv` | after step 2 | the 18-config table |
| `artifacts/baseline_lr_sweep.csv` | after step 3 | the 6-run sweep |
| `artifacts/ablation_paired.csv`, `ablation_per_class.csv` | after step 5 | manuscript tables |
| `artifacts/learning_curve.csv` | after step 6 | manuscript table |
| `artifacts/summary_cv.csv`, `summary_per_class.csv`, `selected_epochs.csv` | after step 7 | manuscript tables |
| `artifacts/figures/*` | after step 7 | the figures |
| `artifacts/edge_benchmark.json` | after step 8 | Pi results |

4. Next session, resumption is automatic, by a different route on each platform:

   - **Colab:** Drive holds the live `artifacts/`. Cell 5 re-establishes the
     symlink and prints how many completed runs the registry already has. Nothing
     needs to have been committed for this to work.
   - **Kaggle:** git is the resume path. `artifacts/registry.jsonl` has a
     `.gitignore` exception and is committed, so the fresh `git clone` in cell 1
     brings it back. `RESUME_FROM` (cell 3) is the documented fallback for when a
     session's results could not be committed; it overwrites `artifacts/` from a
     private Kaggle Dataset, so a stale one hides newer committed results.

   Committing `registry.jsonl` is still worth doing on both: it is the versioned
   record of what has been run, and on Kaggle it is the only one.

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

python scripts/00_build_folds.py --preflight-only   # environment + artefacts + checkpoints
python scripts/00_build_folds.py
python scripts/03_run_cv.py --arms yolo26n --repeat 0 --fold 0   # smoke test
```

Every script takes `--dry-run` (except 00 and 07) and fails with a message naming
the missing file when an input is absent. Scripts 01–05 also take
`--allow-pretrained-fallback`; see §4.2 before you use it.

Nothing in `src/srpcard/` or `configs/` knows which platform it is on. Both
notebooks are adapters: they set `SRPCARD_DATA_ROOT`, arrange for `artifacts/` to
be durable, and call the same scripts.

Note: the one pre-existing registry record (the CPU smoke run) predates the
`checkpoint_resolved` / `pretrained_fallback_used` fields and does not carry them.
Every record written from now on does.
