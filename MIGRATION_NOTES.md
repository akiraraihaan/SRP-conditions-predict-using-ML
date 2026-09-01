# MIGRATION NOTES

What was extracted from the old project, and exactly where each value came from.

**OLD_DIR** = `d:\skripshi\FINAL-pipeline` (read-only reference; nothing in it was
modified, moved or deleted)
**NEW_DIR** = `d:\skripshi\extra-deep` (this repository)

> The task brief wrote the paths as `FINAL_pipeline` / `extra_deep`. The directories
> on disk are `FINAL-pipeline` / `extra-deep` (hyphens). Those are the ones used.

Every value below is quoted from a file in OLD_DIR. Nothing here is inferred or
invented. Where a value could **not** be recovered, it is listed in
[§8 Open items](#8-open-items-nothing-invented) rather than guessed.

---

## 0. Source inventory of OLD_DIR

| Path | Role |
| --- | --- |
| `code.ipynb` | The entire legacy pipeline. 25 cells. Sole source of the protocol. |
| `dataset/` | Raw images, 10 class directories, 695 files. Read-only input. |
| `yolo_dataset/{train,val,test}/` | Materialised legacy 80:10:10 split (letterboxed copies). Ground truth for the split reconstruction. |
| `results/training_registry.json` | 46 grid-search records. **Contains the deflated metrics — see §4.** |
| `results/metrics_canonical.csv` | 46 rows, re-evaluated correctly by `evaluate_clean`. **Authoritative metric source.** |
| `results/metrics_canonical_f1perclass.json` | Per-class F1, val + test, per config. |
| `results/pareto_analysis_best_configs.csv` | Best config per variant + Pareto flag. |
| `results/friedman_summary.json` | Friedman test over the three best configs. |
| `scripts/export_publication_figures.py` | Figure export (Figures 3–7). |
| `yolo26{n,s,m}-cls.pt` | Pretrained checkpoints used as training starting points. |
| `README.MD` | Prose summary of the completed study. |

Notebook cells are cited below as `code.ipynb cell N ["[Cell L]"]`, where `N` is the
0-based index in the `.ipynb` JSON and `L` is the human label written in the cell's
first comment line. The two disagree after index 17 because the author inserted
`19b` and `19c`, so both are given.

| JSON idx | Label | JSON idx | Label |
| --- | --- | --- | --- |
| 3 | `[Cell 4]` | 16 | `[Cell 17]` |
| 4 | `[Cell 5]` | 17 | `[Cell 18]` |
| 5 | `[Cell 6]` | 18 | `[Cell 19]` |
| 7 | `[Cell 8]` | 19 | `[Cell 19b]` |
| 8 | `[Cell 9]` | 20 | `[Cell 19c]` |
| 9 | `[Cell 10]` | 21 | `[Cell 20]` |
| 10–15 | `[Cell 11]`–`[Cell 16]` | 22–24 | `[Cell 21]`–`[Cell 23]` |

---

## 1. Global constants

All from `code.ipynb` **cell 3** `["[Cell 4]"]`.

| Constant | Value | Carried into NEW_DIR as |
| --- | --- | --- |
| `SEED` | `42` | `configs/folds.yaml: cv_seed`, and the base seed for `set_seed()` |
| `IMG_SIZE` | `224` | `configs/arms.yaml: image_size` |
| `SPLIT_RATIO` | `{"train": 0.8, "val": 0.1, "test": 0.1}` | `src/srpcard/legacy_split.py` |
| `TRAIN_OPTIMIZER` | `"MuSGD"` | `configs/arms.yaml: optimizer` (YOLO arms) |
| `PATIENCE` | `20` | `configs/arms.yaml: patience` |
| `INFERENCE_REPEAT` | `100` | superseded — see §6 |
| `STRICT_SPLIT_CLASS` | `True` | asserts are now unconditional |
| `EPOCHS_OPTIONS` | `[25, 50]` | grid axis, script `01` |
| `BATCH_SIZE_OPTIONS` | `[8, 16, 32]` | grid axis, script `01` |
| `LR_OPTIONS` | `[1e-4, 1e-3, 1e-2]` | grid axis, scripts `01` and `02` |
| `MODEL_VARIANTS` | `{"n": ["yolo26n-cls.pt", "yolo11n-cls.pt"], "s": [...s...], "m": [...m...]}` | `src/srpcard/models.py`; the `yolo11*` entries are a fallback if the YOLO26 checkpoint is absent |

Grid definition, verbatim from cell 3:

```python
HYPERPARAMETER_GRID = [
    (e, b, lr) for e in EPOCHS_OPTIONS
    for b in BATCH_SIZE_OPTIONS
    for lr in LR_OPTIONS
]   # 2 x 3 x 3 = 18 combinations per variant
```

The image-extension filter (cell 3, `list_image_files`) is
`{".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}` and is carried over
unchanged into `src/srpcard/data.py`.

`set_seed` in cell 3 covered `random`, `numpy`, `torch`, `torch.cuda`. It did **not**
set `PYTHONHASHSEED`; the new helper does (brief §7).

---

## 2. Preprocessing — letterbox

`letterbox_resize`, `code.ipynb` **cell 5** `["[Cell 6]"]`. Reproduced exactly in
`src/srpcard/data.py`:

1. `max_side = max(w, h)`
2. new `RGB` canvas `max_side x max_side` filled `(0, 0, 0)`
3. paste original at `((max_side - w) // 2, (max_side - h) // 2)` — centred
4. `resize((224, 224), Image.BILINEAR)`

Source images are `convert("RGB")`-ed before letterboxing (cell 5,
`copy_letterbox_to_split`). The cited justification in the old docstring is
Shahnaz & Mollah (2023), <https://link.springer.com/chapter/10.1007/978-981-19-0105-8_6>.

---

## 3. Class names — the normalisation bug, evidenced

### 3.1 What the directories were called

`OLD_DIR/dataset/` **today** holds ten directories with clean names. But
`OLD_DIR/yolo_dataset/`, which was materialised by cell 5 at the time the grid search
ran, preserves the names as they were **then** — four of them carried a numeric prefix:

```
10_severe_vibration          12_full_load_production
29_natural_flowing           30_pump_leakage
```

alongside six unprefixed ones. So the split, the training, and the original
evaluation all ran against prefixed labels.

### 3.2 How that deflated the metrics

`evaluate_blind` (cell 8 `["[Cell 9]"]`) compared the model's predicted class *name*
against the ground-truth directory *name* by string equality:

```python
eval_labels = sorted(set(y_true_names) | set(y_pred_names))
cm = confusion_matrix(y_true_names, y_pred_names, labels=eval_labels)
```

When the two name sets disagreed, the union grew past 10 and every mismatched pair
scored zero. Direct evidence, `results/training_registry.json`, record
`m_ep25_bs16_lr1e-04`, field `classification_report` — **13** class keys, of which
three have `support: 0` (they exist only as predictions):

| key | support |
| --- | --- |
| `10_severe_vibration` | 14 |
| `12_full_load_production` | 5 |
| `29_natural_flowing` | 10 |
| `30_pump_leakage` | 7 |
| `natural_flowing` | **0** |
| `pump_leakage` | **0** |
| `severe_vibration` | **0** |
| (6 others, unprefixed, correct) | 3–11 |

Macro-F1 is averaged over all 13 labels, so the three empty ones drag it down and the
four prefixed ones can never be matched. Registry `f1_macro` for this record is
`0.0240`; the corrected value for the same weights in `metrics_canonical.csv` is
`f1_macro_test = 0.1326`.

### 3.3 How the old code eventually fixed it

`evaluate_clean`, `code.ipynb` **cell 20** `["[Cell 19c]"]`, strips `^\d+_`
symmetrically from **both** the model's names and the folder names:

```python
def _strip(name: str) -> str:
    return _re.sub(r'^\d+_', '', name)
```

and cross-checks `accuracy_score == cm.diagonal().sum() / cm.sum()`. Cell 20 states in
its own comments: *"JANGAN baca field metrik lama dari training_registry.json"*
("do not read the old metric fields from training_registry.json").

### 3.4 What NEW_DIR does

Normalisation moves to **load time**, not evaluation time: `src/srpcard/data.py`
applies `re.sub(r'^\d+_', '', dirname)` to every class directory, then asserts
exactly 10 distinct classes and fails with the offending names otherwise. The 10
canonical names and the expected per-class counts in the brief are confirmed against
`OLD_DIR/dataset/` (counted on disk, 2026-09-01):

| class | count |
| --- | --- |
| collide_pump_and_vibration | 35 |
| full_load_production | 52 |
| gas_influence | 33 |
| gas_influence_and_vibration | 70 |
| insufficient_liquid_supply_and_vibration | 106 |
| natural_flowing | 93 |
| pump_leakage | 70 |
| severe_insufficient_liquid_supply | 52 |
| severe_vibration | 133 |
| vibration | 51 |
| **total** | **695** |

**Consequence for the manuscript:** every metric in
`results/training_registry.json` is deflated and must not be quoted. Only
`results/metrics_canonical.csv` is quotable from the old work.

---

## 4. The legacy 80:10:10 development split

### 4.1 The exact calls

`split_stratified`, `code.ipynb` **cell 5** `["[Cell 6]"]`, verbatim:

```python
train_df, temp_df = train_test_split(
    df,
    test_size=(SPLIT_RATIO["val"] + SPLIT_RATIO["test"]),   # 0.2
    stratify=df["class_name"],
    random_state=seed,                                      # 42
)
val_ratio_adj = SPLIT_RATIO["val"] / (SPLIT_RATIO["val"] + SPLIT_RATIO["test"])  # 0.5
val_df, test_df = train_test_split(
    temp_df,
    test_size=(1.0 - val_ratio_adj),                        # 0.5
    stratify=temp_df["class_name"],
    random_state=seed,                                      # 42
)
```

Recovered facts, all of which `src/srpcard/legacy_split.py` must honour:

- **Two calls, in this order.** Call 1 splits `train` vs `temp`; call 2 splits `temp`
  into `val` (first return value) and `test` (second return value).
- `random_state = 42` on **both** calls (`SEED` from cell 3, passed as
  `split_stratified(index_df_filtered, seed=SEED)` at the bottom of cell 5).
- `test_size = 0.2` then `test_size = 0.5`. Note call 2 uses `test_size`, not
  `train_size`, so `val_df` is the *first* returned frame.
- `stratify` is the class-name column, and for call 2 it is `temp_df["class_name"]`,
  i.e. re-derived from the intermediate frame, not the full frame.
- No `shuffle=` argument, so sklearn's default `shuffle=True` applies.
- The input frame is `index_df_filtered = index_df` — **unfiltered**; despite the cell
  title, no class filtering occurs.
- Row order of the input frame comes from `build_image_index` (cell 4 `["[Cell 5]"]`):
  class directories `sorted()`, and within each, `folder.rglob("*")` order —
  **not** explicitly sorted.

### 4.2 The order-dependence question, and why it does not bite

`rglob` order is filesystem-dependent, so the old code cannot guarantee *which*
image lands in which split. It can, however, guarantee the *per-class counts*:
`train_test_split(stratify=...)` delegates to `StratifiedShuffleSplit`, whose
per-class allocation is fixed by the class counts and `random_state` alone.

The brief requires a stable index (sort by filename within class, classes in
canonical order), which differs from `rglob` order. Two further order questions
were therefore checked by direct simulation rather than assumed:

1. **Does the within-class file order change the count table?** No — it cannot;
   allocation is per class.
2. **Does using normalised names instead of prefixed names change the count table?**
   It could in principle, because `np.unique` sorts the labels and
   `_approximate_mode` distributes rounding remainders in that sorted order —
   and the two name sets sort differently
   (`10_severe_vibration` first vs `collide_pump_and_vibration` first).

Question 2 was resolved empirically with `sklearn 1.8.0` (the version in
`OLD_DIR/.venv`), running both label schemes through the two calls above:

```
== PREFIXED    MATCH   totals (556, 69, 70)
== NORMALIZED  MATCH   totals (556, 69, 70)
```

Both reproduce the target table exactly. `legacy_split.py` therefore uses the
**canonical normalised** names, and this equivalence is recorded here so the choice
is not silently load-bearing.

> **Superseded in part by §13.4.** The replay described above is Route B. It
> reproduces the count table but places 225 of 695 images in different partitions
> from the split the old run actually used. The recovery that ships is Route A,
> which reads the materialised directories. Read §13.4 before relying on §4.2.

### 4.3 Independent ground truth

The reconstruction is checked twice: against the table in the brief, and against the
directory counts physically present in `OLD_DIR/yolo_dataset/`, which were written by
the original run. They agree.

| class (normalised) | train | val | test |
| --- | ---: | ---: | ---: |
| collide_pump_and_vibration | 28 | 4 | 3 |
| full_load_production | 42 | 5 | 5 |
| gas_influence | 26 | 4 | 3 |
| gas_influence_and_vibration | 56 | 7 | 7 |
| insufficient_liquid_supply_and_vibration | 85 | 10 | 11 |
| natural_flowing | 74 | 9 | 10 |
| pump_leakage | 56 | 7 | 7 |
| severe_insufficient_liquid_supply | 42 | 5 | 5 |
| severe_vibration | 106 | 13 | 14 |
| vibration | 41 | 5 | 5 |
| **TOTAL** | **556** | **69** | **70** |

Role in NEW_DIR: **development split only** — hyperparameter selection in scripts
`01` and `02`. It is never a reporting split.

---

## 5. Locked hyperparameters

### 5.1 Where the winners come from

Selection rule, `code.ipynb` **cell 20** `["[Cell 19c]"]`:

```python
_br = _sub.loc[_sub["f1_macro_val"].idxmax()]
```

i.e. **argmax of validation macro-F1**, per variant, computed by `evaluate_clean`.
Values read from `OLD_DIR/results/metrics_canonical.csv`; independently confirmed by
`OLD_DIR/results/pareto_analysis_best_configs.csv`.

| variant | winning key | epochs | batch | lr | `f1_macro_val` | `f1_macro_test` | Pareto-optimal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| n | `n_ep50_bs16_lr1e-02` | 50 | 16 | 1e-2 | 0.8364 | 0.6979 | **True** |
| s | `s_ep50_bs8_lr1e-02` | 50 | 8 | 1e-2 | 0.7257 | 0.5951 | False |
| m | `m_ep25_bs8_lr1e-02` | 25 | 8 | 1e-2 | 0.7657 | 0.7175 | False |

These are exactly the three locked arms in the brief, so the lock is confirmed
against the old code rather than taken on trust.

### 5.2 Why `yolo26m` is provisional

`training_registry.json` holds **46** records, not 54. Counting by variant:

- `n`: 18 of 18 — complete
- `s`: 18 of 18 — complete
- `m`: **10 of 18** — incomplete

The eight missing medium configurations, enumerated against the full
2 x 3 x 3 grid, are all at 50 epochs (only `m_ep50_bs8_lr1e-04` was run there):

```
m_ep50_bs8_lr1e-03     m_ep50_bs16_lr1e-04    m_ep50_bs32_lr1e-04
m_ep50_bs8_lr1e-02     m_ep50_bs16_lr1e-03    m_ep50_bs32_lr1e-03
                       m_ep50_bs16_lr1e-02    m_ep50_bs32_lr1e-02
```

Cell 15 `["[Cell 16]"]` — "Training varian MEDIUM - Epochs = 50" — carries the comment
`# batal` ("cancelled") on its second line, which is why. `m_ep50_bs8_lr1e-04` exists
because it was run before the cancellation.

So `m_ep25_bs8_lr1e-02` is the argmax over 10 of 18 candidates. Script
`01_complete_medium_grid.py` runs exactly the eight above on the development split
under the legacy protocol, recomputes the argmax over all 18, and writes the result
back into `configs/arms.yaml`. The 46 already-completed configurations are **not**
re-run.

### 5.3 Legacy training call

`train_and_evaluate_variant`, `code.ipynb` **cell 9** `["[Cell 10]"]`. The protocol
script `01` must reproduce:

```python
model.train(
    data=str(YOLO_DATASET_DIR), epochs=epochs, imgsz=224, batch=batch_size,
    lr0=lr0, patience=20, seed=42, workers=4, optimizer="MuSGD",
    project=..., name=f"yolo26_{variant_key}_cls_ep{epochs}_bs{batch_size}_lr{lr0:.0e}",
    exist_ok=True,
)
```

Best weights are `runs/classify/<run_name>/weights/best.pt`, falling back to
`last.pt` if absent — Ultralytics selects `best.pt` on the val metric.

### 5.4 Class weights were computed but never applied

`code.ipynb` **cell 7** `["[Cell 8]"]` computes

```python
class_weights_arr = compute_class_weight(
    class_weight="balanced", classes=unique_classes, y=train_labels,
)
CLASS_WEIGHT_DICT = dict(zip(unique_classes, class_weights_arr))
```

and prints *"CLASS_WEIGHT_DICT diteruskan ke parameter loss YOLO saat training"*
("CLASS_WEIGHT_DICT is passed to the YOLO loss parameter during training"), followed by
a caveat that the installed Ultralytics version must support it.

**It is not passed.** A search of all 25 cells finds `CLASS_WEIGHT_DICT` referenced
only inside cell 7 itself (computation, a table, a bar chart). The `model.train(...)`
call in cell 9 takes no class-weight argument, and no other cell mentions the name.

Two consequences, both material:

1. `OLD_DIR/README.MD` lists *"Class-weighted loss untuk menangani ketidakseimbangan"*
   among the methods. The code does not support that claim.
2. The revision's class-weight ablation (script `04`) is therefore not a re-run of
   something already done — it is the **first** actual measurement of the effect.
   NEW_DIR applies balanced weights to the training loss for real, and the ablation's
   `class_weights: none` arm is what corresponds to the old behaviour.

The weighting scheme itself is carried over as specified: sklearn `"balanced"`,
`n_samples / (n_classes * n_samples_c)`, fitted on the **training** portion only.

---

## 6. Efficiency profiling

`profile_model`, `code.ipynb` **cell 8** `["[Cell 9]"]`:

| metric | how the old code obtained it |
| --- | --- |
| `params` | `sum(p.numel() for p in model.model.parameters())` |
| `gflops` | `thop.profile` on a `torch.randn(1, 3, 224, 224)` dummy, then `2.0 * macs / 1e9` |
| `model_size_mb` | `.pt` file size on disk / 1024² |
| `inf_time_ms` / `inf_time_std` | CPU, batch 1, **5** warm-up iterations discarded, then **100** timed, mean and std in ms |

Reference values from `metrics_canonical.csv` (constant per architecture):

| variant | params | GFLOPs | size MB (best run) | inf ms (best run) |
| --- | ---: | ---: | ---: | ---: |
| n | 1,543,914 | 0.398313 | 3.063 | 19.79 |
| s | 5,455,818 | 1.486392 | 10.538 | 31.87 |
| m | 10,366,026 | 4.851194 | 19.932 | 91.22 |

`size_mb` is not perfectly constant per variant in the old CSV (e.g. 19.932 vs 59.501
for medium, 3.060 vs 3.063 for nano) because it is the on-disk size of that run's
`best.pt`, which varies with what Ultralytics serialised. Recorded as-is; not averaged.

The old timing harness is **superseded** by `scripts/07_bench_edge.py`, which the brief
specifies far more strictly (50 warm-up, ≥200 timed, median/IQR/p95, with and without
letterbox, a 10-minute thermal soak, host provenance, peak RSS, INT8 size). The old
numbers above are kept only as a sanity reference, not as reportable results.

---

## 7. Learning curve — old configuration vs new

`code.ipynb` **cell 18** `["[Cell 19]"]` ran the curve at:

```python
LEARNING_CURVE_VARIANT = "n"
LEARNING_CURVE_FRACS   = [0.2, 0.4, 0.6, 0.8, 1.0]
LEARNING_CURVE_BATCH   = BATCH_SIZE_OPTIONS[1]   # 16
LEARNING_CURVE_LR0     = LR_OPTIONS[1]           # 1e-3   <-- not the winner
epochs = 50
```

The comment claims "best hyperparams per varian", but the code indexes the option
lists positionally and lands on **lr = 1e-3**, whereas the locked nano winner is
**lr = 1e-2** (§5.1). Fractions were drawn with
`train_test_split(range(n), train_size=frac, stratify=all_train_labels, random_state=42)`,
one draw per fraction, no repeats, and scored with `evaluate_blind` — so the cached
numbers also carry the §3 deflation.

Cached results, `code.ipynb` **cell 19** `["[Cell 19b]"]`, hard-coded:

| frac | n_train | val_f1_macro |
| ---: | ---: | ---: |
| 0.2 | 111 | 0.4836 |
| 0.4 | 222 | 0.6135 |
| 0.6 | 333 | 0.5583 |
| 0.8 | 444 | 0.6152 |
| 1.0 | 556 | 0.6880 |

**Not reusable.** Script `05_learning_curve.py` re-runs under the final locked nano
configuration (ep50, bs16, **lr 1e-2**), over the CV folds, 3 repeats, reporting mean
and std. The table above is recorded only to document what is being replaced.

---

## 8. Statistical testing in the old work

`code.ipynb` **cell 21** `["[Cell 20]"]`, results in `results/friedman_summary.json`:

```json
{"chi2": 2.3243243243243303, "p_value": 0.31280910516146615,
 "n_blocks": 10, "n_treatments": 3, "significant": false,
 "effective_split": "val", "source": "evaluate_clean (canonical)"}
```

Blocks are the 10 classes, treatments the three best variants, per-class F1 on the
**val** split as the response. `n = 10` blocks over a single split is the weakness the
revision addresses: `artifacts/folds.json` (15 folds) makes every arm comparison
paired across folds instead.

`results/friedman_within_nano.json` and `friedman_within_nano_nemenyi_val.csv` hold a
second test within the nano variant; retained as reference, not migrated.

---

## 9. Environment recorded from OLD_DIR/.venv

Actual installed versions at the time of reading (`pip freeze`), which seed
`requirements.txt`:

```
ultralytics==8.4.53        torch==2.12.0          torchvision==0.27.0
scikit-learn==1.8.0        numpy==2.4.6           pandas==3.0.3
scipy==1.17.1              matplotlib==3.10.9     pillow==12.2.0
opencv-python==4.13.0.92   PyYAML==6.0.3          tqdm==4.67.3
ultralytics-thop==2.0.19   thop==0.1.1.post2209072238
scikit-posthocs==0.13.0    statsmodels==0.14.6    seaborn==0.13.2
```

Notes:
- `torch==2.12.0` in OLD_DIR is the **CPU** build. Kaggle supplies a CUDA build; the
  pin is expressed so the platform wheel resolves (see `requirements.txt`).
- `seaborn` was used throughout the old plotting code. It is **dropped** — the brief
  requires matplotlib only, so `src/srpcard/figures.py` is written from scratch rather
  than ported from `scripts/export_publication_figures.py`.

---

## 10. Deliberate departures from the old code

| # | Old behaviour | New behaviour | Reason |
| --- | --- | --- | --- |
| 1 | Class names normalised at evaluation time (cell 20) | Normalised at load time, with a hard 10-class assert | Prevents §3 recurring anywhere downstream |
| 2 | Split materialised as copied directory trees under `yolo_dataset/` | Split stored as index lists in JSON; images referenced by integer `idx` | 15 folds × 5 arms cannot each be a directory copy |
| 3 | Single 80:10:10 split used for both selection and reporting | Dev split for selection only; 15 CV folds for reporting | Core purpose of the revision |
| 4 | Class weights computed, never applied (§5.4) | Applied for real; absence is the ablation arm | Makes the stated method true |
| 5 | Registry keyed by config string, overwritten in place | Append-only JSONL keyed by a deterministic `run_id` hash | Resumability; a session killed at run 40 loses nothing |
| 6 | Learning curve at lr 1e-3 (§7) | Learning curve at the locked lr 1e-2 | The old curve did not describe the reported model |
| 7 | `seaborn` for all plots | matplotlib only | Brief §10 |
| 8 | Paths hardcoded to `Path.cwd()` | All paths from `configs/data.yaml` | Kaggle / Pi / local portability |
| 9 | Timing: 5 warm-up + 100 iterations, mean ± std | 50 warm-up + ≥200 iterations, median / IQR / p95 + soak | Brief §5, script `07` |

---

## 11. Open items — nothing invented

Values the brief marks `TBD` and which are **not** recoverable from OLD_DIR, because
the old work never trained these architectures at all:

- `mobilenetv3_small`: learning rate. To be determined by `02_lr_sweep_baselines.py`.
- `resnet18`: learning rate. To be determined by `02_lr_sweep_baselines.py`.

Both are written as `null` in `configs/arms.yaml` with `lr_source: "TBD:02_lr_sweep"`,
and the loader refuses to run any arm whose `lr` is still `null`. They are not filled
with a placeholder number.

No other required input was missing. Nothing in this document is estimated.

---

## 12. Duplicate-label conflict groups: discovery, rule and exclusion

This section is written to be quoted in the manuscript.

### 12.1 Discovery

The new image index stores a SHA-1 content hash for every file, added so the
legacy split could be matched across the directory renaming (§13.4). Hashing the
695 images immediately exposed something the old pipeline never checked:
**13 groups of byte-identical files, each carrying more than one class label.**

The check is a two-line group-by, and it had never been run. Nothing in
`OLD_DIR/code.ipynb` hashes, compares or de-duplicates the images.

### 12.2 The 13 groups

Every group is listed in full. `idx` refers to the 695-row
`artifacts/image_index.csv`; the complete record, with file size and pixel
dimensions, is in `artifacts/excluded_images.csv`.

| sha1 (first 12) | n | idx | classes the same bytes are filed under |
| --- | ---: | --- | --- |
| `176da3a3da95` | 2 | 86, 455 | full_load_production / pump_leakage |
| `2360f2e520da` | 2 | 319, 571 | natural_flowing / severe_vibration |
| `2615678a91af` | 2 | 335, 669 | natural_flowing / vibration |
| `4d30394f598e` | 2 | 385, 456 | natural_flowing / pump_leakage |
| `5ff043ea3546` | 2 | 119, 694 | gas_influence / vibration |
| `70bf43d4b232` | **3** | 103, 160, 673 | gas_influence / gas_influence_and_vibration / vibration |
| `85804c6c8e43` | 2 | 384, 454 | natural_flowing / pump_leakage |
| `96c3cfa70782` | 2 | 299, 515 | natural_flowing / severe_vibration |
| `9dea713c5914` | 2 | 64, 682 | full_load_production / vibration |
| `ae3147353341` | 2 | 298, 514 | natural_flowing / severe_vibration |
| `d3d77796110b` | 2 | 439, 685 | pump_leakage / vibration |
| `e040de2a4061` | 2 | 351, 599 | natural_flowing / severe_vibration |
| `ecedf37912cd` | 2 | 386, 458 | natural_flowing / pump_leakage |

Two facts about this table:

- **All 13 groups are cross-class. Zero are within-class.** Benign duplication --
  the same card filed twice under one label -- does not occur at all.
- **All 13 groups share ONE filename across their differing classes.** For
  example `Screenshot 2026-05-05 140627.png` exists under both
  `natural_flowing/` and `pump_leakage/`, byte for byte.

The second fact is the diagnostic one. Two annotators disagreeing about an
ambiguous card would produce two *differently named* files, or one file with a
contested label. One filename, one byte sequence, two directories is the
signature of a **curation copy error** -- a file copied into a second class
directory during dataset assembly -- not of genuine expert disagreement about the
physics.

`natural_flowing` is involved in 8 of the 13 groups, more than any other class.

### 12.3 The exclusion rule

> Group the image index by SHA-1. A group whose members span more than one
> normalised class is a **conflict group**, and **every member of it is
> excluded** -- not merely the extra copies. A group that is byte-identical
> within a single class is benign duplication and is retained.

The rule keeps or drops the whole group rather than keeping one member, because
keeping either member requires deciding which label is correct. Without expert
adjudication that decision cannot be made, and getting it wrong plants a
confidently wrong label in the training data. Dropping the group is mechanical,
requires no judgement, and is auditable from the hash alone.

Implemented in `src/srpcard/data.py:annotate_conflicts`; asserted against
`configs/data.yaml:clean_corpus` by `src/srpcard/data.py:assert_clean_counts`.

**No expert adjudication was performed.** No domain expert was consulted about
which label is correct for any of the 27 files. The exclusion is a mechanical
consequence of the hash, and it is deliberately conservative: it discards
information rather than inventing a label.

### 12.4 Before / drop / after

| class | before | drop | after |
| --- | ---: | ---: | ---: |
| collide_pump_and_vibration | 35 | 0 | 35 |
| full_load_production | 52 | 2 | 50 |
| gas_influence | 33 | 2 | 31 |
| gas_influence_and_vibration | 70 | 1 | 69 |
| insufficient_liquid_supply_and_vibration | 106 | 0 | 106 |
| natural_flowing | 93 | 8 | 85 |
| pump_leakage | 70 | 5 | 65 |
| severe_insufficient_liquid_supply | 52 | 0 | 52 |
| severe_vibration | 133 | 4 | 129 |
| vibration | 51 | 5 | 46 |
| **TOTAL** | **695** | **27** | **668** |

13 conflict groups, 27 files, 3.9 % of the corpus. Imbalance ratio after
exclusion: **129 / 31 = 4.16:1** (before: 133 / 33 = 4.03:1).

### 12.5 Two-track corpus

The exclusion is applied to one track and deliberately not to the other.

| | corpus | labels | used by | role |
| --- | --- | --- | --- | --- |
| **Development split** | raw **695** | original, contaminated | scripts 01, 02 | hyperparameter selection only |
| **CV folds** | clean **668** | conflict groups removed | scripts 03, 04, 05 | all reporting |

The development split is *not* cleaned, and that is intentional. Its only
function is comparability with the 46 grid-search configurations that are already
complete. Those runs were trained and evaluated on the contaminated corpus and
cannot be retro-fixed without re-running them, which the revision forbids.
Cleaning the development split would make the 8 medium configurations completed
by script 01 incommensurable with the 10 that already exist, defeating the
purpose of completing that grid at all.

Correctness matters in the reporting protocol, and that is where the clean corpus
is used. `artifacts/dev_split.json` indexes into the 695;
`artifacts/folds.json` indexes into the 668. Both use the same `idx` space --
positions in the single 695-row `artifacts/image_index.csv` -- so the two views
never diverge into two competing index files. Membership is carried by the
`conflict_group` and `excluded` columns of that one file.

### 12.6 Effect on the fold protocol

After exclusion no SHA-1 is shared across classes, so plain
`RepeatedStratifiedKFold` is valid. Phase 3 asserts this anyway: no hash may
appear on both the train and test side of any fold. Should a benign within-class
group ever appear and straddle a boundary, the protocol switches to
`StratifiedGroupKFold` keyed on SHA-1.

---

## 13. Cross-references against the old results

Computed by `scripts/00_build_folds.py` phase 2b (`src/srpcard/legacy_audit.py`),
recorded in `artifacts/legacy_contamination.json`. Cross-reference 2 required
re-running **inference only**, with the already-trained selected weights; no
grid-search configuration was re-trained, and nothing was written to OLD_DIR.

### 13.1 How much of the reported test F1-macro was contaminated

Where the 27 excluded images fell in the legacy development split:

| partition | n | excluded | share |
| --- | ---: | ---: | ---: |
| train | 556 | 25 | 4.5 % |
| val | 69 | **1** | 1.5 % |
| test | 70 | **1** | 1.4 % |

11 of the 13 conflict groups sit **entirely inside the training partition** --
label noise during training, but not leakage. Only **2 groups straddle a
partition boundary**:

- `176da3a3da95` -- `full_load_production` (**idx 86, test**) and `pump_leakage`
  (idx 455, train)
- `9dea713c5914` -- `full_load_production` (idx 64, train) and `vibration`
  (**idx 682, val**)

So the reported test macro-F1 of **0.6979** rests on 70 images of which exactly
**one** is contaminated. Recomputing the metric with that image removed:

| | n | accuracy | F1-macro |
| --- | ---: | ---: | ---: |
| as published | 70 | 0.7000 | **0.6979** |
| minus the one contaminated image | 69 | 0.7101 | **0.7120** |

Difference **+0.0141**. The contamination *understated* the published figure
rather than inflating it. The reported number is not overstated, and the headline
claim is not at risk from this.

A note on the recomputation: running the stored selected weights over the
recovered test partition reproduces accuracy 0.7000 and F1-macro 0.6979 -- the
published values -- exactly. That is an independent end-to-end confirmation of
the split reconstruction (§13.4), of the 10x10 collapse, and of the published
metric itself.

### 13.2 Are the "inter-class similarity" errors actually contamination?

The manuscript attributes a specific set of off-diagonal confusion-matrix cells
to inter-class visual similarity. Collapsing the selected model's matrix to 10x10
canonical order and identifying every misclassified image:

| cell | errors | involving an excluded image |
| --- | ---: | ---: |
| natural_flowing -> pump_leakage | 3 (idx 303, 318, 326) | **0** |
| pump_leakage -> natural_flowing | 2 (idx 408, 437) | **0** |
| natural_flowing -> severe_vibration | 1 (idx 364) | **0** |
| **total** | **6** | **0** |

**The manuscript's explanation survives.** None of those six errors involves a
contaminated image. They are genuine confusions between visually similar classes,
and the paragraph attributing them to inter-class similarity stands as written.

This is a positive finding rather than an absence of evidence: all eight excluded
`natural_flowing` images and all five excluded `pump_leakage` images sit in the
training partition, so none of them *could* have produced a test-set error.

Exactly one of the model's 21 test errors involves an excluded image, and it
falls in a different cell:

> **idx 86**, true `full_load_production`, predicted **`pump_leakage`**.

Its byte-identical twin, idx 455, is labelled `pump_leakage` and sits in the
training set. The model was trained on those exact pixels under the label
`pump_leakage` and then reproduced that label at test time. This single error is
fully explained by contamination and not by visual similarity -- the one place
where the similarity narrative does not apply.

### 13.3 Which split did the within-nano Friedman test run on?

Reported verbatim, as requested.

`OLD_DIR/results/friedman_within_nano.json` **has no `effective_split` field at
all.** Its top-level keys are `["val", "test"]` -- the test was computed on both
partitions and both results were stored:

| block | chi2 | p | significant | Nemenyi stored |
| --- | ---: | ---: | --- | --- |
| `val` | 15.2769 | **0.0092** | **true** | **yes** |
| `test` | 9.3513 | 0.0958 | false | no (`null`) |

The only post-hoc file on disk is `friedman_within_nano_nemenyi_val.csv`. There
is no `..._test.csv`, because cell 21 runs the Nemenyi post-hoc only when the
omnibus is significant, and only the **val** block was.

Therefore: **the significant within-nano result and its Nemenyi post-hoc come
from the validation partition, not the test partition.** If the manuscript states
that this test ran on the test partition specifically to avoid circularity, that
statement is not supported by the stored artefacts. On the test partition the
omnibus is *not* significant (p = 0.0958).

The circularity concern is concrete rather than theoretical: the winning
configuration was itself selected by argmax of validation macro-F1 (§5.1), so the
significant val-based ranking is computed on the very partition that chose the
winner.

For contrast, the sibling file `friedman_summary.json` -- the across-variant test
-- *does* carry the field, and it reads `"effective_split": "val"`.

**This paragraph of the manuscript needs revision.** Unlike §13.2, it is not
rescued by the evidence.

### 13.4 Recovery of the development split: which route worked

Three routes were attempted in the order specified.

**Route A succeeded**, and is the route used. The old run materialised its split
as directory trees under `OLD_DIR/yolo_dataset/{train,val,test}/<class>/`, whose
directory names preserve the pre-rename state -- including the four prefixed
names `10_severe_vibration`, `12_full_load_production`, `29_natural_flowing`,
`30_pump_leakage`.

Matching was by SHA-1 first, filename second:

- **0 of 695 matched by SHA-1.** The materialised files are letterboxed
  re-encodings, so their bytes differ from the raw originals. This was expected.
- **695 of 695 matched by filename**, scoped to the normalised class. Zero
  unmatched, zero ambiguous, zero double-assigned, zero index rows left
  unassigned -- a bijection.

Filename matching is unambiguous because filenames are unique *within* a class.
They are not globally unique -- the 27 conflict-group files are precisely the
cases where one filename appears under several classes -- which is why the match
is scoped by class rather than by filename alone.

**Route B was also executed, for comparison.** Replaying the two original
`train_test_split` calls reproduces the required count table exactly, so on the
counts alone it looks like a success. It is not:

> **Route A and Route B agree on only 470 of 695 images (67.6 %).**

They place 225 images in different partitions while producing identical per-class
counts. The count table cannot distinguish a correct reconstruction from an
incorrect one, because stratified allocation depends only on class counts and
`random_state`, never on row order -- and the old row order came from an
OS-dependent `rglob` over 14 directory names that no longer exist. Had Route A
been unavailable, Route B would have passed the mandated assert while silently
reconstructing the wrong split.

Route C was not reached.

The recovered split is written to `artifacts/dev_split.json` together with the
full recovery report. Because OLD_DIR does not exist on Kaggle, Route A cannot
run there, so `artifacts/dev_split.json` is **committed** alongside `folds.json`
and `image_index.csv` -- a deliberate departure from the brief's list of
committed artefacts, without which scripts 01 and 02 could not run on Kaggle at
all.

---

## 14. Additional modules beyond the brief's list

Items that exist but that the brief's target structure does not name. Each is
listed here rather than passed off silently.

| item | why |
| --- | --- |
| `src/srpcard/config.py` | Section 7 requires one `set_seed` helper and one place that resolves every path. Neither belongs in `data.py`, `folds.py` or `train.py`. Also holds provenance capture (git commit, installed versions). |
| `src/srpcard/legacy_audit.py` | Makes the §13 cross-references reproducible rather than ad hoc. Read-only with respect to OLD_DIR. |
| `configs/data.yaml: clean_corpus` | The clean-corpus contract of §12. |
| `configs/data.yaml: legacy_reference` | The read-only path to OLD_DIR and the artefacts read from it. |
| `artifacts/excluded_images.csv` | Full record of the 27 excluded files. |
| `artifacts/legacy_contamination.json` | The §13 cross-reference results. |
| `artifacts/legacy_test_predictions.csv` | Per-image predictions behind §13.2. |
| `artifacts/dev_split.json` **committed** | Route A cannot run on Kaggle (§13.4). |

`set_seed` additionally forces `torch.backends.cudnn.deterministic = True` and
`torch.backends.cudnn.benchmark = False`; benchmark mode selects convolution
algorithms by timing them, which is nondeterministic. The ~10-15 % slowdown is
accepted. Ultralytics may not honour `torch.use_deterministic_algorithms`, so the
achieved status is recorded into the registry rather than hard-failing.

Seeds are pure functions of `(repeat, fold)` and identical across every arm:
`run_seed = 10000 + repeat*100 + fold`, `val_seed = run_seed + 50000`. A
per-repeat seed was considered and rejected: it would make all five folds within
a repeat share one initialisation, sampling training stochasticity three times
instead of fifteen and confounding partition with seed.

---

## 15. Best-weight selection criterion: why it is macro-F1, not loss

Recorded because the criterion was changed once, deliberately, and the reason
matters more than the choice.

### 15.1 What was tried first

The first implementation selected the best epoch by **validation loss**
(unweighted cross-entropy on the fold's 10 % slice). The reasoning was that the
slice is only 54 images and the rarest class, `gas_influence`, contributes 2 of
them (`artifacts/folds_report.md`), so macro-F1 moves in coarse steps for that
class while loss is continuous and less noisy.

### 15.2 The evidence that it was wrong

A single smoke run — `yolo26n`, repeat 0 fold 0, the locked configuration
(50 epochs, batch 16, lr 1e-2) — showed the two criteria disagreeing by 27
epochs:

| criterion | selected epoch | val macro-F1 | val loss |
| --- | ---: | ---: | ---: |
| validation loss (rejected) | 7 | 0.5446 | **1.2384** |
| validation macro-F1 (adopted) | **34** | **0.6182** | 1.4359 |

Validation loss bottomed out at epoch 7 and rose monotonically thereafter, while
validation macro-F1 kept climbing to epoch 34. This is the calibration drift
documented by Guo et al. (2017), *On Calibration of Modern Neural Networks*:
validation NLL degrades while validation accuracy continues to improve, because
the network becomes overconfident on a minority of examples faster than it stops
getting them right. On a 54-image slice, cross-entropy is dominated by those few
confident errors.

### 15.3 Why that is disqualifying rather than merely unfortunate

The trade was noise for **bias**, and bias is worse here:

1. Macro-F1's quantisation noise averages out over 15 folds. Loss-based
   under-training does not: it selects an early, under-trained checkpoint on
   *every* fold of *every* arm, in the same direction each time.
2. Overconfidence drift rates differ between architectures. Loss selection would
   therefore penalise the five arms unequally, contaminating the
   between-architecture comparison — the study's core claim — and not merely the
   level of the numbers.
3. Selecting on one metric while reporting another invites an obvious reviewer
   question for no compensating benefit.
4. The quantisation concern was overstated. `gas_influence` contributes one tenth
   of the macro average, so its 2-image granularity shifts the criterion by only
   about 0.05. The loss tie-break absorbs the residual.

### 15.4 The rule as it now stands

> Select the epoch with the highest **validation macro-F1** on the fold's 10 %
> slice. Break ties with the lower **unweighted** validation cross-entropy.

Properties that did not change: a single fixed rule, identical across all five
arms and both ablation arms, computed on the fold's validation slice, and never
the weighted loss — so the class-weight ablation still compares like with like.

Each registry record keeps the full per-epoch `history` (`val_loss`, `val_f1`,
`val_acc`, `train_loss`, `lr`), plus `selected_epoch` and — for the record —
`min_val_loss_epoch`, the epoch a loss criterion would have chosen. The
distribution of `selected_epoch` across the 75 runs is therefore auditable after
the fact: if it clusters near the epoch budget, the budget is too short.

### 15.5 Early stopping removed

`patience=20` is gone; the full locked epoch budget now runs every time.

It was an unvalidated hyperparameter, it made "50 epochs" not actually 50 epochs
when the budget is itself a locked grid-search output, and it interacted with the
selection criterion (under loss selection the smoke run stopped at epoch 27 of
50, discarding the region where macro-F1 was still improving). On GPU the compute
is not the binding constraint. `stopped_early` is recorded as `false` for
provenance.

### 15.6 An honest note on the smoke run

Under the adopted criterion that run scored **test macro-F1 0.5484**; under the
rejected one it scored **0.5848**. On this single fold the rejected criterion
happened to give the better test score.

That is one fold out of fifteen and it settles nothing — the case for macro-F1
selection is the systematic-bias argument in §15.3, not a per-fold outcome. It is
recorded here so the comparison is not quietly omitted.
