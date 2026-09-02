# Kaggle setup

How to upload the 695 images as a private Kaggle Dataset and wire `DATA_ROOT` to it.

---

## 1. The dataset

| | |
| --- | --- |
| slug | `raihanakirar/srp-dyna-card` |
| visibility | **Private** |
| mount point | `/kaggle/input/srp-dyna-card` |
| `DATA_ROOT` | `/kaggle/input/srp-dyna-card/dataset` |

Note the `dataset/` **wrapper directory** inside the mount. `DATA_ROOT` points one
level below the mount point, not at it.

### Required layout

The data is raw and labelled only by directory. There is no train/val/test
structure — all partitioning happens in code.

```
srp-dyna-card.zip
└── dataset/
    ├── collide_pump_and_vibration/           35 images
    ├── full_load_production/                 52
    ├── gas_influence/                        33
    ├── gas_influence_and_vibration/          70
    ├── insufficient_liquid_supply_and_vibration/  106
    ├── natural_flowing/                      93
    ├── pump_leakage/                         70
    ├── severe_insufficient_liquid_supply/    52
    ├── severe_vibration/                    133
    └── vibration/                            51
                                       total 695
```

**Exactly 10 directories, no numeric prefixes.** The uploaded copy is already
clean. The loader still strips a leading `^\d+_` defensively, because the older
copy in `../FINAL-pipeline` had 14 directories where four classes were duplicated
under a prefix (`10_severe_vibration` beside `severe_vibration`) — a collision
that silently deflated every metric in the previous study. If normalisation does
not leave exactly 10 classes, loading fails and names the offending directories.

### Uploading

1. Zip the `dataset/` directory so the archive contains `dataset/<class>/*.png`.
2. kaggle.com → **Datasets** → **New Dataset**.
3. Title `srp-dyna-card`, visibility **Private**.
4. Upload the zip; Kaggle expands it at the mount point.
5. Confirm the slug reads `raihanakirar/srp-dyna-card`.

Or with the CLI:

```bash
mkdir -p upload && cp -r dataset upload/
cd upload
kaggle datasets init -p .
# edit dataset-metadata.json: "title": "srp-dyna-card",
#                             "id": "raihanakirar/srp-dyna-card"
kaggle datasets create -p . --dir-mode zip
```

To replace the data later:

```bash
kaggle datasets version -p upload -m "reason for the new version"
```

### Verifying after attachment

```python
!ls -1 /kaggle/input/srp-dyna-card/dataset          # expect 10 directories
!find /kaggle/input/srp-dyna-card/dataset -type f | wc -l   # expect 695
```

`scripts/00_build_folds.py` asserts the per-class counts and the total, and fails
with a per-class diff if either is wrong. Do not work around that failure — a
count mismatch means the committed `image_index.csv` and `folds.json` no longer
describe the data, and every fold index would point at the wrong image.

---

## 2. Wiring `DATA_ROOT`

Nothing is hardcoded to `/kaggle`. Resolution order:

1. `$SRPCARD_DATA_ROOT` — what the notebook sets.
2. `configs/data.yaml: data_root` — defaults to the Kaggle path.

```python
import os
os.environ['SRPCARD_DATA_ROOT'] = '/kaggle/input/srp-dyna-card/dataset'
```

**Wrapper safety net.** If the resolved directory contains exactly one
subdirectory and that subdirectory holds the class directories, the loader
descends into it and logs that it did. So pointing at the mount
(`/kaggle/input/srp-dyna-card`) instead of one level down also works.

Locally:

```bash
export SRPCARD_DATA_ROOT=/path/to/dataset
python scripts/00_build_folds.py
```

---

## 3. Notebook setup

1. **Code** → **New Notebook**.
2. **File → Import Notebook** → upload `notebooks/kaggle_runner.ipynb`.
3. Right panel → **Input** → **Add Input** → your private dataset.
4. Right panel → **Settings**:
   - **Accelerator: GPU T4 x2** (or P100). Without it, training falls back to CPU
     and takes roughly an order of magnitude longer.
   - **Internet: On** — required to clone the repository and let ultralytics
     download pretrained checkpoints.
   - **Persistence**: leave off; `artifacts/` is carried out explicitly instead.
5. `REPO_URL` in cell 1 is already set to
   `https://github.com/akiraraihaan/SRP-conditions-predict-using-ML.git`. For a
   private repository use a token:
   `https://<token>@github.com/akiraraihaan/SRP-conditions-predict-using-ML.git`,
   and prefer a Kaggle Secret over pasting the token into the notebook.

---

## 4. Surviving the 12-hour session limit

Kaggle sessions terminate after about 12 hours. Every script is resumable:
`artifacts/registry.jsonl` is append-only and flushed after each completed run,
so a session killed at run 40 loses nothing.

The loop is:

1. Run one script.
2. **Always** run the "copy artifacts back out" cell, even after an interruption.
3. Download `artifacts_bundle.zip` from the output panel.
4. Commit `artifacts/registry.jsonl` — and `configs/arms.yaml` if scripts 01 or 02
   changed it.
5. Next session: **git is the primary resume path.** `registry.jsonl` is committed
   (it has a `.gitignore` exception), so the clone in cell 1 brings it back and the
   scripts skip what is already done. Nothing needs uploading as a Dataset.
   `RESUME_FROM` in cell 3 is the **fallback**, for when a session's results could
   not be committed; it overwrites `artifacts/` from a private Dataset, so a stale
   one hides newer committed results.

Re-running a completed script is safe and cheap: it prints
`N already complete (skipped), 0 remaining` and exits.

---

## 5. Script order

| order | script | runs | writes back to `configs/arms.yaml` |
| ---: | --- | ---: | --- |
| 1 | `00_build_folds.py` | — | no — runs the phase 0 preflight first |
| 2 | `01_complete_medium_grid.py` | 8 | **yes** — `yolo26m` epochs/batch/lr |
| 3 | `02_lr_sweep_baselines.py` | 6 | **yes** — both baselines' `lr` |
| 4 | `03_run_cv.py` | 75 | no |
| 5 | `04_run_ablation.py` | 15 | no |
| 6 | `05_learning_curve.py` | 75 | no |
| 7 | `06_export_figures.py` | — | no |

**Commit `configs/arms.yaml` after steps 2 and 3.** The next session clones the
repository fresh; an uncommitted hyperparameter is lost, and `03_run_cv.py` would
silently train the provisional `yolo26m` configuration or refuse to run the two
baselines (`lr: null`).

`07_bench_edge.py` does **not** run on Kaggle. It is standalone, CPU-only, and
belongs on the Raspberry Pi.

### Preflight, first thing in every session

`00_build_folds.py` opens with a preflight that reports the GPU and torch/CUDA
versions, whether `cudnn.deterministic` took effect, the resolved DATA_ROOT and the
per-class counts actually found, whether the committed artefacts still match their
fingerprints, and whether all five arms' pretrained checkpoints load. It exits
non-zero if any of that failed. `--preflight-only` runs just that part.

### `--allow-pretrained-fallback`

Each YOLO arm declares a YOLO11 `pretrained_fallback`. It is never taken
automatically: a run that trained YOLO11 while every table said YOLO26 could not be
detected afterwards. If the YOLO26 checkpoint cannot be loaded, scripts 01–05
refuse and name both architectures. Passing `--allow-pretrained-fallback` permits
the substitution, prints a loud banner, and sets `pretrained_fallback_used` on
every affected registry record. Do not pass it to get a session unstuck.

---

## 6. Things that will bite

| symptom | cause | fix |
| --- | --- | --- |
| `DATA_ROOT does not exist` | dataset not attached, or wrong slug | Add Input; check the slug |
| count-mismatch failure in phase 1 | wrong or partial upload | re-upload; do not bypass the assert |
| `Corpus fingerprint mismatch` | `folds.json` was built on different data | investigate before doing anything else — every fold index would be wrong |
| `arm 'resnet18' has no learning rate yet` | script 02 not run, or `arms.yaml` not committed | run 02, commit `arms.yaml` |
| no GPU | accelerator not enabled | Settings → Accelerator → GPU |
| ultralytics cannot download weights | Internet off | Settings → Internet → On |
| the run dies at 12 h | session limit | expected; copy artifacts out and resume |
