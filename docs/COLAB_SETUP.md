# Running on free Google Colab

The primary execution path. It needs a Google account and nothing else — no card,
no verification, no deposit.

`notebooks/colab_runner.ipynb` is the runner. It contains no experiment logic: it
mounts Drive, clones the repository, points `DATA_ROOT` at the dataset in Drive,
redirects `artifacts/` into Drive, calls one script, and zips the results for
committing.

---

## 1. Why this path

| platform | blocker |
| --- | --- |
| Kaggle | GPU and Internet are both gated behind account verification |
| Google Cloud Free Trial | the trial billing account forbids attaching GPUs to VM instances and forbids quota-increase requests; new-account GPU quota is zero, so the $300 credit cannot buy a GPU without a card, an upgrade and an approved quota request |
| RunPod | requires a $10 deposit |
| **free Colab** | **none — a Google account is enough** |

The cost is reliability, not correctness: free Colab interrupts you. Everything
below is about making an interruption cheap rather than pretending it will not
happen.

---

## 2. Drive layout

Two directories at the top of your Drive. Create the first by uploading; the
notebook creates the second.

```
MyDrive/
├── srp-dyna-card/
│   └── dataset/                 <- the wrapper directory. Do not flatten it.
│       ├── collide_pump_and_vibration/
│       ├── full_load_production/
│       ├── gas_influence/
│       ├── gas_influence_and_vibration/
│       ├── insufficient_liquid_supply_and_vibration/
│       ├── natural_flowing/
│       ├── pump_leakage/
│       ├── severe_insufficient_liquid_supply/
│       ├── severe_vibration/
│       └── vibration/
└── srp-artifacts/               <- created by cell 5; artifacts/ points here
```

Both paths are set in **cell 0** of the notebook and nowhere else:

```python
DATA_ROOT_DRIVE = '/content/drive/MyDrive/srp-dyna-card/dataset'
ARTIFACTS_DRIVE = '/content/drive/MyDrive/srp-artifacts'
```

### Uploading the 695 images

The `dataset/` wrapper must survive the upload. `configs/data.yaml` documents it,
and `resolve_data_root` will descend through a single wrapper directory if you point
one level too high — but only one level, and only when the directory it finds is
unambiguous. Getting it right is easier than relying on that.

The reliable route is a zip:

1. On your machine, zip the directory **containing** the 10 class directories so the
   archive holds `dataset/<class>/...`.
2. Upload the zip to `MyDrive/srp-dyna-card/` through the Drive web interface. One
   large file uploads far faster and far more reliably than 695 small ones, and the
   Drive web uploader silently drops files often enough to matter at this count.
3. Unzip it from the notebook, once:

   ```python
   !cd /content/drive/MyDrive/srp-dyna-card && unzip -q dataset.zip
   ```

Do not drag 695 loose files into the browser. A partial upload produces per-class
counts that are quietly wrong, and while `00_build_folds.py` asserts those counts and
would catch it, you will have wasted the session discovering it.

### Verifying the upload

Cell 4 of the notebook checks this before anything else runs. It prints the resolved
path and the per-class counts, and **fails printing that path** when the 10 canonical
class directories are not there. The counts it prints must match
`configs/data.yaml:expected_counts`:

| class | n |
| --- | ---: |
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

The most common failure is pointing `DATA_ROOT_DRIVE` at `srp-dyna-card/` instead of
`srp-dyna-card/dataset/`.

---

## 3. The `artifacts/` symlink — the important part

Cell 5 replaces `artifacts/` in the clone with a symlink to `ARTIFACTS_DRIVE`.

Every file the scripts write therefore lands in Drive **at the moment it is
written**: `registry.jsonl` after every completed run (it is flushed and `fsync`'d),
the summary tables, the figures. Nothing is staged in the container waiting for you
to copy it out.

**A dropped session therefore costs at most the run that was in flight.** Reconnect,
re-run cells 1–5, re-run the script; it skips everything already in the registry.

The cell copies in two directions, and they are deliberately not symmetric:

- **Frozen inputs → Drive, every session.** `image_index.csv`, `folds.json`,
  `dev_split.json`, `excluded_images.csv`, `folds_report.md`,
  `legacy_contamination.json`, `legacy_grid_metrics.csv` and
  `legacy_test_predictions.csv` are refreshed from the clone. The repository is
  authoritative for these; no script writes them.
- **`registry.jsonl` ← Drive, never the reverse.** The committed copy is **empty**.
  Copying it over Drive's would erase every completed run. The cell seeds Drive from
  the clone only when Drive has no registry at all.

Re-running cell 5 is safe: it detects an existing symlink and leaves the Drive
contents alone.

**Re-run cell 5 after every re-run of the clone cell.** A fresh clone brings a
plain `artifacts/` directory, so the symlink is gone until cell 5 re-establishes
it, and anything written in between goes to the container's disk and dies with
the session. Cell 2 says so when it happens, and the run cell refuses to start
until the symlink is back.

### What does *not* go to Drive

`runs/` — training weights and ultralytics output. Large, regenerable, and
gitignored. It stays on the container's local disk and dies with the session, which
is what you want.

`configs/arms.yaml` also stays in the clone — it is a tracked config file, not a
generated artefact, so it is not symlinked. Scripts 01 and 02 **rewrite** it with the
hyperparameters they select, and it dies with the session.

That would be dangerous on its own, because `epochs`, `batch` and `lr` feed the
`run_id` hash: a clone carrying the reverted config does not resume, it retrains the
same folds under the old settings. So scripts 01 and 02 also write
**`artifacts/resolved_arms.yaml`** — a full snapshot of the resolved config, with the
resolving script and timestamp in the header. That file *is* in `artifacts/`, so Drive
has it the moment it is written.

In a fresh clone:

```bash
python scripts/restore_arms.py --check   # show what differs, change nothing
python scripts/restore_arms.py           # put the resolved values back
```

Then commit `configs/arms.yaml`. Cell 4 of `00_build_folds.py`'s preflight tells you
whether you need to: it reports every arm's state and whether the snapshot differs
from the live file. See HANDOVER.md §4.5 and §4.6.

Still download and commit `configs/arms.yaml` when a resolving script finishes — the
snapshot is a safety net, not a substitute.

---

## 4. Runtime setup

1. **Runtime → Change runtime type → Hardware accelerator: GPU** → Save.
2. Run cells 0–5 in order.
3. Cell 3 prints the resolved torch version, CUDA version and GPU name.

### When Colab hands you a CPU runtime

Free-tier GPU quota is dynamic and undisclosed. After heavy use Colab will give you
a CPU-only runtime without saying so — the notebook still runs, the scripts still
work, and everything is simply ~10× slower.

Cell 3 makes this loud rather than silent. If it prints the CPU-ONLY banner:

1. **Runtime → Change runtime type → GPU → Save.** If it reconnects with a GPU,
   re-run cells 1–5 and carry on.
2. If GPU is greyed out or the banner returns, the quota is spent. It replenishes on
   its own, typically after several hours to a day. There is no appeal and no
   indicator of how long.
3. Meanwhile, `00_build_folds.py --preflight-only` and `06_export_figures.py` are
   both fine on CPU. So is a single fold for a sanity check:
   `03_run_cv.py --arms yolo26n --repeat 0 --fold 0`.

Do not start a 75-run script on CPU. It will not finish, and the runs it does
complete are still valid — they are recorded with the device in
`determinism_status` — but you will burn a session for a handful of folds.

---

## 5. Working with the session limits

| limit | consequence |
| --- | --- |
| ~90 min idle disconnect | the runtime is reclaimed when the browser tab stops talking to it |
| dynamic session length | no guaranteed 12 h; a session can end at any point |
| dynamic GPU quota | heavy use drops you to CPU for hours |

**Keep the tab open and the machine awake while a script runs.** Idle disconnection
is measured from browser activity, not from what the GPU is doing.

**Split the long scripts across sessions and across days.** `03_run_cv.py` is 75 runs
and `05_learning_curve.py` another 75; neither is a single-sitting job on free tier.
The intended rhythm:

1. Start a session, run cells 0–5.
2. Start the script. Let it run for as long as the session lasts.
3. When it ends — by your hand or Colab's — do nothing special. Drive already has
   every completed run.
4. Next session: cells 0–5 again, same script again. It prints
   `N already complete (skipped), M remaining` and continues.

Cell 8 shows progress against the target run count for each script, so you can see
how many sessions are left.

### Suggested order

| order | script | runs | notes |
| ---: | --- | ---: | --- |
| 1 | `00_build_folds.py` | — | every fresh session; runs the preflight |
| 2 | `01_complete_medium_grid.py` | 8 | **rewrites `configs/arms.yaml`** — commit it; also snapshots to `artifacts/` |
| 3 | `02_lr_sweep_baselines.py` | 6 | **rewrites `configs/arms.yaml`** — commit it; also snapshots to `artifacts/` |
| 4 | `03_run_cv.py` | 75 | expect several sessions |
| 5 | `04_run_ablation.py` | 15 | needs 4's `yolo26n` folds |
| 6 | `05_learning_curve.py` | 75 | expect several sessions |
| 7 | `06_export_figures.py` | — | CPU is fine |

`07_bench_edge.py` runs on **none** of these platforms. It is CPU-only by design and
refuses to run when CUDA is visible; it belongs on the Raspberry Pi.

---

## 6. Resuming after a dropped session

Nothing to recover. In order:

1. Reconnect (or open a new runtime).
2. Run cells 0–5. Cell 5 prints where the symlink points and how many completed runs
   the registry already holds — that number is your proof the state survived.
3. Re-run the same script. It resumes.

If cell 5 reports fewer runs than you expect, stop and check that
`ARTIFACTS_DRIVE` is the same path as last session before running anything: a typo
there creates a second, empty artifacts directory rather than failing.

---

## 7. Committing results

Drive holds the live copy; git holds the versioned history. They are different jobs,
and cell 7 is for the second one only.

Run cell 7 when you want to commit. The bundle mirrors the repository layout —
paths inside are `artifacts/...` and `configs/arms.yaml` — so it unpacks straight
over a checkout. A copy also goes to Drive so it survives the session.

```bash
unzip -o artifacts_bundle.zip
git add artifacts configs/arms.yaml && git status
```

Then commit:

- `artifacts/registry.jsonl` — the resume state and the record of every completed run
- `configs/arms.yaml` — after scripts 01 and 02 only
- `artifacts/resolved_arms.yaml` — the snapshot of those resolved hyperparameters
- the summary tables and `artifacts/figures/` as they appear

See HANDOVER.md §4.7 for the full list of what to commit and when.

Unlike the Kaggle runner, **you do not have to run cell 7 before the session ends**.
Missing it costs you a commit, not a session's work.

---

## 8. Things that will bite

| symptom | cause | fix |
| --- | --- | --- |
| cell 4 fails naming the path | `DATA_ROOT_DRIVE` points at `srp-dyna-card/` not `.../dataset/` | add the `dataset/` wrapper to the path |
| per-class counts are short | partial Drive upload | re-upload as a zip and unzip in place (§2) |
| cell 5 reports 0 completed runs when you expect more | `ARTIFACTS_DRIVE` differs from last session, **or the clone cell was re-run without re-running cell 5** | fix the path, or re-run cell 5; the clone replaces the symlink with a plain directory and cell 2 now warns when it has |
| everything is ~10× slower | CPU-only runtime | §4 |
| `03_run_cv.py` skips the two baselines | their `lr` is still `null` | run `02_lr_sweep_baselines.py` and commit `configs/arms.yaml` |
| a script aborts with HYPERPARAMETER DRIFT | `configs/arms.yaml` reverted after a dropped session, and completed runs disagree with it | `python scripts/restore_arms.py`, then commit `configs/arms.yaml`. HANDOVER.md §4.5 |
| the preflight says the snapshot DIFFERS | same cause, caught before anything runs | same fix |
| `06_export_figures.py` refuses with MIXED HYPERPARAMETERS | an arm has runs from two regimes in the registry | pick the real regime, delete the other's `run_ids`, restore the config. HANDOVER.md §4.5 |
| a script refuses, naming two architectures | the YOLO26 checkpoint would not download | see HANDOVER.md §4.2; do **not** reach for `--allow-pretrained-fallback` by reflex |
| the registry warns about schema drift | a record predates the current schema | HANDOVER.md §4.3 — delete the offending line and let the run happen again |
| reading images is slow | the dataset is read from Drive over FUSE each session | optional: `!cp -r "$SRPCARD_DATA_ROOT" /content/dataset` then set `os.environ['SRPCARD_DATA_ROOT'] = '/content/dataset'`. Costs a minute, saves it back on every script. |
