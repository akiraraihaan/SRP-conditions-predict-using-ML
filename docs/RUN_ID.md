# What a `run_id` is, and how to verify one

A `run_id` is the identity of a run. It is a 16-character SHA-1 prefix over the
fields that **define** the run — never over its outcome. The resume path works
by this hash: a planned run whose `run_id` already appears in
`artifacts/registry.jsonl` is skipped.

That makes it load-bearing in a way an ordinary id is not. If a hash **moves**,
the run stops being recognised, the next dry run proposes repeating it, and an
append-only registry ends up holding two records for one run under two names.

```python
RUN_ID_FIELDS = (
    "script", "arm", "architecture", "split_kind", "repeat", "fold",
    "epochs", "batch", "lr", "class_weights", "run_seed",
    "optimizer",          # an OVERRIDE only -- see below
    "extra",
)
```

Anything not in that tuple — metrics, timings, hardware, model size — is an
outcome, not an identity, and must never enter the hash.

---

## The trap: `extra` is hashed as a marker, not as the dict

`extra` is in `RUN_ID_FIELDS`, so you would expect a record's `extra` field to
be what was hashed. **It is not.** Every script hashes a short string marker,
while the record stores a rich dictionary under the same key.

| script | value hashed as `extra` | what the record stores under `extra` |
| --- | --- | --- |
| `01_complete_medium_grid` | `"legacy_protocol"`, or `"control_rerun"` for the AMP control | `{"key": "m_ep50_bs8_lr1e-03", "protocol": "legacy_unweighted_ultralytics", …}` |
| `01b_uniform_grid` | `"uniform_grid"` | `{"key": …, "f1_macro_val": …, …}` |
| `02_lr_sweep_baselines` | `"lr_sweep"` | `{"key": …, "f1_macro_val": …, …}` |
| `03_run_cv` | `None` | `{"protocol": "uniform", "selection_metric": …, …}` |
| `04_run_ablation` | `None` | `{"protocol": "uniform", …}` |
| `05_learning_curve` | `"lc_frac0.20"` … `"lc_frac1.00"` | `{"protocol": "uniform", "fraction": 0.2, "n_train": …}` |

**How this was found.** A sweep that tried to recompute all 234 stored
`run_id`s reported 144 of them unverifiable. Nothing was wrong with the
registry: the sweep had guessed the wrong markers. A record that cannot verify
its own identity without reading the source of the script that wrote it is not
reproducible in any sense that matters, and the failure mode is silent — it
looks exactly like corruption.

### The fix, for records written from now on

`registry.build_record()` takes a **required** `run_id_extra`, and every call
site passes the same object the spec was hashed with:

```python
run_id = registry.compute_run_id(**spec)
...
registry.build_record(..., run_id_extra=spec["extra"])
```

The record then carries `extra.run_id_extra`, and verification needs nothing
but the record. It is required with no default because a field that can be
forgotten will be.

**Existing records are not backfilled.** The registry is append-only, and a
harmless-looking rewrite of 234 lines is exactly the kind of change that loses
something quietly. The table above is how you verify a record written before
this field existed.

---

## `optimizer` is an OVERRIDE, not the optimizer that was used

`optimizer` was added to `RUN_ID_FIELDS` so that a run deliberately deviating
from an arm's configured optimizer cannot collide with, or be skipped because
of, the arm's existing records.

It hashes **a deviation from what `configs/arms.yaml` declares**, not the value
that was used. `None` means "whatever the arm declares" and is omitted from the
payload entirely.

This distinction is not cosmetic:

```yaml
yolo26n:            optimizer: MuSGD
yolo26s:            optimizer: MuSGD
yolo26m:            optimizer: MuSGD
mobilenetv3_small:  optimizer: SGD
resnet18:           optimizer: SGD
```

**165 of the 234 records are YOLO runs, and they were trained with MuSGD.** A
scheme where "absent means SGD" — which looks obviously right, and was the
first attempt — would have hashed `"musgd"` for every one of them, moved all 45
YOLO `run_id`s in script 03, and proposed re-running the entire YOLO campaign.
The arm's own choice is already pinned by `arm` plus the snapshot in
`artifacts/resolved_arms.yaml`; hashing it again is redundant *and* wrong.

So for the MuSGD arms the contrast run is `--optimizer SGD`, not the reverse.

### Identity-neutral values

```python
IDENTITY_NEUTRAL_DEFAULTS = {"optimizer": (None,)}
```

A neutral value is **omitted** from the hashed payload rather than serialised as
`null`. That is what allows a field to be added to `RUN_ID_FIELDS` at all:
`compute_run_id` serialises every listed field, absent ones included, so
appending a field the naive way changes every existing hash.

---

## How to verify

Print the identity of any planned run without executing it:

```bash
python scripts/03_run_cv.py --arms yolo26n --repeat 0 --fold 0 --explain-run-id
```

It prints the fields that entered the hash, the resulting `run_id`, and — the
part that matters when something is skipped unexpectedly — which fields were
omitted as identity-neutral and why.

Two tests enforce all of this:

- `tests/test_run_id_identity.py` — the past cannot move, including a case for
  each MuSGD arm, and a deliberate override still gets its own identity.
- `tests/test_registry_run_ids_intact.py` — **every** record in the real
  registry recomputes to its own stored `run_id`. Not one example: all 234.

---

## Known reference values

`03_run_cv`, repeat 0, fold 0:

| arm | `run_id` |
| --- | --- |
| `yolo26n` | `e3521ef16ba949e1` |
| `yolo26s` | `1204eabcf71b26f7` |
| `yolo26m` | `15d862a09c87609a` |
| `mobilenetv3_small` | `b17a48bad33d504a` |
| `resnet18` | `015b27bc5a45f8c9` |

If any of these changes, something has altered the identity scheme and every
completed run is about to be proposed for a re-run.
