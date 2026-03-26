# continual-medseg-distill

Research scaffold for **continual learning + distillation** in **3D medical segmentation**.

This repo is intentionally lightweight and executable. It includes method skeletons for:
- sequential fine-tune
- replay
- distillation
- distillation + replay + EWC (skeleton)

> Status: scaffold for fast iteration. Many components are marked with `TODO` for full research implementation.

## Project layout

```text
continual-medseg-distill/
  configs/
    base.yaml
    tasks/ct_mri_sequence.yaml
    methods/{finetune,replay,distill,distill_replay_ewc}.yaml
    datasets/open_source_examples.yaml
  docs/
    experiment_protocol.md
    dataset_notes.md
  scripts/
    train.py
    eval.py
    prepare_data.py
    run_ablations.py
    package_results.py
  src/
    data/
    models/
    engine/
    methods/
    utils/
```

## Environment preflight

```bash
python scripts/doctor.py --dataset-root /media/user/data2/data2/data
```

## Setup

### Option A: one-shot bootstrap

```bash
bash scripts/bootstrap_env.sh
```

### Option B: editable install

```bash
cd continual-medseg-distill
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### Option B: direct deps

```bash
pip install torch monai nibabel pyyaml tqdm numpy pandas scipy
```

## Quickstart (synthetic, end-to-end)

Run train + eval using synthetic 3D data (no dataset download required):

```bash
python scripts/train.py --config configs/base.yaml --dry-run
python scripts/eval.py --config configs/base.yaml
```

Training writes outputs to `output.dir` (default: `outputs/continual_medseg_scaffold`):

- `metrics.csv` — epoch-wise `train_loss` and validation metrics (includes `dice_mean`, per-class Dice, and HD95 when available)
- `checkpoints/last.pt` — latest epoch checkpoint
- `checkpoints/best.pt` — best checkpoint by `output.best_metric` (default `voxel_acc`)

Print resolved config:

```bash
python scripts/train.py --config configs/base.yaml --dry-run --print-config
```

## Switching methods

Edit `configs/base.yaml` include path:

```yaml
includes:
  method: configs/methods/replay.yaml
```

or use one of:
- `configs/methods/finetune.yaml`
- `configs/methods/replay.yaml`
- `configs/methods/distill.yaml`
- `configs/methods/distill_replay_ewc.yaml`

## Dataset adapter examples (local-path only)

BraTS21 example:

```bash
python scripts/train.py \
  --config configs/base.yaml \
  --dataset-config configs/datasets/brats21_example.yaml \
  --dry-run
```

ACDC example:

```bash
python scripts/train.py \
  --config configs/base.yaml \
  --dataset-config configs/datasets/acdc_example.yaml \
  --dry-run
```

Notes:
- Update `root` and split IDs/manifest paths to your local extracted datasets.
- No dataset download logic is included in this repo.

## Data preparation

`prepare_data.py` does **not** download restricted datasets. It only creates a placeholder metadata JSON from a local dataset path:

```bash
python scripts/prepare_data.py \
  --dataset msd_task_placeholder \
  --input-dir /path/to/local/extracted_dataset \
  --output-json data/splits/msd_train_val.json \
  --dry-run
```

## Progress logging automation

Use the helper script to append standardized checkpoint entries to `docs/PROGRESS.md`.

Example:

```bash
python scripts/update_progress.py \
  --scope "Added KD loss weighting config and trainer wiring" \
  --impact "Makes distillation experiments reproducible across runs" \
  --next "Run ablation for alpha in {0.5, 1.0, 2.0}"
```

Entry template fields:
- `date` (defaults to today)
- `commit` (defaults to current git short SHA)
- `scope`
- `impact`
- `next`

## Packaging ablation results (paper-friendly bundle)

After running `scripts/run_ablations.py`, package one run directory into a compact bundle:

```bash
python scripts/package_results.py \
  outputs/ablations/ablation_run_YYYYMMDD_HHMMSS
```

Optional custom output path:

```bash
python scripts/package_results.py \
  outputs/ablations/ablation_run_YYYYMMDD_HHMMSS \
  --output-dir outputs/ablations/ablation_run_YYYYMMDD_HHMMSS/paper_bundle
```

Bundle outputs include:
- `summary.md` — human-readable ranking table
- `method_summary.csv` — merged metrics + ranking fields
- `checkpoint_refs.csv` + `checkpoint_refs/*.txt` — references to key checkpoints (`best.pt`, `last.pt`)
- `aggregate_metrics.csv` copy (when present)

Ranking logic:
- Methods are ranked by `dice_mean` (or nearest dice-like field found)
- Forgetting ranking is included when a forgetting-like metric exists
- Missing fields are explicitly called out in `summary.md`

## Next implementation targets

1. Real dataset adapters + MONAI transforms in `src/data/registry.py`
2. Replay buffer strategy and sampler in `src/methods/replay.py`
3. Teacher snapshot + KD losses in `src/methods/distill.py`
4. Fisher estimation + EWC penalty in `src/methods/distill_replay_ewc.py`
5. Metric/reporting polish (confidence intervals, cohort-level summaries) + checkpointing strategy tuning
