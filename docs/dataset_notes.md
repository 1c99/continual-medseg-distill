# Dataset Notes (Open-source candidates)

This project supports **config-driven dataset registration**. Actual loaders are TODO.

## Candidate datasets

- **BTCV (Beyond the Cranial Vault)**
  - Task: multi-organ abdominal CT segmentation
  - Access: often requires registration/data use agreement depending on host mirror
  - Caveat: verify redistribution constraints

- **KiTS (Kidney Tumor Segmentation; e.g., KiTS19/KiTS23)**
  - Task: kidney/tumor CT segmentation
  - Access: open challenge dataset, may require registration/terms acceptance
  - Caveat: check version-specific license and citation requirements

- **MSD (Medical Segmentation Decathlon)**
  - Task: multiple segmentation tasks across modalities
  - Access: publicly available, task-wise downloads
  - Caveat: each task may have distinct preprocessing and label conventions

- **AMOS**
  - Task: multi-organ segmentation (CT/MRI)
  - Access: challenge dataset, registration may apply
  - Caveat: confirm non-commercial/research-only clauses

- **CHAOS**
  - Task: abdominal organ segmentation (CT/MRI)
  - Access: challenge-style distribution, often with registration
  - Caveat: verify usage terms before publication/commercial usage

- **BraTS**
  - Task: brain tumor MRI segmentation
  - Access: registration is typically required
  - Caveat: strict citation + data-use policy

## Important policy

- This repository **does not include private download scripts**.
- Use official dataset portals and comply with license/DUA requirements.
- Store local paths in config files only (e.g., `configs/datasets/open_source_examples.yaml`).

## TotalSegmentator scaffold split manifests

The TotalSegmentator scaffold loader supports split manifests for subject IDs:

- Config key: `data.totalseg.split_manifest`
- Manifest file types: `.json`, `.yaml`, `.yml`
- Split keys: `train`/`val` (preferred), `train_ids`/`val_ids` (compatibility)

A minimal example is included at:
- `data/splits/example_totalseg_split.json`

If `split_manifest` is not provided, the loader still supports explicit config lists:
- `data.totalseg.train_ids`
- `data.totalseg.val_ids`

## TODO

- [ ] Add dataset-specific parsing adapters in `src/data/registry.py`
- [ ] Add MONAI transforms per dataset/task
- [ ] Add harmonized spacing/intensity preprocessing pipelines
