# Data Description (Workstation Inventory + Recommended Subset)

This document consolidates the current workstation datasets at:
`/media/user/data2/data2/data/`
and proposes a practical subset for continual 3D CT/MRI segmentation experiments.

---

## 1) Current Workstation Dataset Inventory

### AAPM_CT
AAPM Low-Dose CT Grand Challenge (CT denoising / reconstruction)
- `Training_Image_Data/` — 4 subdirs (1mm B30, 1mm D45, 3mm B30, 3mm D45)
- `Training_Projection_Data/` — raw projection (sinogram)
- `Testing_Image_Data/` — 4 matching subdirs
- `Testing_Projection_Data/`
- `ACR_Phantom_Data/`

### Brats21
BraTS 2021 brain tumor segmentation (multi-modal MRI, NIfTI)
- `imagesTr/`, `labelsTr/`
- `imagesTs/`, `labelsTs/`
- metadata files: `dataset.json`, `splits_final.pkl`, `stats.xlsx`, modality CSVs

### FastMRI
NYU fastMRI brain dataset (k-space reconstruction; very large)
- `brain_multicoil_train_batch_0-9`
- `brain_multicoil_val_batch_0-2`
- `brain_multicoil_test_batch_0-2`
- `brain_multicoil_test_full_batch_0-2`
- `brain_fastMRI_DICOM.tar.gz`

### IXI_Brain_MRI
IXI healthy brain MRI dataset
- `IXI-T1/`, `IXI-MRA/`, `IXI-DTI/`
- `wm/` white matter maps
- `IXI.xls`

### nlst
National Lung Screening Trial (lung CT screening)
- `csv/` metadata
- `nii/` converted NIfTI + inference output dirs
- `inference/`

### PLCO
PLCO chest X-rays (TIF)
- `batch_1a-1f`, `batch_2a-2f` (total ~91,719 images)
- package manifest (2023)

### Totalsegmentator_dataset
TotalSegmentator whole-body CT segmentation
- ~1,204 subjects (`s0000...`)
- per subject:
  - `ct.nii.gz`
  - `segmentations/` (multi-organ masks)
- `meta.csv`

### Vin_CXR
VinDr-CXR chest radiographs
- `train/`, `test/`
- `annotations/` CSV files

### SwinUnetR
Cardiac segmentation resources
- `ACDC/` (150 patients)
- `MICCAI_2015/` challenge data

---

## 2) Recommended Datasets for THIS Project

Project goal: continual distillation for **3D CT/MRI segmentation** with forgetting analysis.

### Primary (strongly recommended)
1. **TotalSegmentator_dataset** (CT, multi-organ, 3D)
   - Excellent for large-scale teacher signal and organ diversity.
2. **Brats21** (MRI, brain tumor, 3D)
   - Strong modality-shift task for continual learning (CT -> MRI).
3. **ACDC** from SwinUnetR folder (MRI, cardiac, 3D)
   - Good additional MRI task for sequential adaptation.

### Optional / secondary
4. **nlst (NIfTI subset)**
   - Use only if segmentation labels are available/derivable for selected tasks.

### Not primary for current scope
- **AAPM_CT**: reconstruction/denoising (not direct segmentation benchmark)
- **FastMRI**: reconstruction-focused, very heavy storage/compute
- **PLCO / Vin_CXR**: 2D X-ray detection/classification (outside current 3D segmentation core)
- **IXI**: useful for pretraining/auxiliary analysis, but not main labeled segmentation benchmark by default

---

## 3) Proposed Continual Task Sequence (Practical)

Suggested stream:
1. **Task A (CT):** TotalSegmentator multi-organ subset
2. **Task B (CT):** TotalSegmentator small-structure subset (harder granularity)
3. **Task C (MRI):** BraTS21 tumor segmentation
4. **Task D (MRI):** ACDC cardiac segmentation

Why this sequence:
- starts with robust CT representation,
- introduces intra-modality transfer first,
- then tests cross-modality adaptation (CT -> MRI),
- enables strong forgetting analysis across clinically different domains.

---

## 4) Data Governance / Reproducibility Notes

- enforce patient-level split across all tasks
- avoid leakage between tasks and validation/test sets
- keep a frozen split manifest in repo (CSV/JSON)
- record preprocessing versions (resampling, intensity normalization, label mapping)
- track per-task class definitions explicitly

---

## 5) Next Integration Steps in Repo

1. Add dataset adapters in `src/data/registry.py` for:
   - TotalSegmentator
   - BraTS21
   - ACDC
2. Add task YAMLs under `configs/tasks/` with exact path + label maps.
3. Add `docs/DATA_SPLIT_PROTOCOL.md` with frozen split policy.
4. Add simple data sanity script:
   - shape checks
   - label-value checks
   - missing-file audit

---

## 6) Caveat

Before publication use, confirm each dataset's licensing/terms and citation requirements.
Some challenge datasets may require registration and specific usage constraints.
