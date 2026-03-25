# Detailed Dataset Catalog (Workstation)

Source root:
`/media/user/data2/data2/data/`

This file consolidates the detailed workstation inventory shared during project planning, plus project-fit notes.

---

## 1) AAPM_CT (~213 GB)

AAPM Low-Dose CT Grand Challenge (dose reduction / reconstruction).

- Patients: 10 training + 20 testing + 1 ACR phantom (31 total)
- Formats: `.IMA` image data, `.DCM` projection data
- Total files: ~229,701 (DCM + IMA + ZIP)
- Recon variants: 1mm/3mm × B30/D45
- Dose levels: FD (full dose), QD (quarter dose)
- Docs: minimal text metadata, no full README

Project fit:
- Not primary for segmentation benchmarking; useful for reconstruction or auxiliary denoising pretext.

---

## 2) BraTS21 (~7 GB images + labels)

BraTS 2021 brain tumor segmentation challenge.

- Subjects: ~1,152 train (+ test cohorts)
- Modalities: T1, T1ce (+ FLAIR/T2 in extended folders)
- Format: `.nii.gz`
- Typical dimensions: `155 × 240 × 240`, ~1mm isotropic
- Labels: binary and 4-class variants available
- Metadata: `dataset.json`, split files, modality stats
- License: CC-BY-SA 4.0

Project fit:
- Strong MRI continual-learning task; excellent for CT→MRI shift analysis.

---

## 3) FastMRI (~1.8 TB total)

NYU fastMRI brain multi-coil dataset (reconstruction-focused).

- Format: HDF5 (`.h5`)
- Sequence naming includes AXT1/AXT2/AXFLAIR/AXT1PRE/AXT1POST variants
- Training: 10 large compressed batches
- Validation: 3 batches (partial extraction)
- Test: standard + full-resolution batches (partially extracted)
- Extracted footprint: ~433 GB, remainder still compressed
- Documentation: minimal

Project fit:
- Not primary for current segmentation objective; potential representation pretraining source.

---

## 4) IXI_Brain_MRI (~29 GB)

IXI healthy brain MRI collection.

- Subjects: ~581
- `IXI-T1`: 581 `.nii` volumes
- `IXI-MRA`: 570 `.nii.gz`
- `IXI-DTI`: thousands of DTI components/volumes
- `wm/`: white matter maps
- Metadata: `IXI.xls`

Project fit:
- Good auxiliary MRI domain source; not a direct primary labeled segmentation benchmark unless task-specific labels are defined.

---

## 5) NLST (~4.2 TB)

National Lung Screening Trial CT data ecosystem.

- Cohorts: `nlst_1` to `nlst_15`
- NIfTI files: very large-scale archive
- Metadata CSVs: cohort-level DICOM path mapping
- Inference directories: multiple precomputed prediction outputs
- Includes filtered subsets and task-specific derived outputs

Project fit:
- High-potential CT source if labels/task definitions are standardized for this project.
- Strong candidate for downstream external validation once labeling protocol is fixed.

---

## 6) PLCO (~540+ GB)

PLCO chest X-ray data + rich clinical metadata.

- Images: ~91,719 TIFF files across 12 batches
- Format: `.tif` (large per-image size)
- Rich linked tabular metadata package (screening/procedure/treatment records)

Project fit:
- Mostly 2D X-ray; outside the core 3D CT/MRI segmentation scope.
- Useful for future multimodal or weak-supervision side projects.

---

## 7) TotalSegmentator_dataset

Whole-body CT segmentation corpus.

- Subjects: ~1,204
- Per-subject CT: `ct.nii.gz`
- Per-organ masks under `segmentations/`
- Metadata: `meta.csv`

Project fit:
- **Primary dataset** for our current continual segmentation experiments.
- Best source for multi-organ CT teacher/student distillation stage.

---

## 8) VinDr-CXR

VinDr-CXR chest radiograph detection dataset.

- Train/test split with annotation CSVs
- Primarily 2D X-ray detection/classification scope

Project fit:
- Not primary for current 3D segmentation work.

---

## 9) SwinUNETR resources (ACDC + MICCAI 2015)

Cardiac segmentation resources.

- ACDC (cardiac MRI) includes patient-wise image/label structure
- Additional MICCAI 2015 challenge data

Project fit:
- Strong MRI cardiac task for later continual stages after CT tasks.

---

## Recommended Core Subset for This Project

1. **TotalSegmentator** (CT, multi-organ, 3D)
2. **BraTS21** (MRI brain tumor, 3D)
3. **ACDC** (MRI cardiac, 3D)
4. Optional later: **NLST** (if segmentation protocol is clearly defined)

---

## Proposed Continual Sequence

- Task A: TotalSegmentator (CT multi-organ)
- Task B: TotalSegmentator hard subset / small structures
- Task C: BraTS21 (MRI brain)
- Task D: ACDC (MRI cardiac)

This sequence gives strong modality-shift and forgetting analysis potential.

---

## Reproducibility + Governance Notes

- enforce patient-level splits
- freeze split manifest in repo
- maintain exact preprocessing versioning
- perform leakage checks between tasks/splits
- verify license/registration terms before publication use

---

## Action Items in Repo

- [ ] Add robust TotalSegmentator split-manifest ingestion to production path
- [ ] Add BraTS21 adapter
- [ ] Add ACDC adapter
- [ ] Add data sanity scripts (shape/label/value checks)
- [ ] Add DATA_SPLIT_PROTOCOL.md with frozen split policy
