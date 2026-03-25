# Detailed Dataset Catalog (Workstation)

Source root:
`/media/user/data2/data2/data/`

This file stores the detailed, researcher-ready inventory shared during project planning.

---

## 1) AAPM_CT (~213 GB)

AAPM Low-Dose CT Grand Challenge — CT dose reduction / reconstruction.

- Patients: 10 training + 20 testing + 1 ACR phantom (31 total)
- File formats: `.IMA` image data, `.DCM` projection data
- Total files: 229,701 (DCM + IMA + ZIP)
- Reconstruction variants: 1mm/3mm × B30/D45
- Dose levels: FD (full dose), QD (quarter dose)
- Naming style example: `L632_QD_3_1.CT....IMA`
- Metadata/docs: minimal text metadata, no full README

Notes for this project:
- Primary use is reconstruction/denoising, not direct segmentation baseline.
- Could be used as auxiliary pretraining or data quality stress-test.

---

## 2) Brats21 (~7 GB images + labels)

BraTS 2021 Brain Tumor Segmentation Challenge.

- Patients: 1,152 training + test cohorts (labeled/unlabeled subsets)
- Modalities: T1, T1ce (+ FLAIR, T2 in extended dirs)
- File format: `.nii.gz`
- Typical image size: `155 × 240 × 240`, ~1mm isotropic
- Labels: binary and 4-class variants available
- Metadata: `dataset.json`, split files, modality stats CSVs
- License: CC-BY-SA 4.0

Notes for this project:
- Strong MRI continual task candidate.
- Good for CT→MRI shift analysis in forgetting studies.

---

## 3) FastMRI (~1.8 TB total)

NYU fastMRI Brain Multi-coil dataset.

- Very large-scale MRI reconstruction resource
- Multiple train/val/test batches in compressed archives
- Primarily k-space reconstruction benchmark

Notes for this project:
- Not primary for current 3D segmentation focus.
- Potentially useful for representation pretraining in future work.

---

## To be continued

Additional dataset sections will be appended as detailed catalog messages continue (IXI, NLST, PLCO, TotalSegmentator, VinDr-CXR, SwinUNETR resources, etc.).
