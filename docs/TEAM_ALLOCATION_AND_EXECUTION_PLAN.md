# Team Allocation and Execution Plan

## Mission
Build a publishable continual distillation framework for 3D CT/MRI segmentation with reproducible engineering and strong clinical validity.

---

## Team Roles

## 1) Research Lead (PhD #1) — Method Core
**Owns**
- Distillation method design (logit/feature/uncertainty/boundary)
- Mathematical formulation and novelty framing

**Deliverables**
- method specification note
- ablation design plan
- method section draft

## 2) Continual Learning Lead (PhD #2) — Forgetting & Stability
**Owns**
- replay policy design
- regularization (EWC/SI variants)
- forgetting metrics and order sensitivity

**Deliverables**
- continual protocol and evaluation scripts
- forgetting/BWT/FWT analysis plots
- robustness appendix draft

## 3) Medical Data Lead (PhD #3) — Dataset Integrity
**Owns**
- dataset adapter validity
- split governance and leakage prevention
- task label harmonization

**Deliverables**
- frozen split manifests
- data protocol docs
- clinical failure taxonomy

## 4) ML Engineer #1 — Training Infrastructure
**Owns**
- training/eval pipeline reliability
- checkpointing and run automation
- method config integration

**Deliverables**
- stable baseline suite runner
- deterministic run manifests
- training reliability checklist

## 5) ML Engineer #2 — Evaluation and Packaging
**Owns**
- segmentation metrics implementation
- result aggregation and packaging
- figure/table export pipeline

**Deliverables**
- metric integrity report
- result bundle utility outputs
- publication-ready summary CSVs

## 6) MLOps/Systems Engineer (part-time)
**Owns**
- environment reproducibility
- compute/storage scheduling
- dependency and path health checks

**Deliverables**
- bootstrap/doctor scripts
- runtime resource dashboard notes
- reproducibility environment lock notes

---

## Workstream Breakdown (Parallel)

## Stream A — Method
- KD baseline
- KD + feature distillation
- KD + replay
- KD + replay + uncertainty/boundary weighting

## Stream B — Data
- TotalSegmentator integration and validation
- BraTS21 adapter and split lock
- ACDC adapter and split lock

## Stream C — Evidence
- baseline matrix
- ablation matrix
- statistical testing and confidence reporting
- failure analysis

## Stream D — Manuscript
- living draft for intro/method/results while experiments run

---

## Two-Week Sprint Plan

## Week 1 (Execution readiness)
- baseline suite fully runnable
- real-data sanity runs for TotalSegmentator
- split manifests frozen and validated
- first result bundle generated

## Week 2 (Paper evidence)
- KD variants + replay comparisons complete
- initial ablation table and core figures
- failure analysis panel v1
- manuscript draft v0.1 (intro+method+results skeleton)

---

## Governance Rules

1. No run without committed config + split manifest.
2. Every reported result must include commit hash + run directory.
3. Claim-to-evidence mapping is mandatory before writing conclusions.
4. Weekly review kills weak ideas quickly; preserve only evidence-backed directions.

---

## Ownership Matrix (Template)

- Method core owner: [name]
- Continual evaluation owner: [name]
- Dataset governance owner: [name]
- Training infra owner: [name]
- Evaluation packaging owner: [name]
- Systems/MLOps owner: [name]

---

## Checkpoint Cadence

- Daily: short execution update (completed / blocked / next)
- Twice weekly: experiment review and reprioritization
- Weekly: paper evidence review (figures + tables + claims)

---

## Ready-to-Execute Checklist

- [ ] split protocol frozen
- [ ] baseline suite stable
- [ ] metric pipeline validated
- [ ] first ablation matrix complete
- [ ] draft figures generated
- [ ] draft manuscript v0.1 started
