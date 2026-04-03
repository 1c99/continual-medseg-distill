# Draft Manuscript Skeleton

## Working Title
Continual Distillation with Orthogonal LoRA for 3D Medical Segmentation under Task Shift

## Abstract (Draft v0)
Continual learning for medical image segmentation is challenged by catastrophic forgetting, especially when models are sequentially adapted across anatomy-focused tasks. We present a practical continual distillation framework for 3D segmentation that combines teacher-guided knowledge transfer, replay-based retention, and orthogonality-regularized low-rank adaptation. Our pipeline is designed for reproducible real-world experimentation with explicit dataset governance, split manifests, artifact packaging, and run-level provenance. We instantiate the framework on a Task A→B protocol within TotalSegmentator (organs to muscles), with optional extension to cross-modality transfer settings. The method stack supports multiple distillation modes (logit, feature, weighted, boundary-aware), resumable multi-task training, and robust forgetting analytics (BWT/FWT). We show that infrastructure-level rigor enables reliable comparison of retention-performance trade-offs and establishes a strong basis for publication-grade continual segmentation studies.

## 1. Introduction (Draft bullets)
- Continual segmentation is clinically relevant because data/task expansion is sequential, not static.
- Naive fine-tuning causes forgetting of previous anatomical targets.
- Distillation and replay are promising, but practical integration with reproducibility and robust analysis is often under-specified.
- We propose a reproducible, configurable continual distillation framework with Orthogonal LoRA support.

### Contributions (Draft)
1. A modular continual distillation framework for 3D medical segmentation with teacher backends and multi-mode KD.
2. A task-sequential training engine with resumability and explicit forgetting analytics.
3. Orthogonal LoRA integration for reduced interference during continual adaptation.
4. A reproducible experimentation stack (validation, fairness checks, packaging, and traceable manifests).

## 2. Related Work (To expand)
- Continual learning in medical imaging segmentation
- Distillation in segmentation (2D/3D teacher-student paradigms)
- Parameter-efficient adaptation and LoRA variants
- Orthogonality-based mitigation of task interference

## 3. Method
### 3.1 Problem Definition
Given sequential tasks {T1, T2, ..., Tk}, optimize new-task learning while preserving prior-task performance.

### 3.2 Overall Objective
\[
\mathcal{L}_{total} = \mathcal{L}_{seg} + \lambda_{kd}\mathcal{L}_{kd} + \lambda_{rep}\mathcal{L}_{replay} + \lambda_{ewc}\mathcal{L}_{ewc} + \lambda_{ortho}\mathcal{L}_{ortho}
\]

### 3.3 Distillation Modes
- Logit KD
- Feature KD
- Weighted KD (uncertainty-aware)
- Boundary-aware KD

### 3.4 Orthogonal LoRA (concept)
Constrain adapter updates to minimize overlap with prior-task adaptation subspace.

### 3.5 Multi-task Training + Resume
- task progression state
- checkpoint + method-state persistence
- per-task evaluation matrix

## 4. Experimental Protocol
### 4.1 Task Design
- Task A: TotalSegmentator organs subset
- Task B: TotalSegmentator muscles subset

### 4.2 Baseline Conditions
- C1: Finetune
- C2: Distill baseline
- C3: Replay
- C4: Distill + Replay + EWC
- C5: Distill + Replay + EWC + Orthogonal LoRA

### 4.3 Metrics
- Dice (mean/per-class)
- HD95
- Forgetting, BWT, FWT
- compute/reporting metrics (throughput, VRAM)

### 4.4 Statistical Plan (placeholder)
- multi-seed comparisons
- confidence intervals and paired tests

## 5. Results (structure)
- A→B retention-performance trade-off
- effect of orthogonal LoRA
- distillation mode ablation
- failure-case analysis

## 6. Discussion
- Why certain methods fail under short schedules
- Data integrity and split governance impact
- deployment and reproducibility implications

## 7. Limitations
- dependency on teacher checkpoint availability
- sensitivity to task definition and class grouping
- runtime cost for full multi-seed matrix

## 8. Conclusion
A robust continual distillation stack can transform unstable sequential segmentation experiments into reproducible evidence pipelines suitable for publication-grade analysis.

---

## Immediate writing TODOs
- [ ] Expand Related Work with citations
- [ ] Add exact equations per KD mode
- [ ] Fill dataset details table from finalized manifests
- [ ] Insert real results once Phase execution is complete
