# Literature Search Notes — 2026-04-04

Systematic search across 7 campaigns for GRACE paper positioning.

---

## Campaign 1: Foundation Model Distillation in CL Settings

**No paper distills SAM/MedSAM in a continual learning loop. GRACE's combination is novel.**

| # | Title | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| 1 | Task-Specific KD from Vision Foundation Model for Medical Segmentation (Liang et al.) | 2025 | arXiv / KBS | VFM+LoRA distilled to lightweight student for medical seg. Single-task only. |
| 2 | From Generalist to Specialist: Distilling a Mixture of Foundation Models for Medical Segmentation | 2025 | MICCAI 2025 | Ensemble of FMs distilled to specialist. No CL. |

## Campaign 2: Teacher-Student Dynamics in CL

**"Adapt Your Teacher" (WACV 2024) is the closest competitor. SATCH is also relevant.**

| # | Title | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| 3 | **Adapt Your Teacher: Improving KD for Exemplar-Free CL** (Szatkowski et al.) | 2024 | WACV 2024 | **HIGHLY RELEVANT.** Observes frozen teacher BN stats diverge under task shift. Proposes Teacher Adaptation (TA) to update BN. GRACE should cite and differentiate: their teacher is a snapshot; GRACE's is external FM. |
| 4 | SATCH: Specialized Assistant Teacher to Reduce Catastrophic Forgetting | 2024 | OpenReview | Small auxiliary teacher for current task complements main self-distilled teacher. Addresses staleness. |
| 5 | Self-Distillation Enables Continual Learning (Shenfeld et al.) | 2026 | arXiv | Self-distillation for CL in LLMs. Conceptually relevant, different domain. |

## Campaign 3: Medical CL Segmentation (2024-2025)

| # | Title | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| 6 | **Continual Learning for Abdominal Multi-Organ and Tumor Segmentation** (Zhang et al.) | 2023 | MICCAI 2023 | **KEY BENCHMARK.** Class-specific heads + CLIP embeddings + pseudo-labeling. 25 organs + 6 tumors on abdominal CT. github.com/MrGiovanni/ContinualLearning |
| 7 | Multi-modality Multiorgan Segmentation Using CL with eHAT (Wu et al.) | 2025 | Medical Physics | HAT-based CL across CT+MRI modalities. 3-4 task scenarios. |
| 8 | Continual Learning in Medical Image Analysis: A Survey | 2024 | Computers in Biology and Medicine | Comprehensive survey. Useful for positioning GRACE in the taxonomy. |

## Campaign 4: Selective/Gated KD

| # | Title | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| 9 | Select and Distill: Selective Dual-Teacher KD for CL on VLMs (Yu et al.) | 2024 | ECCV 2024 | Dual teachers, feature-level selection. Not per-pixel gating. |
| 10 | **ReCo-KD: Region- and Context-Aware KD for 3D Medical Segmentation** (Lan et al.) | 2025 | arXiv | **CLOSEST TO CGAD.** Class-aware scale masks weight voxel-wise KD. BUT uses GT-label-based statistics, not a learned correctness gate. Key differentiation point. |
| 11 | Teaching-Assistant-in-the-Loop: KD from Imperfect Teachers (Zhou & Ai) | 2024 | ACL 2024 | Instance-level quality filtering for noisy teachers. NLP domain. |
| 12 | Attention-guided Feature Distillation for Semantic Segmentation (Mansourian et al.) | 2024 | arXiv | CBAM attention for feature KD. Implicit spatial weighting, not correctness-aware. |

## Campaign 5: Adapter Reliability/Composition

| # | Title | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| 13 | **O-LoRA: Orthogonal Subspace Learning for LM Continual Learning** (Wang et al.) | 2023 | EMNLP 2023 | **CLOSEST TO TRA.** Orthogonal low-rank subspaces per task. TRA extends with frozen shared core + per-task residuals. |
| 14 | AdapterFusion: Non-Destructive Task Composition (Pfeiffer et al.) | 2021 | EACL 2021 | Attention-based adapter fusion. TRA avoids separate fusion stage. |
| 15 | **CL-LoRA: Continual Low-Rank Adaptation** (He et al.) | 2025 | CVPR 2025 | Shared orthogonal down-projections + KD + gradient reassignment. Most contemporary comparison for TRA. |

## Campaign 6: Snapshot Degradation Over Sequences

| # | Title | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| 16 | **Progress & Compress: Scalable CL Framework** (Schwarz et al.) | 2018 | ICML 2018 | **PRIMARY CITATION for Proposition 2.** Explicitly argues single-model CL degrades over long sequences. Built dual-network solution. |
| 17 | In Defense of Learning Without Forgetting for Task Incremental Learning (Oren & Wolf) | 2021 | ICCV 2021 Workshop | Counterpoint: LwF can be rescued with architecture choices. Cite for fairness. |

## Campaign 7: KD Theory Under Shift

| # | Title | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| 18 | **Does Knowledge Distillation Really Work?** (Stanton et al.) | 2021 | NeurIPS 2021 | KD fidelity gap exists even under i.i.d. Compounds under shift. Already ref [32]. |
| 19 | **Revisiting KD under Distribution Shift (ShiftKD)** (Zhang et al.) | 2023 | arXiv | **Directly supports teacher reliability gap.** No KD method consistently works under shift. Teacher robustness bounds student performance. |
| 20 | Gradient Episodic Memory for CL (Lopez-Paz & Ranzato) | 2017 | NeurIPS 2017 | Gradient alignment framework. Already ref [31]. |

---

## Novelty Assessment Summary

| Claim | Status | Key Differentiator |
|-------|--------|-------------------|
| Foundation model KD in continual segmentation | **Novel** — nobody does this | Papers 1-2 are single-task only |
| Teacher reliability gap as a named problem | **Novel** — "Adapt Your Teacher" observes it for BN stats but doesn't name/formalize it | Paper 3 closest, but different mechanism |
| Learned correctness gate (CGAD) | **Novel** — ReCo-KD does spatial weighting but from GT statistics, not learned gate | Paper 10 closest |
| Frozen core + per-task residuals (TRA) | **Incremental** — O-LoRA and CL-LoRA are close | Papers 13, 15 closely related; differentiate via frozen shared core |
| Snapshot degradation formalization (Proposition 2) | **Novel** — no formal proof exists; Progress & Compress motivates it empirically | Paper 16 empirical only |

## Papers to Add to Reference List

Priority additions (cite in Related Work):
- [33] Szatkowski et al. Adapt Your Teacher. WACV 2024.
- [34] Zhang et al. Continual Learning for Abdominal Multi-Organ and Tumor Segmentation. MICCAI 2023.
- [35] Schwarz et al. Progress & Compress. ICML 2018.
- [36] Wang et al. O-LoRA. EMNLP 2023.
- [37] He et al. CL-LoRA. CVPR 2025.
- [38] Lan et al. ReCo-KD. arXiv 2025.
- [39] Zhang et al. Revisiting KD under Distribution Shift. arXiv 2023.

Optional additions:
- [40] Pfeiffer et al. AdapterFusion. EACL 2021.
- [41] Yu et al. Select and Distill. ECCV 2024.
- [42] Oren & Wolf. In Defense of LwF. ICCV 2021 Workshop.
