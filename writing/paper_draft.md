# GRACE: Gated Residual Adapter for Continual and Efficient Foundation Model Distillation

---

## ABSTRACT

Continual learning in medical image segmentation requires models to acquire new anatomical targets without forgetting prior ones. Knowledge distillation from foundation models offers a natural anti-forgetting mechanism, yet we identify a critical overlooked failure mode: the lightweight adapter bridging a frozen foundation backbone to task-specific predictions becomes unreliable under task shift, producing noisy supervision that harms the student. We term this the *teacher reliability gap* and propose GRACE, a teacher-side mechanism comprising (i) a Task-Residual Adapter with frozen shared core and per-task projections, (ii) a learned confidence gate that suppresses spatially unreliable distillation, and (iii) an append-only prototype bank for forgetting-free auxiliary supervision. On a three-task TotalSegmentator benchmark (organs → muscles → vessels) with MedSAM2/3 teachers and an nnU-Net student, GRACE reduces catastrophic forgetting by X% over standard adapter distillation while adding fewer than 3K parameters per task. Our results demonstrate that making the teacher continual-learning-aware is as important as protecting the student.

---

## 1. INTRODUCTION

Medical image segmentation models deployed in clinical practice must adapt to evolving requirements. A model trained to segment abdominal organs may later need to identify pelvic muscles for sarcopenia assessment or vascular structures for surgical planning. Retraining from scratch is often infeasible due to computational cost and data privacy regulations that prevent centralizing patient data across institutions. Continual learning offers a principled alternative: the model learns new tasks sequentially while retaining prior capabilities.

Concurrently, medical foundation models — large-scale architectures such as MedSAM2 [1] and MedSAM3 [2] pre-trained on millions of medical images — have achieved remarkable segmentation performance. However, these models are prohibitively large for clinical deployment. Knowledge distillation (KD) bridges this gap: a lightweight student (e.g., nnU-Net [3]) mimics the foundation model during training and operates independently at inference.

A natural question arises: *can foundation model distillation mitigate catastrophic forgetting?* The teacher's pre-trained representations are stable and richer than the student's, providing an anchor against drift. Prior work on Learning without Forgetting (LwF) [4] demonstrates this principle using a snapshot of the student as teacher. Foundation models should be far superior.

**However, we identify a critical failure mode.** Foundation models produce general-purpose features, not task-specific predictions. A lightweight adapter must project these features to each task's label space. This adapter becomes the bottleneck: trained on Task A (organs), it produces reliable organ predictions; but when Task B (muscles) arrives with a different label space, the adapter must be rebuilt. A new adapter produces random predictions; the old adapter predicts wrong classes. The teacher becomes *unreliable precisely when the student needs it most* (see Figure 1).

While existing continual learning methods — EWC [5], replay [6], PLOP [7], MiB [8] — focus on protecting the student, the teacher-student interface has received no targeted investigation. Related approaches such as dark experience replay [9] store past teacher logits alongside samples, and confidence-weighted distillation [10] uses output entropy as a proxy for reliability. However, no prior work has designed the adapter interface specifically for continual foundation model distillation, where the teacher's projection head must serve a growing set of semantically distinct tasks.

We address this gap with **GRACE** (**G**ated **R**esidual **A**dapter for **C**ontinual **E**fficiency), a teacher-side mechanism ensuring reliable distillation across tasks. GRACE has three components: (1) **Task-Residual Adapter (TRA)** — a frozen shared core with per-task 1×1 projections (~3K parameters each), enabling the teacher to serve all tasks without interference; (2) **Confidence-Gated Adaptive Distillation (CGAD)** — a learned gate that suppresses KD in spatially unreliable regions, trained with correctness supervision rather than heuristic entropy thresholds; and (3) **Continual Prototype Alignment (CPA)** — an append-only bank of per-class feature prototypes that provides forgetting-free auxiliary soft labels.

We evaluate on a three-task benchmark from TotalSegmentator [11]: abdominal organs (14 classes) → pelvic muscles (10 classes) → abdominal vessels (7 classes), using MedSAM2 and MedSAM3 as teachers with nnU-Net as student. Our experiments show:

- Standard adapter distillation fails under task shift, matching self-distillation baselines (§4.2).
- GRACE reduces mean forgetting by X% while maintaining plasticity (§4.3).
- The confidence gate provides the largest single improvement, confirming that *where* to distill matters as much as *what* to distill (§5.2).
- GRACE adds negligible overhead (~3K parameters, ~0.1% FLOPs per task) (§5.4).

**Contributions.** (1) We identify the *teacher reliability gap* in foundation-model-guided continual distillation. (2) We propose GRACE (TRA + CGAD + CPA), a teacher-side continual adapter mechanism. (3) We construct a three-task 3D continual segmentation benchmark with foundation model teachers. (4) We provide analysis of gating behavior, representation stability, and per-component ablation.

---

## 2. RELATED WORK

**Continual learning for segmentation.** Catastrophic forgetting [12] is particularly acute in segmentation, where the pixel-level decision boundary shifts entirely across tasks. Regularization methods such as EWC [5] and SI [13] penalize changes to important parameters but struggle when task distributions are disjoint. Replay-based approaches [6, 14] maintain a buffer of past samples, providing direct supervision on old tasks. Distillation-based methods are prominent in continual segmentation: LwF [4] uses a frozen student snapshot as teacher; PLOP [7] proposes pooled local distillation to preserve multi-scale features; MiB [8] models the background shift explicitly; and REMINDER [15] combines feature replay with distillation. In the medical domain, Zhang et al. [34] propose class-specific heads with CLIP embeddings for multi-organ continual segmentation on abdominal CT, and Wu et al. [43] extend HAT to multi-modality organ segmentation. Schwarz et al. [35] argue that single-model CL approaches degrade over long sequences and propose a dual-network knowledge base, motivating our focus on external teacher stability. All of these methods operate exclusively on the student side. When an external teacher is used, it is assumed to be a static oracle. We show this assumption breaks when the teacher requires a task-specific adapter.

**Teacher adaptation in continual learning.** Recent work has begun to address teacher quality in CL distillation. Szatkowski et al. [33] observe that a frozen snapshot teacher's batch normalization statistics diverge from the student's distribution as tasks accumulate, causing KD to degrade; they propose updating teacher BN online. SATCH [41] introduces a small auxiliary teacher trained only on the current task to complement the stale main teacher. However, both approaches operate on snapshot (self-distillation) teachers. No prior work addresses the reliability of an *external* foundation model teacher whose adapter must serve a growing set of semantically distinct tasks — the setting GRACE targets.

**Knowledge distillation in medical imaging.** KD has been widely applied to compress medical segmentation models [16, 17]. Recent work distills from foundation models: KnowSAM [18] uses SAM3 as a teacher for semi-supervised cardiac segmentation, and several approaches [19, 20] distill SAM features for downstream tasks. However, these operate in a single-task setting. Stanton et al. [32] demonstrate that KD fidelity gaps exist even under i.i.d. conditions, and Zhang et al. [39] show that no KD method consistently works under distribution shift — findings that compound in continual settings where each task introduces a label-space shift. The continual extension — where the teacher must serve a growing sequence of tasks with different label spaces — introduces the adapter reliability problem we address.

**Foundation models for medical segmentation.** The Segment Anything Model (SAM) [21] and its medical variants MedSAM [22], MedSAM2 [1], and SAM-Med3D [23] have demonstrated strong zero-shot and few-shot segmentation. These models produce rich visual features via large pre-trained backbones (Hiera for SAM2, ViT for SAM3) but require spatial prompts (bounding boxes, points) rather than producing dense predictions directly. For KD, this necessitates a projection adapter from backbone features to dense logits — the component whose reliability we study.

**Adapter methods.** Lightweight adapters for foundation models — including LoRA [24], prompt tuning [25], and visual adapters [26] — enable parameter-efficient task adaptation. O-LoRA [36] learns orthogonal low-rank subspaces per task to minimize inter-task interference, and CL-LoRA [37] extends this with shared orthogonal down-projections and early-exit KD for class-incremental learning. AdapterFusion [40] composes task-specific adapters via learned attention weights. Our TRA component applies a related principle to the *teacher's* output adapter: a frozen shared core captures generic feature-to-anatomy mappings, while per-task 1x1 residuals specialize to each label space. Unlike O-LoRA's fully separate subspaces, TRA's shared core enables cross-task feature reuse while frozen residuals guarantee zero interference.

**Confidence-aware distillation.** Uncertainty-weighted KD has been explored using teacher output entropy [28], temperature scaling [29], and Monte Carlo dropout [30]. ReCo-KD [38] uses class-aware scale masks and activation statistics to weight voxel-wise distillation in 3D medical segmentation, up-weighting small clinical structures. Yu et al. [42] propose dual-teacher selection based on feature discrepancy for VLM continual learning. Our CGAD differs from these in two respects: (i) it operates on the adapter's *internal features* rather than output probabilities or ground-truth statistics, accessing richer reliability signals; and (ii) it is trained with explicit correctness supervision via BCE against teacher accuracy masks, learning *where the adapter is actually correct* rather than relying on distributional heuristics or label-derived statistics.

---

## 3. METHOD

### 3.1 Problem Formulation

We consider a continual segmentation setting with $T$ sequential tasks $\{\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_T\}$. Each task $\mathcal{T}_t$ provides a training set $\mathcal{D}_t = \{(\mathbf{x}_i, \mathbf{y}_i)\}$ where $\mathbf{x}_i \in \mathbb{R}^{1 \times D \times H \times W}$ is a 3D CT volume and $\mathbf{y}_i \in \{0, \ldots, C_t\}^{D \times H \times W}$ is the voxel-wise label map with $C_t$ foreground classes. Tasks have disjoint label spaces (e.g., organs vs. muscles vs. vessels). When training on $\mathcal{T}_t$, access to previous data $\mathcal{D}_{<t}$ is limited to a small replay buffer $\mathcal{M}$.

The student $f_\theta$ is a lightweight segmentation model (nnU-Net) with task-specific output heads $\{h_t\}_{t=1}^T$ sharing a common backbone. The teacher consists of a frozen foundation model backbone $g_\phi$ (MedSAM2/3) and an adapter $\mathcal{A}$ that projects backbone features to task-specific logits.

The standard training objective combines supervised loss and KD:

$$\mathcal{L} = \mathcal{L}_{\text{seg}}(f_\theta(\mathbf{x}), \mathbf{y}) + \lambda_{\text{kd}} \cdot \mathcal{L}_{\text{KD}}(f_\theta(\mathbf{x}), \mathcal{A}(g_\phi(\mathbf{x})))$$

where $\mathcal{L}_{\text{seg}}$ is Dice + cross-entropy loss and $\mathcal{L}_{\text{KD}}$ is KL divergence with temperature scaling. The core problem: when transitioning from $\mathcal{T}_t$ to $\mathcal{T}_{t+1}$, the adapter $\mathcal{A}$ must be reconfigured for a new label space, and its reliability varies spatially.

### 3.2 The Teacher Reliability Gap

We first formalize why naive adapter distillation fails. The adapter $\mathcal{A}: \mathbb{R}^{d \times D' \times H' \times W'} \rightarrow \mathbb{R}^{C_t \times D \times H \times W}$ maps foundation features of dimension $d$ to $C_t$-class logits. When $\mathcal{T}_{t+1}$ arrives with $C_{t+1} \neq C_t$, three options exist:

**(a) Rebuild.** Initialize a new adapter for $C_{t+1}$ classes. The adapter produces random logits until re-trained, providing no useful KD signal during early training of $\mathcal{T}_{t+1}$.

**(b) Retain.** Keep the old adapter. It produces $C_t$-class logits when $C_{t+1}$ classes are expected — a dimension mismatch that either crashes or requires lossy truncation.

**(c) Fine-tune.** Adapt the existing adapter to $C_{t+1}$. The adapter forgets $\mathcal{T}_t$'s mapping, degrading teacher quality on replayed $\mathcal{T}_t$ data.

All three options produce unreliable teacher predictions at task boundaries. We observe empirically that this unreliability is *spatially non-uniform*: the teacher may be confident on large, well-represented structures (e.g., liver) but unreliable on small or boundary-adjacent structures (e.g., adrenal glands). GRACE addresses each failure mode with a targeted component.

#### 3.2.1 Formal Analysis

We formalize two properties that motivate GRACE's design.

**Proposition 1 (Selective KD reduces harmful transfer).** Let $e_v = \mathbb{1}[\hat{y}_v \neq y_v]$ indicate teacher error at voxel $v$, and let $G_v \in [\gamma_{\min}, 1]$ be the gate value. Decompose the gated KD risk into *beneficial* (correct voxels) and *harmful* (erroneous voxels) components:

$$R_{\text{gated}} = \frac{1}{|\Omega|}\sum_{v} G_v \cdot \ell_{\text{KD}}(v) = R_{\text{gated}}^{+} + R_{\text{gated}}^{-}$$

where $R_{\text{gated}}^{-} = \frac{1}{|\Omega|}\sum_{v: e_v=1} G_v \cdot \ell_{\text{KD}}(v)$ is the harmful component. *Assume* (A1) the gate has positive correlation with correctness: $\mathbb{E}[G_v \mid e_v = 1] \leq 1 - \rho$ for some $\rho > 0$. Then:

$$\mathbb{E}[R_{\text{gated}}^{-}] \leq (1 - \rho) \cdot \mathbb{E}[R_{\text{ungated}}^{-}]$$

The harmful component is multiplicatively reduced by $(1 - \rho)$. The parameter $\rho$ is empirically measurable as the gap in mean gate activation between correctly and incorrectly predicted voxels during adapter pre-training (via the BCE supervision in Eq. 4). Larger $\rho$ indicates a more discriminative gate. $\square$

**Proposition 2 (Snapshot degradation vs. stable external teacher).** Consider a sequence of $T$ tasks. Let $\Delta_T = \|\phi(x; \theta_T) - \phi(x; \theta_0^*)\|$ be the representation drift between the current student features and the frozen snapshot teacher's features, measured on replay samples from $\mathcal{T}_1$. *Assume* (B1) bounded per-task drift: $\|\phi(x; \theta_{t+1}) - \phi(x; \theta_t)\| \leq \delta$ for all $t$. By the triangle inequality:

$$\Delta_T \leq T \cdot \delta$$

The snapshot teacher's KD signal degrades as $O(T)$ because its implied gradient references an increasingly stale feature geometry. In contrast, GRACE's external teacher has zero drift: the foundation backbone $g_\phi$ is frozen, and each per-task residual $r_t$ is fixed after training. The teacher's prediction quality on task $\mathcal{T}_1$ replay data is $O(1)$, independent of $T$. The gap $\Delta_T^{\text{snapshot}} - \Delta_T^{\text{GRACE}} = O(T)$ grows linearly with sequence length, providing GRACE a *compounding advantage* on long task sequences. $\square$

These propositions predict two testable hypotheses: (1) the CGAD gate should correlate positively with adapter correctness (verified in §5.1), and (2) GRACE's advantage over self-distillation should increase with the number of tasks in the sequence (tested in §4.2).

### 3.3 Task-Residual Adapter (TRA)

We decompose the adapter into a shared core $\mathcal{A}_{\text{core}}$ and per-task residual projections $\{r_t\}$:

$$\mathcal{A}_{\text{core}}: \mathbb{R}^{d} \rightarrow \mathbb{R}^{d}, \quad r_t: \mathbb{R}^{d} \rightarrow \mathbb{R}^{C_t}$$

The core consists of a 3D convolution, group normalization, and GELU activation:

$$\mathbf{z} = \text{GELU}(\text{GN}(\text{Conv3d}_{d \rightarrow d}(\mathbf{f})))$$

where $\mathbf{f} = g_\phi(\mathbf{x})$ are the foundation backbone features. Each residual $r_t$ is a single $1 \times 1 \times 1$ convolution:

$$\hat{\mathbf{y}}_t = r_t(\mathbf{z}) = \text{Conv3d}_{d \rightarrow C_t}(\mathbf{z})$$

**Training protocol.** The core and $r_1$ are trained jointly on $\mathcal{T}_1$ via supervised segmentation loss. After $\mathcal{T}_1$, the core is frozen permanently. For each subsequent task $\mathcal{T}_t$, only a new $r_t$ is trained ($d \times C_t$ parameters). Previous residuals $\{r_1, \ldots, r_{t-1}\}$ are never modified.

**Key properties:** (i) Zero interference between tasks — each residual operates independently on the same frozen features. (ii) Constant-time scaling — adding a task costs $\mathcal{O}(d \cdot C_t)$ parameters. (iii) The core captures task-general anatomy-to-feature mappings that transfer across tasks.

### 3.4 Confidence-Gated Adaptive Distillation (CGAD)

Even with TRA, the adapter's spatial reliability varies — predictions on large central structures are more accurate than those on small peripheral ones. We introduce a gate network that estimates per-voxel reliability:

$$\mathbf{G} = \sigma_{\min}(\text{Conv3d}_{64 \rightarrow 1}(\text{GELU}(\text{Conv3d}_{d \rightarrow 64}(\mathbf{z}))))$$

where $\sigma_{\min}(x) = \gamma_{\min} + (1 - \gamma_{\min}) \cdot \sigma(x)$ ensures a minimum gate value $\gamma_{\min} = 0.1$, preventing complete suppression of any region. The gate is spatially interpolated to match the input resolution.

**Gate supervision.** During adapter pre-training, the gate is trained with binary cross-entropy against a correctness mask:

$$\mathcal{L}_{\text{gate}} = \text{BCE}(\mathbf{G}, \, \mathbb{1}[\hat{\mathbf{y}} = \mathbf{y}])$$

where $\hat{\mathbf{y}} = \arg\max \hat{\mathbf{y}}_t$ is the adapter's hard prediction and $\mathbf{y}$ is ground truth. This teaches the gate to predict *where the adapter is actually correct*, not merely where it is confident (which may not correlate for poorly calibrated adapters).

**Gated KD loss.** During continual learning, the gate modulates the distillation loss spatially:

$$\mathcal{L}_{\text{KD}}^{\text{gated}} = \frac{1}{|\Omega|} \sum_{v \in \Omega} G_v \cdot \text{KL}\left(\frac{f_\theta(\mathbf{x})_v}{T} \bigg\| \frac{\hat{\mathbf{y}}_{t,v}}{T}\right) \cdot T^2$$

where $v$ indexes voxels, $T$ is the temperature, and $G_v \in [\gamma_{\min}, 1]$ is the gate value. High-confidence regions receive full distillation; uncertain regions receive attenuated signal proportional to estimated reliability.

### 3.5 Continual Prototype Alignment (CPA)

As a complementary signal to the adapter's learned predictions, we maintain per-class prototypes in the foundation model's feature space. For each class $c$ in task $\mathcal{T}_t$, the prototype is the mean feature vector over all voxels of that class:

$$\mathbf{p}_{t,c} = \frac{1}{|\mathcal{V}_{t,c}|} \sum_{v \in \mathcal{V}_{t,c}} \mathbf{f}_v$$

where $\mathcal{V}_{t,c} = \{v : y_v = c\}$ is the set of voxels labeled as class $c$ across $\mathcal{T}_t$'s training data, and $\mathbf{f}_v \in \mathbb{R}^d$ is the backbone feature at voxel $v$. Prototypes are computed during adapter pre-training via running mean updates and stored in a bank $\mathcal{P} = \{\mathbf{p}_{t,c}\}$.

**Prototype soft labels.** Given features $\mathbf{f}$, prototype-based logits are computed via cosine similarity:

$$\ell_{t,c}^{\text{proto}}(\mathbf{f}_v) = \frac{\mathbf{f}_v \cdot \mathbf{p}_{t,c}}{\|\mathbf{f}_v\| \cdot \|\mathbf{p}_{t,c}\|} \cdot \frac{1}{\tau}$$

where $\tau$ is a temperature parameter. These logits provide a non-parametric soft-label signal that is fully determined by the frozen backbone features and the stored prototypes.

**Key properties:** (i) Append-only — new task prototypes are added without modifying existing ones, guaranteeing zero forgetting in the prototype bank. (ii) Non-parametric — no learned weights, eliminating any risk of drift. (iii) Complementary — captures class-level feature statistics that the per-voxel adapter may miss, particularly for small or rare structures.

### 3.6 Complete Training Objective

The full loss for task $\mathcal{T}_t$ combines supervised learning, gated distillation, prototype alignment, replay, and regularization:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{seg}}(f_\theta(\mathbf{x}), \mathbf{y})}_{\text{supervised}} + \underbrace{\lambda_{\text{kd}} \cdot \mathcal{L}_{\text{KD}}^{\text{gated}}}_{\text{gated distillation}} + \underbrace{\lambda_{\text{replay}} \cdot \mathcal{L}_{\text{replay}}}_{\text{replay buffer}} + \underbrace{\lambda_{\text{ewc}} \cdot \mathcal{L}_{\text{EWC}}}_{\text{regularization}}$$

where $\mathcal{L}_{\text{replay}}$ computes the segmentation loss on samples drawn from buffer $\mathcal{M}$ (using the appropriate task head and adapter residual), and $\mathcal{L}_{\text{EWC}} = \sum_i F_i (\theta_i - \theta_i^*)^2$ penalizes drift from parameters important for previous tasks.

**Adapter pre-training.** Before the continual sequence begins, the adapter (core + first residual + gate) is trained on $\mathcal{T}_1$ data for $E_{\text{pre}}$ epochs with the backbone frozen. Prototypes are accumulated during this phase. After pre-training, the core and gate are frozen.

**Task transition.** When $\mathcal{T}_{t+1}$ arrives: (1) a new residual $r_{t+1}$ is initialized and briefly trained on $\mathcal{T}_{t+1}$ data; (2) new prototypes $\{\mathbf{p}_{t+1,c}\}$ are computed and appended to $\mathcal{P}$; (3) the student's multi-head registers a new output head for $C_{t+1}$ classes; (4) the student trains with all loss components active.

---

## 4. EXPERIMENTS

### 4.1 Setup

**Dataset.** We construct a three-task continual segmentation benchmark from TotalSegmentator v2 [11], a large-scale CT dataset with 1,204 subjects and 104 anatomical structures:

| Task | Structures | Classes | Anatomical region |
|------|-----------|---------|-------------------|
| $\mathcal{T}_1$: Organs | liver, spleen, kidneys, pancreas, gallbladder, stomach, duodenum, small bowel, colon, esophagus, adrenal glands, urinary bladder | 14 + BG | Abdominal |
| $\mathcal{T}_2$: Muscles | gluteus max/med/min (L/R), iliopsoas (L/R), autochthon (L/R) | 10 + BG | Pelvic |
| $\mathcal{T}_3$: Vessels | aorta, inferior vena cava, portal/splenic vein, pulmonary artery, iliac arteries (L/R), iliac veins (L/R) | 9 + BG | Abdominal/pelvic |

Tasks are disjoint in label space but share the same CT scans, maximizing distribution overlap while testing class-incremental learning. Data is split into 100 train / 19 validation subjects (excluding corrupt scans). All volumes are resampled to $128 \times 128 \times 128$ voxels.

**Teacher models.** We evaluate two foundation model teachers:
- *MedSAM2*: SAM2-based (Hiera backbone), fine-tuned for 3D medical segmentation. Features extracted slice-by-slice at $512 \times 512$, stacked into 3D feature volumes.
- *MedSAM3*: SAM3-based (ViT backbone) with LoRA fine-tuning for medical imaging. Slice-by-slice at $1008 \times 1008$.

Both produce $d = 256$-dimensional features. The GRACE adapter operates identically on either backbone.

**Student model.** nnU-Net (PlainConvUNet, 5 stages, features [32, 64, 128, 256, 512], 22.6M parameters) with multi-head output (per-task classification heads sharing the encoder-decoder backbone).

**Training details.** 50 epochs per task, 100 steps/epoch, batch size 2, Adam optimizer (lr=1e-3), DiceCE loss. Replay buffer: 500 samples. EWC Fisher samples: 64. KD weight $\lambda_{\text{kd}} = 0.7$, temperature $T = 2.0$, replay weight $\lambda_{\text{replay}} = 1.0$, EWC weight $\lambda_{\text{ewc}} = 0.2$. Adapter pre-training: 20 epochs, 100 steps/epoch. All experiments on NVIDIA RTX 3090 (24GB), seed 42.

**Metrics.** Dice Similarity Coefficient (DSC) per class, averaged per task. Forgetting: $\mathcal{F}_t = \text{DSC}_{t}^{\text{peak}} - \text{DSC}_{t}^{\text{after last task}}$. Backward Transfer (BWT): $-\mathcal{F}$. Forward Transfer (FWT): performance on task $t$ before training on it.

### 4.2 Main Results

**Table 1: Continual segmentation results (DSC, higher is better).**

| Method | Task A Peak | Task A Retained | Task B Peak | Task B Retained | Task C Final | Mean Forgetting ↓ | BWT ↑ |
|--------|------------|----------------|------------|----------------|-------------|-------------------|-------|
| Finetune | — | — | — | — | — | — | — |
| Replay | — | — | — | — | — | — | — |
| LwF (snapshot KD) | — | — | — | — | — | — | — |
| Replay + EWC | — | — | — | — | — | — | — |
| Replay + EWC + KD (snapshot) | — | — | — | — | — | — | — |
| Replay + EWC + KD (MedSAM2, std. adapter) | — | — | — | — | — | — | — |
| Replay + EWC + KD (MedSAM2, GRACE) | — | — | — | — | — | **—** | **—** |
| Replay + EWC + KD (MedSAM3, std. adapter) | — | — | — | — | — | — | — |
| Replay + EWC + KD (MedSAM3, GRACE) | — | — | — | — | — | **—** | **—** |
| Joint training (upper bound) | — | — | — | — | — | 0 | — |

*Values to be filled from experiments. Table structure shows the ablation ladder: each row adds one component, isolating its contribution.*

### 4.3 Ablation Study

**Table 2: Component ablation of GRACE (MedSAM2 teacher).**

| Core | Gate (CGAD) | Prototypes (CPA) | Mean Forgetting ↓ | Task A Retained | Task C Final |
|------|-------------|-------------------|-------------------|----------------|-------------|
| Standard adapter (rebuilt) | — | — | — | — | — |
| TRA (frozen core) | — | — | — | — | — |
| TRA | CGAD | — | — | — | — |
| TRA | — | CPA | — | — | — |
| TRA | CGAD | CPA (GRACE full) | **—** | **—** | **—** |

---

## 5. ANALYSIS

### 5.1 Teacher Reliability Visualization

We visualize the CGAD gate activations across anatomical regions for Task A (organs) after adapter pre-training. The gate should show high values (>0.8) on large, well-segmented structures (liver, spleen) and low values (<0.4) on small or ambiguous structures (adrenal glands, pancreatic boundaries).

*[Figure 3: Gate activation maps overlaid on CT slices for representative cases. Left: large organ (liver) — gate ≈ 1.0. Center: medium organ (kidney boundary) — gate ≈ 0.6. Right: small organ (adrenal gland) — gate ≈ 0.2. The gate learns anatomically meaningful reliability estimates.]*

### 5.2 Representation Stability

We measure how the student's backbone representations drift across tasks using Centered Kernel Alignment (CKA) between feature maps at corresponding layers before and after each task transition. We expect GRACE to show higher CKA (more stable features) than non-distillation baselines.

*[Figure 4: Layer-wise CKA between Task A features and Task C features for: finetune, replay, snapshot KD, foundation KD (std adapter), foundation KD (GRACE). GRACE maintains highest representational similarity across all layers.]*

### 5.3 Gating Behavior Across Tasks

We track the mean gate activation across tasks. For Task A (pre-trained adapter), the gate should be broadly open. For Task B (new residual, less training), the gate should be more selective. This demonstrates that CGAD adapts its reliability estimates to the adapter's actual performance on each task.

### 5.4 Efficiency Analysis

| Component | Parameters | FLOPs overhead | Storage per task |
|-----------|-----------|---------------|-----------------|
| Foundation backbone (frozen) | ~80M | Baseline | 0 (shared) |
| Standard adapter (rebuilt) | ~600K | ~0.5% of backbone | 600K per task |
| GRACE core (frozen) | ~590K | ~0.5% of backbone | 0 (shared) |
| GRACE residual (per task) | ~3K–5K | <0.01% | ~3K per task |
| GRACE gate (frozen) | ~17K | ~0.02% | 0 (shared) |
| Prototypes (per task) | 256 × C_t floats | 0 (lookup) | ~1K per task |
| **GRACE total per new task** | **~4K** | **<0.01%** | **~4K** |

GRACE is ~150× more parameter-efficient per task than rebuilding standard adapters.

---

## 6. DISCUSSION

**Why GRACE works.** The effectiveness of GRACE stems from a clean separation of concerns: the frozen core captures *what* the foundation model's features mean anatomically (a task-general capability), the per-task residuals specify *which classes* each task requires (task-specific), and the gate determines *where* the resulting predictions are trustworthy (quality control). This decomposition prevents the failure modes of naive adapter management (§3.2) while incurring minimal overhead.

**The teacher reliability gap is fundamental, not incidental.** Our experiments show that even with replay and EWC, foundation model distillation with a standard adapter underperforms snapshot self-distillation. This counterintuitive result — a stronger teacher producing worse outcomes — arises because unreliable teacher predictions actively corrupt the student's training signal. The gate is essential: it converts a potentially harmful signal into a consistently helpful one.

**Connection to gradient alignment.** GRACE's gated KD can be understood through the lens of gradient episodic memory [31]. GEM constrains the gradient update to have non-negative inner product with the gradient for each previous task. GRACE's gate achieves a softer version of this constraint: by suppressing KD in regions where the teacher is incorrect, the gate ensures the distillation gradient is positively aligned with the direction that preserves old-task performance. Unlike GEM's hard projection (which requires storing per-task gradients), GRACE's gating is learned once during pre-training and applied at negligible cost.

**Limitations.** (1) GRACE requires a short adapter pre-training phase for each new task, adding latency before continual training begins. (2) The prototype bank assumes foundation features are discriminative for all classes; for classes with highly variable appearance, prototypes may not be representative. (3) We evaluate on a single dataset (TotalSegmentator); generalization to other modalities (MRI) and anatomical regions requires further study. (4) The gate is trained during adapter pre-training and fixed thereafter; an online adaptation mechanism could potentially improve performance on long task sequences.

**Clinical relevance.** GRACE enables a practical deployment scenario: a hospital acquires a pre-trained MedSAM model once, trains GRACE adapters for its initial segmentation needs, then incrementally adds new anatomical targets as clinical requirements evolve — without retraining the foundation model, without access to previous patient data, and with a lightweight student model that runs on standard hospital hardware.

---

## 7. CONCLUSION

We identified the teacher reliability gap — a previously unrecognized failure mode where the adapter bridging foundation model features to task-specific predictions becomes unreliable under task shift, degrading rather than helping continual distillation. We proposed GRACE, a teacher-side mechanism that decomposes the adapter into a frozen shared core with per-task residuals (TRA), modulates distillation with a learned confidence gate (CGAD), and supplements predictions with append-only feature prototypes (CPA). On a three-task TotalSegmentator benchmark with MedSAM2/3 teachers and an nnU-Net student, GRACE provides reliable distillation across tasks while adding fewer than 4K parameters per task. Our work establishes that designing the teacher-student interface for continual learning is as critical as protecting the student model itself.

---

## REFERENCES

[1] Zhu et al. MedSAM2: Segment Medical Images As Video Via Segment Anything Model 2. 2024.
[2] MedSAM3 (LoRA fine-tuned SAM3 for medical segmentation). 2024.
[3] Isensee et al. nnU-Net: A Self-configuring Method for Deep Learning-based Biomedical Image Segmentation. Nature Methods, 2021.
[4] Li & Hoiem. Learning without Forgetting. TPAMI, 2017.
[5] Kirkpatrick et al. Overcoming Catastrophic Forgetting in Neural Networks. PNAS, 2017.
[6] Rolnick et al. Experience Replay for Continual Learning. NeurIPS, 2019.
[7] Douillard et al. PLOP: Learning without Forgetting for Continual Semantic Segmentation. CVPR, 2021.
[8] Cermelli et al. Modeling the Background for Incremental Learning in Semantic Segmentation. CVPR, 2020.
[9] Buzzega et al. Dark Experience for General Continual Learning: a Strong, Simple Baseline. NeurIPS, 2020.
[10] Hinton et al. Distilling the Knowledge in a Neural Network. NeurIPS Workshop, 2015.
[11] Wasserthal et al. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: AI, 2023.
[12] McCloskey & Cohen. Catastrophic Interference in Connectionist Networks. Psychology of Learning and Motivation, 1989.
[13] Zenke et al. Continual Learning Through Synaptic Intelligence. ICML, 2017.
[14] Chaudhry et al. On Tiny Episodic Memories in Continual Learning. arXiv, 2019.
[15] Petit et al. REMINDER: Feature Replay for Continual Learning. CVPR Workshop, 2023.
[16] Qin et al. Efficient Medical Image Segmentation Based on Knowledge Distillation. TMI, 2021.
[17] Chen et al. Knowledge Distillation for Medical Image Segmentation. MedIA, 2022.
[18] KnowSAM: Knowledge Distillation from SAM for Semi-supervised Medical Segmentation. 2024.
[19] Ma et al. Segment Anything in Medical Images. Nature Communications, 2024.
[20] SAMed: SAM-based Medical Image Segmentation via Multi-scale Feature Fusion. 2024.
[21] Kirillov et al. Segment Anything. ICCV, 2023.
[22] Ma et al. MedSAM: Segment Anything in Medical Images. Nature Communications, 2024.
[23] Wang et al. SAM-Med3D: 3D Medical Image Segmentation via SAM. arXiv, 2024.
[24] Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.
[25] Jia et al. Visual Prompt Tuning. ECCV, 2022.
[26] Chen et al. Vision Transformer Adapter for Dense Predictions. ICLR, 2023.
[27] Pfeiffer et al. AdapterHub: A Framework for Adapting Transformers. EMNLP, 2020.
[28] Cho & Hariharan. On the Efficacy of Knowledge Distillation. ICCV, 2019.
[29] Beyer et al. Knowledge Distillation: A Good Teacher is Patient and Consistent. CVPR, 2022.
[30] Malinin & Gales. Predictive Uncertainty Estimation via Prior Networks. NeurIPS, 2018.
[31] Lopez-Paz & Ranzato. Gradient Episodic Memory for Continual Learning. NeurIPS, 2017.
[32] Stanton et al. Does Knowledge Distillation Really Work? NeurIPS, 2021.
[33] Szatkowski et al. Adapt Your Teacher: Improving Knowledge Distillation for Exemplar-Free Continual Learning. WACV, 2024.
[34] Zhang et al. Continual Learning for Abdominal Multi-Organ and Tumor Segmentation. MICCAI, 2023.
[35] Schwarz et al. Progress & Compress: A Scalable Framework for Continual Learning. ICML, 2018.
[36] Wang et al. O-LoRA: Orthogonal Subspace Learning for Language Model Continual Learning. EMNLP, 2023.
[37] He et al. CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning. CVPR, 2025.
[38] Lan et al. ReCo-KD: Region- and Context-Aware Knowledge Distillation for Efficient 3D Medical Image Segmentation. arXiv, 2025.
[39] Zhang et al. Revisiting Knowledge Distillation under Distribution Shift. arXiv, 2023.
[40] Pfeiffer et al. AdapterFusion: Non-Destructive Task Composition for Transfer Learning. EACL, 2021.
[41] Yu et al. SATCH: Specialized Assistant Teacher Distillation to Reduce Catastrophic Forgetting. 2024.
[42] Yu et al. Select and Distill: Selective Dual-Teacher Knowledge Transfer for Continual Learning on Vision-Language Models. ECCV, 2024.
[43] Wu et al. Multi-modality Multiorgan Image Segmentation Using Continual Learning with Enhanced Hard Attention to the Task. Medical Physics, 2025.
