# Coherent Multi-Teacher Knowledge Distillation through Multi-Agent Reinforcement Learning

**A Formal Theory of Knowledge Aggregation, Synchronization, and Optimal Teacher Cardinality**

This repository provides a clean-room PyTorch implementation of **CoMTKD-MARL** for CIFAR-100. The project follows the top-level organization and command style of the MTKD-RL reference implementation while adding cooperative teacher agents, a centralized coherence critic, a synchronization oracle, counterfactual teacher credit, and a learned teacher-cardinality policy.

The intended repository location is:

```text
https://github.com/PanasheC/CoMTKD-MARL
```

## Abstract

Multi-teacher knowledge distillation can expose a compact student to complementary predictive, representational, and relational knowledge. Existing methods mainly optimize teacher weights from confidence or teacher-student discrepancy, while reinforcement learning based methods typically place one decision agent above the teacher pool. This formulation does not explicitly coordinate the teachers, synchronize the knowledge being transferred, quantify redundancy and conflict, or determine when an additional teacher no longer improves learning. This paper develops Coherent Multi-Teacher Knowledge Distillation with Multi-Agent Reinforcement Learning, a framework in which each teacher is represented by a cooperative knowledge agent. The agents select participation, teaching strength, temperature, and knowledge channels from local observations, while a centralized coherence critic and a synchronization operator coordinate the joint teaching policy during training. The student remains independent at inference. The framework aggregates calibrated logits, projected features, relational structure, uncertainty, and marginal novelty, and it penalizes epistemic disagreement, duplicated information, unstable weighting, communication cost, and student-capacity mismatch. Three formal results organize the theory. The Multi-Teacher Advantage Theorem gives sufficient covariance and transfer conditions under which a coordinated teacher coalition has lower expected risk than every admissible single teacher. The Knowledge Coherence Theorem proves geometric contraction of disagreement under a connected synchronization graph and gives a nonzero robustness floor under bounded communication perturbations. The Teacher Saturation Theorem combines conditional mutual information, a bounded student information channel, and increasing coordination cost to establish a finite optimal teacher cardinality and an operational stopping rule. A complete bilevel learning objective, counterfactual credit assignment mechanism, cardinality controller, algorithms, complexity analysis, and reproducible experimental protocol are provided. The resulting theory explains both the advantage of diverse teachers and the saturation or deterioration that occurs when redundant, conflicting, or undistillable teachers are added.

## Research question

> Under what mathematical and operational conditions does a coordinated set of teachers transfer more useful knowledge than a single teacher, and at what point does adding teachers cease to improve the student?

## Central thesis

Multiple teachers improve student learning when their errors are sufficiently diverse, their information is complementary, their signals are compatible with the student, and the joint target remains coherent. The marginal value of a new teacher decreases as redundancy, conflict, synchronization cost, compute cost, and student-capacity mismatch increase. The optimal number of active teachers is therefore a learned property of the teacher pool, sample, student state, and training stage.

## Formal theorem summary

### 1. Multi-Teacher Advantage Theorem

Let the teacher error vector be

$$
 e_m(x)=p_m(x)-p^*(x),
$$

and define the teacher error second-moment matrix

$$
 C_{mn}=\mathbb{E}_X\left[\langle e_m(X),e_n(X)\rangle\right].
$$

For a probability barycenter with weights $\mathbf{w}\in\Delta^{M-1}$,

$$
\mathcal{R}_T(\mathbf{w})=\mathbf{w}^{\top}C\mathbf{w}.
$$

When $C\succ0$ and $C^{-1}\mathbf{1}$ is strictly positive, the interior optimum is

$$
\mathbf{w}^{*}=\frac{C^{-1}\mathbf{1}}{\mathbf{1}^{\top}C^{-1}\mathbf{1}},
\qquad
\mathcal{R}_T(\mathbf{w}^{*})=\frac{1}{\mathbf{1}^{\top}C^{-1}\mathbf{1}}.
$$

The coordinated coalition has lower expected risk than every admissible single teacher when

$$
\frac{1}{\mathbf{1}^{\top}C^{-1}\mathbf{1}}+\epsilon_{\mathrm{tr}}^{(M)}
<
\min_m C_{mm}+\underline{\epsilon}_{\mathrm{tr}}^{(1)}.
$$

The code estimates $C$, solves the simplex-constrained quadratic problem, and reports the empirical risk margin.

### 2. Knowledge Coherence Theorem

Let $K^{(r)}\in\mathbb{R}^{M\times d}$ contain teacher knowledge after synchronization round $r$. Define

$$
J=\frac{1}{M}\mathbf{1}\mathbf{1}^{\top},
\qquad
\Pi=I-J,
\qquad
\mathrm{CI}^{(r)}=\frac{1}{M}\|\Pi K^{(r)}\|_F^2.
$$

For a primitive, doubly stochastic mixing matrix $W$ with

$$
\rho=\|W-J\|_2<1,
$$

exact synchronization satisfies

$$
\mathrm{CI}^{(r)}\leq \rho^{2r}\mathrm{CI}^{(0)}.
$$

With bounded perturbation $\|\Pi E^{(r)}\|_F\leq\varepsilon_E$,

$$
\sqrt{\mathrm{CI}^{(r)}}
\leq
\rho^r\sqrt{\mathrm{CI}^{(0)}}
+
\frac{\varepsilon_E}{\sqrt{M}(1-\rho)}.
$$

The synchronization oracle constructs a symmetric Metropolis mixing matrix, performs graph consensus, measures coherence before and after synchronization, and records the spectral contraction factor.

### 3. Teacher Saturation Theorem

For an ordered teacher set $\mathcal{S}_M$, define gross information

$$
G_M=I(Y;K_{\mathcal{S}_M}),
$$

and conditional gain

$$
g_{M+1}=I\left(Y;K_{j_{M+1}}\mid K_{\mathcal{S}_M}\right).
$$

If the student information channel is bounded by

$$
I(Z_M;K_{\mathcal{S}_M})\leq B_S,
$$

then usable information is bounded by

$$
I(Y;Z_M)\leq\min\{G_M,B_S\}.
$$

For net value

$$
U_M=\min\{G_M,B_S\}-C_M,
$$

a sufficient stopping rule is

$$
\min\left\{g_{M+1},[B_S-G_M]_+\right\}\leq d_{M+1},
$$

where $d_{M+1}=C_{M+1}-C_M$ is the incremental coordination cost. The code provides a learned cardinality policy and an offline theorem-validation utility for fixed-cardinality sweeps.

## Architecture

Each teacher has a separate actor. The actor observes teacher quality, teacher-student compatibility, novelty, redundancy, conflict, and cost. It produces:

$$
a_{m,t}=(g_{m,t},\alpha_{m,t},\tau_{m,t},\boldsymbol{\beta}_{m,t}),
$$

where:

- $g_{m,t}$ is a soft participation gate.
- $\alpha_{m,t}$ is an importance score.
- $\tau_{m,t}$ is a sample-specific distillation temperature.
- $\boldsymbol{\beta}_{m,t}$ allocates teaching strength across logits, features, relations, and uncertainty.

The normalized teacher weights are

$$
w_{m,t}=
\frac{g_{m,t}\exp(\alpha_{m,t})}
{\sum_j g_{j,t}\exp(\alpha_{j,t})+\varepsilon}.
$$

A separate cardinality policy chooses the active count from $1$ to $M$. The highest-scoring teachers form the active coalition. A centralized critic observes the joint state and joint action during training. The student is the only model required at inference.

The team reward implements the paper objective:

$$
\begin{aligned}
R_t={}&
\frac{V_t-V_{t+1}}{|V_t|+\varepsilon}
+\eta_N\mathrm{Novelty}_t
-\eta_C\mathrm{CI}_t \\
&-\eta_R\mathrm{Red}_t
-\eta_X\mathrm{Conf}_t
-\eta_K\sum_m g_{m,t}
-\eta_S\|\mathbf{w}_t-\mathbf{w}_{t-1}\|_2^2.
\end{aligned}
$$

The default implementation uses contextual MAPPO updates. Each CIFAR-100 sample acts as a one-step cooperative decision context, while the evolving student supplies the non-stationary environment across batches.

## Repository structure

```text
CoMTKD-MARL/
├── dataset/
│   ├── __init__.py
│   └── cifar100.py
├── distiller_zoo/
│   ├── __init__.py
│   ├── coherent_losses.py
│   └── feature_mse_mtkd_rl.py
├── helper/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── logger.py
│   ├── metrics.py
│   ├── reproducibility.py
│   └── theorem_metrics.py
├── models/
│   ├── comtkd_marl/
│   │   ├── actor.py
│   │   ├── cardinality.py
│   │   ├── controller.py
│   │   ├── critic.py
│   │   ├── observations.py
│   │   ├── ppo.py
│   │   └── synchronization.py
│   ├── cifar_resnet.py
│   ├── registry.py
│   ├── torchvision_wrappers.py
│   ├── util.py
│   └── wide_resnet.py
├── experiments/
│   ├── cardinality_sweep.py
│   ├── summarize_runs.py
│   └── validate_theorems.py
├── tests/
├── paper/
│   └── CoMTKD_MARL_Algorithmica_NeurIPS.tex
├── setting.py
├── train_baseline.py
├── train_loops.py
├── train_student_avg.py
├── train_student_rl.py
├── train_student_comtkd_marl.py
├── evaluate.py
├── requirements.txt
└── README.md
```

The top-level `dataset`, `distiller_zoo`, `helper`, and `models` layout mirrors the reference repository. The CoMTKD-MARL modules are isolated under `models/comtkd_marl`.

## Installation

Recommended environment:

- Ubuntu 20.04 or newer
- Python 3.10 or newer
- PyTorch 2.2 or newer
- CUDA 11.8 or newer for GPU training

```bash
git clone https://github.com/PanasheC/CoMTKD-MARL.git
cd CoMTKD-MARL
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the unit tests:

```bash
make test
```

Run the CPU end-to-end smoke test:

```bash
make smoke
```

## CIFAR-100 experiment

`torchvision.datasets.CIFAR100` downloads and verifies the dataset automatically when `--download` is enabled.

### Step 1. Train the teacher pool

```bash
python train_baseline.py \
  --model RegNetY_400MF \
  --data-folder ./data \
  --checkpoint-dir ./checkpoints/teachers \
  --amp

python train_baseline.py \
  --model RegNetX_400MF \
  --data-folder ./data \
  --checkpoint-dir ./checkpoints/teachers \
  --amp

python train_baseline.py \
  --model resnet32x4 \
  --data-folder ./data \
  --checkpoint-dir ./checkpoints/teachers \
  --amp

python train_baseline.py \
  --model wrn_28_4 \
  --data-folder ./data \
  --checkpoint-dir ./checkpoints/teachers \
  --amp
```

The convenience script runs all four commands:

```bash
bash scripts/train_teachers.sh ./data ./checkpoints/teachers
```

Update `setting.py` with the exact best-checkpoint paths produced by these runs. You can also pass paths explicitly:

```bash
--teacher-checkpoint RegNetY_400MF=/path/to/RegNetY_400MF_best.pth.tar
```

### Step 2. Equal-weight multi-teacher baseline

```bash
python train_student_avg.py \
  --data ./data \
  --arch ShuffleV2 \
  --checkpoint-dir ./checkpoints/average \
  --teacher-name-list RegNetY_400MF RegNetX_400MF resnet32x4 wrn_28_4 \
  --amp
```

### Step 3. CoMTKD-MARL

```bash
python train_student_rl.py \
  --data ./data \
  --arch ShuffleV2 \
  --dynamic \
  --checkpoint-dir ./checkpoints/comtkd_marl \
  --teacher-name-list RegNetY_400MF RegNetX_400MF resnet32x4 wrn_28_4 \
  --sync-rounds 3 \
  --rollout-size 1024 \
  --amp
```

The descriptive alias is equivalent:

```bash
python train_student_comtkd_marl.py [same arguments]
```

### Fixed-cardinality ablations

To train with exactly three active teachers per sample:

```bash
python train_student_rl.py \
  --data ./data \
  --arch ShuffleV2 \
  --forced-cardinality 3 \
  --teacher-name-list RegNetY_400MF RegNetX_400MF resnet32x4 wrn_28_4 \
  --amp
```

To run a multi-seed cardinality sweep:

```bash
python experiments/cardinality_sweep.py \
  --max-teachers 4 \
  --seeds 11 22 33 44 55 \
  --output-root ./checkpoints/cardinality_sweep \
  --extra --data ./data --arch ShuffleV2 --amp
```

## Validate the theoretical quantities

```bash
python experiments/validate_theorems.py \
  --teacher-name-list RegNetY_400MF RegNetX_400MF resnet32x4 wrn_28_4 \
  --data ./data \
  --output ./results/theorem_validation \
  --sync-rounds 8 \
  --per-teacher-cost 0.001
```

The command writes:

- `summary.json`, including the teacher error covariance, optimal simplex weights, coalition risk, best single-teacher risk, and estimated optimal cardinality.
- `coherence_curve.csv`, including coherence and spectral contraction by synchronization round.
- `cardinality_curve.csv`, including the greedy nested coalition, risk, gross gain, coordination cost, and net value.

These measurements test the stated sufficient conditions. They do not replace mathematical proof and they do not guarantee that nonconvex student optimization reaches the target-risk optimum.

## Main observation vector

Each teacher actor receives twelve normalized variables:

1. Teacher cross-entropy loss.
2. Teacher predictive entropy.
3. Teacher confidence.
4. Teacher correctness indicator.
5. Teacher-student KL divergence.
6. Teacher-student feature similarity.
7. Logit-space gradient alignment.
8. Marginal novelty proxy.
9. Peer redundancy.
10. Peer conflict measured by Jensen-Shannon divergence.
11. Student-capacity mismatch proxy.
12. Normalized teacher compute cost.

## Reproducibility protocol

For publication experiments:

1. Use at least five independent seeds.
2. Preserve the same pretrained teacher checkpoints across all student methods.
3. Report mean, standard deviation, and 95 percent confidence intervals.
4. Compare single-teacher KD, equal averaging, MTKD-RL style independent weighting, fixed-cardinality CoMTKD-MARL, and learned-cardinality CoMTKD-MARL.
5. Report top-1 accuracy, top-5 accuracy, expected calibration error, Brier score, active teacher count, coherence index, spectral contraction, wall-clock time, and peak GPU memory.
6. Validate theorem conditions separately from final student accuracy.
7. Publish all configuration files, seed lists, checkpoints, and JSONL metric logs.

No experimental numbers are hard-coded in this repository.

## Citation

```bibtex
@article{chiurunge2026comtkd,
  title={Coherent Multi-Teacher Knowledge Distillation through Multi-Agent Reinforcement Learning: A Formal Theory of Knowledge Aggregation, Synchronization, and Optimal Teacher Cardinality},
  author={Chiurunge, Panashe},
  year={2026}
}
```

## Acknowledgment of the reference implementation

The repository structure and CIFAR-100 command conventions were informed by the public MTKD-RL implementation:

```text
https://github.com/winycg/MTKD-RL
```

This repository is a clean-room implementation. It does not copy the original source files and it introduces a distinct cooperative MARL architecture and theoretical validation suite.

## License

MIT License. See `LICENSE`.
