---
title: answer
order: 2
---

## Part 1/4: Positioning and synthetic rank comparisons

We thank you for asking us to clarify the contribution of the representation and separate it from the efficiency of the neural model. Your comments have helped us refine the paper's positioning. TSFlow focuses on matching the prior to the temporal structure of the data. We ask whether the prior is a good probabilistic forecaster in its own right. When it is, how much can a very small neural network gain from the predictive information it already contains? We build on TSFlow's use of Gaussian process priors in flow matching to learn a small correction to that forecast. The goal is to make the most of that starting forecast with minimal training cost. For real data, our empirical Gaussian prior follows Lin et al., *Empirical Gaussian Processes* (ICML 2026; https://arxiv.org/abs/2602.12082).

Our results show that a small learned correction improves the statistical forecast and reaches accuracy competitive with larger task-trained forecasters, at substantially lower training cost in the profiled implementations.

The forecasting comparisons measure how much the correction adds to the starting forecast and how much training it requires. Matched controls separate this benefit from the effect of the coordinates. The original representation studies focused on optimization and stability. New truncation experiments explore whether the prior can also guide a smaller correction.

### W1: Contribution beyond a coordinate rotation

We added a synthetic variant that learns a correction in fewer directions. The starting GP forecast $y_0\sim\mathcal{N}(\mu,\Sigma)$ stays at full resolution. The MLP operates in the top $k$ eigenvectors $\Phi_k\in\mathbb{R}^{L_f\times k}$ of the conditional future covariance $\boldsymbol{\Sigma}$:
$$\hat{y}_f = y_0 + \Phi_k (\hat{c}_1 - c_0).$$
At full rank ($k=L_f$), changing the basis preserves the possible corrections. At reduced rank ($k\ll L_f$), the basis selects which directions the model can correct. This restriction can also regularize learning.

Here $\mu$ and $\Sigma$ are the conditional forecast's mean and covariance. The horizon length is $L_f$, and $k$ is the correction rank. The starting coordinates are $c_0=\Phi_k^\top y_0$, and $\hat c_1$ is their learned endpoint. The network is a multilayer perceptron (MLP).

We compare the prior's eigenbasis (Mercer) with principal component analysis (PCA), the discrete cosine transform (DCT), a Fourier basis, and time coordinates (Raw). PCA estimates its directions from data. DCT and Fourier use fixed directions.

We report the continuous ranked probability score (CRPS), which measures probabilistic forecast error. Lower is better. Percentages show how much of the prior-to-oracle gap the correction closes: (prior CRPS − model CRPS) / (prior CRPS − oracle CRPS). The oracle score is the synthetic benchmark's reference.

Table 1: Effect of truncation rank $k$ on non-stationary modulated periodic data ($L_f=32$, GP Prior CRPS = 0.2565, Oracle = 0.0200).

| Basis | $k=4$ CRPS $\downarrow$ | $k=6$ CRPS $\downarrow$ | $k=10$ CRPS $\downarrow$ (Gap $\uparrow$) | $k=32$ (Full) CRPS $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: |
| Mercer (Ours) | 0.1926 | 0.1619 | 0.0768 (76.0%) | 0.0531 |
| PCA | 0.1959 | 0.1660 | 0.0884 (71.1%) | 0.0547 |
| DCT | 0.2385 | 0.2315 | 0.2140 (18.0%) | 0.0528 |
| Fourier | 0.2421 | 0.2304 | 0.2167 (16.8%) | 0.0543 |
| Raw | 0.2411 | 0.2331 | 0.2180 (16.3%) | 0.0546 |

At full rank, forecast errors are close across bases. With ten correction directions, Mercer reaches CRPS 0.0768, compared with 0.2140–0.2180 for DCT, Fourier, and Raw. The error is nearly three times lower. Here the basis determines which corrections the smaller model can learn.

---

## Part 2/4: Basis estimation, optimizer analysis, and reporting

### W1: Basis estimation and regularization

We added two controlled studies of this choice. First, we fix the correction's training data (512 samples, $k=8$) and vary only the data used to estimate PCA's basis. Mercer gives lower CRPS when PCA's calibration sample is scarce or comes from a shifted historical regime.

Second, we compare rank-6 and full-rank corrections with only $N=64$ training samples. For the tested radial basis function (RBF) and Matérn-5/2 kernels, six directions capture more than 99.6% of prior variance. The smaller correction gives lower CRPS. This shows a regularization benefit in this small-sample setting.

Table 2: Absolute CRPS effects in the controlled studies (negative differences favor the first method).

| Setting | Comparison | CRPS difference |
|---|---|---:|
| Periodic, rank 10 | Mercer − Fourier | −0.1399 |
| Periodic, rank 10 | Mercer − PCA | −0.0116 |
| Scarce calibration | Mercer − PCA | −0.0137 |
| Shifted calibration | Mercer − PCA | −0.0359 |
| RBF, N=64 | Rank 6 − full rank | −0.0313 |

The prior can help select useful correction directions at low rank. It can also reduce sensitivity to scarce or shifted PCA calibration data. Once the kernel is chosen, its basis requires no separate calibration sample. Covariance construction and eigendecomposition still incur computation.

### Q2: Optimizer analysis and conditional interpolants

We have corrected the conditioning analysis. It examines how the prior's coordinates affect the geometry of learning. For fixed covariance $K$ and orthogonal basis $\Phi$, write $A=\Phi^\top K\Phi$ and $S=\operatorname{diag}(A)$. The corrected frozen input-curvature surrogate is $S^{-1/4}AS^{-1/4}$. In the eigenbasis, this matrix is diagonal. Its eigenvalues are the square roots of those of $A$, so its condition number is $\sqrt{\kappa(A)}$.

Measurements on the source–target interpolation test this geometry in the synthetic training distributions. Actual Adam and AdamW also depend on residuals, momentum, and evolving second moments. The surrogate explains a mechanism, while forecast scores measure accuracy. The real-data experiments use Adam.

### Q3: Effect sizes and statistical reporting

We will replace threshold-based rankings with absolute scores and effect sizes, as in the tables above. The real-data results use one training seed per task and describe the observed benchmark effects. The controlled studies identify settings with large representation effects. The full-rank real-data controls show a smaller effect.

---

## Part 3/4: Real-data benchmark and efficiency attribution

### W2 / Q1: Efficiency attribution and real-data comparisons

The $4.7\times$ memory and $4\times$ runtime gains over TSFlow come from replacing S4 with an MLP. We will attribute them to that architecture change. To isolate the coordinates' effect, we now compare the same MLP and training recipe in time coordinates and in the prior's eigenbasis. TiDE provides a direct-forecast MLP reference.

A smaller MLP alone leaves the role of the representation unresolved. Your questions motivated matched controls and a broader evaluation across all 54 GIFT-Eval tasks whose datasets contain single-channel series. We use official splits, forecast origins, and scoring. The controls identify where improvements come from. The external comparisons assess accuracy and training cost.

### Prior and representation controls

We compare the statistical forecast alone with three learned models. GP-only is the fitted forecast without a correction. Raw-GP adds a correction in the original time coordinates. It uses the same Gaussian starting forecast and correction network as our model. Raw-noise uses that network in time coordinates but starts from independent Gaussian noise. Our model starts from the GP forecast and learns the correction in its eigenbasis. All neural models use the same training recipe.

The matched controls below use the same 54 tasks and equal-weight geometric mean wQL.

| Method | Geom. mean wQL ↓ |
|---|---:|
| GP-only (statistical forecast) | 0.163321 |
| Raw-noise (correction from independent noise) | 0.156343 |
| Raw-GP (prior + correction in time coordinates) | 0.141639 |
| Ours (prior + correction in its eigenbasis) | 0.140626 |
| Ours, half input/output ranks (width 192) | 0.142803 |

Halving both the input and correction output ranks while keeping the network width at 192 increases aggregate wQL by only 1.55%, preserving most of the full-rank model's accuracy.

The main gain comes from learning a correction to a useful forecast. The correction improves on the GP alone, and starting from that forecast also improves on starting from independent noise. Changing to the prior's eigenbasis gives a smaller additional gain. The synthetic studies examine settings where the choice of correction directions matters more.

### Broader forecasting comparison

We compare against two groups of forecasters. DLinear, TiDE, PatchTST, and TFT are trained separately on each dataset. DLinear is linear, TiDE is a direct-forecast MLP, and PatchTST is a patch-based transformer. TFT is the Temporal Fusion Transformer. Chronos-2, TiRex, and TimesFM-2.5 are pretrained probabilistic models used without dataset-specific training (zero-shot). Our model fits an empirical Gaussian forecast to the training data, then learns a correction in its covariance eigenbasis.

We report weighted quantile loss (wQL) for probabilistic forecasts and mean absolute scaled error (MASE) for forecast medians. Training cost is measured in millions of floating-point operations (MFLOPs) per sample. Lower is better for all three.

| Method | Geom. mean wQL ↓ | Geom. mean MASE ↓ | Geom. mean training MFLOPs/sample ↓ |
|---|---:|---:|---:|
| Ours (empirical eigenbasis + GP) | 0.1406 | 1.1025 | 2.763 |
| TSFlow (locally reproduced) | 0.1578 | 1.1926 | 663.046 (estimated) |
| DLinear | 0.2054 | 1.3466 | 1.456 |
| TiDE | 0.1751 | 1.3677 | 0.341 |
| PatchTST | 0.1449 | 1.1488 | 11.653 |
| TFT | 0.1420 | 1.1521 | 71.051 |
| Chronos-2 (zero-shot) | 0.1189 | 0.9829 | — |
| TiRex (zero-shot) | 0.1205 | 0.9936 | — |
| TimesFM-2.5 (zero-shot) | 0.1207 | 0.9895 | — |

All columns are geometric means over the same 54 tasks, with equal weight per task. Our results use one training seed. Context lengths and checkpoints are selected on validation data. Our forecasts use 20 velocity-field evaluations per trajectory and 100 draws. TSFlow is locally reproduced on all 54 tasks using seed 6432: we select among OU, squared-exponential, and periodic kernels by the best validation wQL through epoch 40, then continue the selected kernel to 400 total epochs. Its model, optimizer, EMA, gradient clipping, normalization, and sampling settings follow the original TSFlow recipe; early kernel selection and single-seed evaluation define this campaign's budget. The other external accuracy scores come from the published GIFT-Eval results.

Our correction achieves 10.86% lower aggregate wQL and 7.55% lower aggregate MASE than this TSFlow reproduction on the complete shared panel.

Training profiles cover a forward pass, loss computation, and backward pass for one univariate window at batch size one. They include our prior conditioning, sampling, and coordinate transforms. TSFlow uses its final selected kernels and native training loss, including GP conditioning and factorization; S4 arithmetic uses a dense Cauchy reference and analytical FFT estimates. Shared S4 kernel generation and GP factorization are counted once per window at batch one and amortize over larger batches. Optimizer updates, clipping, EMA, validation, and one-time preparation are excluded. Other task-trained baseline costs use profiled reference architectures and context lengths. Accuracy scores follow the published evaluation recipes. The pretrained models require no task-specific training, so their entries are blank in this column. They rely on substantial computation already invested in pretraining. That cost is separate from the task-specific training measured here.

A strong statistical forecast lets a small learned correction reach accuracy competitive with PatchTST and TFT at substantially lower profiled training cost. The aggregate accuracy gains are modest, and performance varies across tasks. The component controls give the clearest evidence for the approach: the correction improves the starting forecast itself. The pretrained models achieve stronger accuracy and require no downstream training.

---

## Part 4/4: Scope of the optimization comparison

These comparisons use Adam with learning rate 0.001 and a matched training budget. They measure performance under a shared training recipe. The synthetic rank comparisons examine the larger representation effects that appear when the correction is restricted to fewer directions.
