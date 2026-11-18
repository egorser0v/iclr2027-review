---
title: answer
order: 2
---

## Part 1/4: Positioning and synthetic rank comparisons

This part clarifies our contribution and presents the synthetic evidence for representation choice under truncation.

We thank you for asking us to clarify the contribution of the representation and distinguish it from backbone efficiency. The question motivating our work is how much neural computation probabilistic forecasting requires when a statistical model already provides a strong forecast. Building on flow matching with Gaussian process priors, as in TSFlow, we ask how to integrate that forecast so that a small supervised model can compete with larger neural forecasters. The MLP tests how little neural capacity is needed for correction, while the covariance eigenbasis addresses how the prior's structure can support optimization and stability. In the empirical study, the fitted Gaussian prior builds on the data-driven prior formulation of Lin et al., Empirical Gaussian Processes (ICML 2026; https://arxiv.org/abs/2602.12082).

The evidence below addresses two aspects of this question. The forecasting comparisons and component controls establish how much a small neural correction adds to a strong statistical forecast, and where its accuracy and computational savings come from. The representation studies examine how the correction should use the prior's structure: the original paper focused on optimization and stability, while the truncation experiments added for this rebuttal explore whether that structure can also support a smaller correction. Together, these studies explain both the practical value of the lightweight model and the specific role of its coordinates.

### W1: Contribution beyond a coordinate rotation

To address this concern, we added a synthetic low-rank formulation and will distinguish it from full-rank coordinate changes in the method description. In this formulation, the Gaussian process (GP) conditional sample $y_0 \sim \mathcal{N}(\mu, \Sigma)$ is preserved at full resolution, while the MLP velocity field operates strictly within the top-$k$ eigenvectors $\Phi_k \in \mathbb{R}^{L_f \times k}$ ($k \ll L_f$) of the conditional horizon covariance $\boldsymbol{\Sigma}$:
$$\hat{y}_f = y_0 + \Phi_k (\hat{c}_1 - c_0).$$
Unlike a full-rank rotation ($k=L_f$), truncating to $k \ll L_f$ restricts the expressive subspace of the flow correction and acts as a spectral regularizer.

Here $\mu$ and $\Sigma$ are the conditional mean and covariance of the future, $L_f$ is the forecast horizon length, and $k$ is the retained correction rank. The projected source is $c_0=\Phi_k^\top y_0$, and $\hat c_1$ is its learned endpoint. The network is a multilayer perceptron (MLP).

We compare the Mercer covariance eigenbasis, principal component analysis (PCA) estimated from data, the discrete cosine transform (DCT), a Fourier basis, and Raw time coordinates. We report the continuous ranked probability score (CRPS), which measures probabilistic forecast error; lower is better. Where percentages are shown, the fraction of the prior-to-oracle gap closed is (prior CRPS − model CRPS) / (prior CRPS − oracle CRPS), using the synthetic benchmark's oracle reference score.

Table 1: Effect of truncation rank $k$ on non-stationary modulated periodic data ($L_f=32$, GP Prior CRPS = 0.2565, Oracle = 0.0200).

| Basis | $k=4$ CRPS $\downarrow$ | $k=6$ CRPS $\downarrow$ | $k=10$ CRPS $\downarrow$ (Gap $\uparrow$) | $k=32$ (Full) CRPS $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: |
| Mercer (Ours) | 0.1926 | 0.1619 | 0.0768 (76.0%) | 0.0531 |
| PCA | 0.1959 | 0.1660 | 0.0884 (71.1%) | 0.0547 |
| DCT | 0.2385 | 0.2315 | 0.2140 (18.0%) | 0.0528 |
| Fourier | 0.2421 | 0.2304 | 0.2167 (16.8%) | 0.0543 |
| Raw | 0.2411 | 0.2331 | 0.2180 (16.3%) | 0.0546 |

Full-rank CRPS values are close in this experiment. At reduced rank, Mercer reduces CRPS by nearly $3\times$ relative to DCT, Fourier, and Raw at $k=10$ (`0.0768` vs. `0.2140–0.2180`). This is where basis choice changes the learnable correction subspace.

---

## Part 2/4: Basis estimation, optimizer analysis, and reporting

This part addresses basis estimation, regularization, the corrected optimizer analysis, and effect-size reporting.

### W1: Basis estimation and regularization

We added two controlled studies to examine when a specified-prior basis is useful. In the calibration study, we held the MLP training set fixed (512 training samples, $k=8$) and changed only the sample used to fit PCA. Mercer had lower CRPS when PCA calibration was scarce or drawn from a shifted historical regime.

In the smooth-kernel study, we compared rank-6 and full-rank corrections at a training sample size of $N=64$. Six modes capture more than 99.6% of the prior variance for the tested radial basis function (RBF) and Matérn-5/2 kernels. The reduced-rank correction had lower CRPS, illustrating a regularization benefit in this small-sample setting.

Table 2: Absolute CRPS effects in the controlled studies (negative differences favor the first method).

| Setting | Comparison | CRPS difference |
|---|---|---:|
| Periodic, rank 10 | Mercer − Fourier | −0.1399 |
| Periodic, rank 10 | Mercer − PCA | −0.0116 |
| Scarce calibration | Mercer − PCA | −0.0137 |
| Shifted calibration | Mercer − PCA | −0.0359 |
| RBF, N=64 | Rank 6 − full rank | −0.0313 |

These results identify two benefits: selecting useful correction directions at low rank, and reducing sensitivity to scarce or shifted PCA calibration data. A specified-prior basis requires no separate basis-calibration sample after the kernel is chosen; its covariance construction and eigendecomposition still incur computation.

### Q2: Optimizer analysis and conditional interpolants

We have corrected the conditioning analysis and will revise its interpretation to distinguish the surrogate from actual optimizer dynamics. For a fixed covariance $K$ and orthogonal basis $\Phi$, write $A=\Phi^\top K\Phi$ for the covariance in that basis and $\kappa$ for its condition number. The corrected frozen input-curvature surrogate is $S^{-1/4}AS^{-1/4}$, where $S=\operatorname{diag}(A)$. In an eigenbasis the matrix is diagonal, and the scaling takes the square root of each covariance eigenvalue, giving condition number $\sqrt{\kappa(A)}$. Measurements on the flow-matching interpolation between source and target test this geometry in the synthetic training distributions. Actual Adam/AdamW dynamics also depend on residuals, momentum, and evolving second moments, so we use the surrogate to explain a mechanism and direct forecast scores to assess accuracy. The current real-data runs use Adam.

### Q3: Effect sizes and statistical reporting

We will replace threshold-based rankings in the paper with absolute scores and effect sizes, as reported in the tables above. The real-data results have one training seed per task and describe the observed benchmark effects. The controlled studies show where representation produces a large difference; the full-rank real-data controls show where its effect is small.

---

## Part 3/4: Real-data benchmark and efficiency attribution

This part clarifies backbone attribution and presents the 54-task GIFT-Eval accuracy and computation comparison.

### W2 / Q1: Efficiency attribution and real-data comparisons

We agree that the $4.7\times$ memory and $4\times$ runtime improvements over TSFlow arise from replacing S4 with an MLP. We will rewrite the corresponding efficiency discussion to attribute these gains to the backbone change. We now present matched corrections in the original time coordinates and the prior's eigenbasis, using the same MLP and training recipe to isolate representation effects, and include TiDE as a direct-forecast MLP reference.

We agree that a smaller MLP alone leaves the contribution of representation unresolved. Your questions about absolute gains and matched controls motivated a broader evaluation on all 54 GIFT-Eval tasks whose datasets contain single-channel series, using official splits, forecast origins, and scoring. We first use matched controls to identify the sources of improvement, then compare accuracy and training computation with external forecasters.

### Prior and representation controls

To identify what each component contributes, we compare three controls: GP-only is the fitted statistical forecast without a neural correction; Raw-GP uses the same Gaussian starting forecast and correction network in the original time coordinates; Raw-noise uses that network in the original coordinates but starts from independent Gaussian noise. These controls share the neural training recipe wherever a network is used.

The matched controls below use the same 54 tasks and equal-weight geometric mean wQL.

| Method | Geom. mean wQL ↓ |
|---|---:|
| GP-only (statistical forecast) | 0.163321 |
| Raw-noise (correction from independent noise) | 0.156343 |
| Raw-GP (prior + correction in time coordinates) | 0.141639 |
| Ours (prior + correction in its eigenbasis) | 0.140626 |

These controls explain where the gain comes from: correction improves wQL by 13.90% over GP-only and 10.05% over Raw-noise, while the full-rank eigenbasis improves Raw-GP by 0.72%. Thus the real-data evidence favors lightweight correction of a fitted prior, with the synthetic studies identifying when basis choice contributes more strongly.

### Broader forecasting comparison

The benchmark includes two complementary reference groups: models trained separately on each dataset—the linear DLinear model, the direct-forecast MLP TiDE, the patch-based transformer PatchTST, and the Temporal Fusion Transformer (TFT)—and pretrained probabilistic models used without dataset-specific training (zero-shot)—Chronos-2, TiRex, and TimesFM-2.5. Our real-data model fits an empirical Gaussian prior and learns a correction in its covariance eigenbasis.

We report weighted quantile loss (wQL) for probabilistic forecasts, mean absolute scaled error (MASE) for median forecasts, and millions of floating-point operations (MFLOPs) per training sample. Lower values are better for all three.

| Method | Geom. mean wQL ↓ | Geom. mean MASE ↓ | Geom. mean training MFLOPs/sample ↓ |
|---|---:|---:|---:|
| Ours (empirical eigenbasis + GP) | 0.1406 | 1.1025 | 2.763 |
| DLinear | 0.2054 | 1.3466 | 1.456 |
| TiDE | 0.1751 | 1.3677 | 0.341 |
| PatchTST | 0.1449 | 1.1488 | 11.653 |
| TFT | 0.1420 | 1.1521 | 71.051 |
| Chronos-2 (zero-shot) | 0.1189 | 0.9829 | — |
| TiRex (zero-shot) | 0.1205 | 0.9936 | — |
| TimesFM-2.5 (zero-shot) | 0.1207 | 0.9895 | — |

All columns are geometric means over the same 54 tasks, with equal task weights. wQL measures probabilistic forecast error; MASE measures median-forecast error. Our results use one training seed, validation-selected contexts/checkpoints, 20 velocity-field evaluations per forecast trajectory, and 100 forecast draws; external accuracy scores are published GIFT-Eval results.

Training MFLOPs count forward, loss, and backward per univariate context/target window at batch one, including our prior conditioning/sampling and coordinate transforms; optimizer updates and one-time preparation are excluded. Baseline counts use profiled reference architectures and contexts, while accuracy uses published evaluation recipes. Pretrained models require no downstream training, but rely on substantial computation already invested in pretraining. Their training-operation entries are left blank because the table measures task-specific training cost.

The main practical result is that a strong statistical forecast lets a small supervised correction achieve competitive accuracy at substantially lower training cost than the larger task-trained models profiled here. The aggregate accuracy advantages over PatchTST and TFT are modest, and performance varies across tasks. The stronger evidence for the design comes from improving the statistical forecast itself, as the component controls show. The pretrained models provide a complementary reference with stronger accuracy and no downstream training.

---

## Part 4/4: Scope of the optimization comparison

This part clarifies what the shared training recipe establishes.

For the tuning question, these controls measure a shared optimization recipe (Adam learning rate 0.001) and matched budget. The synthetic rank comparisons above isolate the larger representation effects that appear when the correction subspace is restricted.
