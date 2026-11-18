---
title: answer
order: 2
---

## Part 1/4: Positioning and optimizer analysis

This part clarifies our contribution and explains the corrected Adam conditioning analysis.

We thank you for identifying the error in our Adam analysis and asking us to distinguish representation effects from backbone efficiency. The question motivating our work is how much neural computation probabilistic forecasting requires when a statistical model already provides a strong forecast. TSFlow establishes flow matching with Gaussian process priors as a way to integrate statistical structure into neural forecasting. We investigate how far this integration can reduce the neural computation required, using a small MLP correction and studying how coordinates aligned with the prior affect optimization and stability. For the real-data experiments, we fit an empirical Gaussian prior following Lin et al., Empirical Gaussian Processes (ICML 2026; https://arxiv.org/abs/2602.12082).

The evidence below separates the optimization question from the forecasting question. The corrected analysis clarifies how the choice of coordinates affects the proposed conditioning mechanism, while the representation experiments examine when that choice matters for the learned correction. The original paper focused on optimization and stability; the truncation experiments added for this rebuttal explore whether the same statistical structure can also support a smaller correction. The broader forecasting comparisons and component controls then establish the practical value of the lightweight model and distinguish the contributions of the prior, correction, and backbone.

### W1 / Q1: Adam conditioning analysis

Here $K$ is a fixed positive-definite covariance, $\Phi$ is an orthogonal basis, and $A=\Phi^\top K\Phi$ is the covariance in that basis. We write $\kappa$ for the condition number, $\operatorname{diag}(A)$ for the diagonal matrix formed from $A$, and $\lambda_1\geq\cdots\geq\lambda_k$ for the retained covariance eigenvalues.

We agree that the original analysis applied the second-moment scaling incorrectly. We have corrected the analysis and will replace Proposition 2 and its interpretation accordingly. Under the Kronecker model (Balles & Hennig, 2018), Adam scales gradients by the square root of the second moment, making the effective preconditioned matrix $M_{\text{Adam}} = D^{1/2} A D^{1/2}$ (with $D = \text{diag}(A)^{-1/2}$) rather than $D A D$.
We interpret this as a frozen input-curvature surrogate: actual Adam also depends on residuals, momentum, and evolving second moments. The eigenbasis calculation below gives condition number $\sqrt{\kappa(A)}$ for this surrogate.

In raw coordinates ($\Phi = I$), since $\text{diag}(A)$ is constant on a stationary grid, $D \propto I$, so $\kappa(M_{\text{Adam}}) = \kappa(A)$ and off-diagonal coupling remains dense.

In the Mercer basis ($\Phi^\top K \Phi = \Lambda$), the covariance is strictly diagonalized, reducing the condition number to the square root $\kappa(M_{\text{Adam}}) = \sqrt{\kappa(A)}$ while eliminating off-diagonal gradient coupling entirely (zero off-diagonal covariance). On the retained eigenspace, the condition number is $\sqrt{\lambda_1/\lambda_k}$, so the improvement depends on the retained spectrum. Whitening is a separate, nonorthogonal transformation and changes directional loss weights unless the loss is adjusted.

The following experiment compares the Mercer covariance eigenbasis with principal component analysis (PCA), a basis estimated from data; the discrete cosine transform (DCT); a Fourier basis; and Raw, the original time coordinates. Here the horizon-only representation acts on the future to be forecast, with horizon length $L_f$. Here $c_0$ denotes the source in the chosen coordinates, and $c_t$ denotes its flow-matching interpolation with the target at time $t$. OffDiag summarizes off-diagonal covariance coupling, with zero indicating a diagonal covariance.

Table 1: First-layer interpolant geometry ($c_t$) and prior decorrelation ($c_0$) on Horizon-Only Matérn-3/2 ($L_f=32$).

| Basis | Prior OffDiag ($c_0$) $\downarrow$ | Interpolant OffDiag ($c_t$) $\downarrow$ | $\kappa(DAD)$ (Train / Test) $\downarrow$ | $\kappa(D^{1/2}AD^{1/2})$ (Train / Test) $\downarrow$ |
|:---|:---:|:---:|:---:|:---:|
| Mercer (Ours) | 0\.000 | 0\.087 | 3\.1 / 4.4 | 250 / 320 |
| PCA | 0\.181 | 0\.101 | 48\.0 / 49.0 | 330 / 450 |
| DCT | 0\.524 | 0\.538 | 640 / 870 | 900 / 1,100 |
| Fourier | 0\.545 | 0\.561 | 4,500 / 5,500 | 2,900 / 3,800 |
| Raw | 0\.966 | 0\.970 | 58,000 / 68,000 | 46,000 / 56,000 |

Under the corrected surrogate $D^{1/2} A D^{1/2}$ scaling, Mercer reduces the condition number by two orders of magnitude over Raw and eliminates off-diagonal coordinate coupling on the prior.

---

## Part 2/4: Synthetic representation experiments

This part addresses the conditional-horizon basis and tests representation choice under truncation and different kernels.

### Q2: Conditional-horizon basis and low-rank correction

Writing $p$ for past observations and $f$ for future values, $K_{pp}$ and $K_{ff}$ are their covariance blocks and $K_{fp}=K_{pf}^\top$ is their cross-covariance. The conditional Gaussian mean is $\mu$, and its covariance is $\Sigma$.

Following your suggestion in Question 2, our synthetic low-rank variant uses the eigenbasis of the conditional horizon covariance $\boldsymbol{\Sigma} = K_{ff} - K_{fp}K_{pp}^{-1}K_{pf} = \Phi \Lambda \Phi^\top \in \mathbb{R}^{L_f \times L_f}$. The Gaussian process (GP) prior sample $y_0 \sim \mathcal{N}(\mu, \Sigma)$ is kept at full rank, history $[y_p, \mu]$ enters the multilayer perceptron (MLP) via a separate conditioning branch, and the flow learns a low-rank residual correction restricted to the top-$k$ modes:

$$\hat{y}_f = y_0 + \Phi_k(\hat{c}_1 - c_0), \quad \Phi_k \in \mathbb{R}^{L_f \times k}.$$

Here $y_p$ is the observed history, $\Phi_k$ contains the $k$ retained eigenvectors, $c_0=\Phi_k^\top y_0$ contains the initial projected coordinates, and $\hat c_1$ is the learned endpoint in those coordinates.

While full-rank orthonormal bases ($k=L_f$) parameterize the same function class and perform similarly, low-rank truncation ($k \ll L_f$) separates the bases dramatically when the kernel departs from a stationary Ornstein–Uhlenbeck (OU) process.

We report the continuous ranked probability score (CRPS), which measures probabilistic forecast error; lower is better. Where percentages are shown, the fraction of the prior-to-oracle gap closed is (prior CRPS − model CRPS) / (prior CRPS − oracle CRPS), using the synthetic benchmark's oracle reference score.

Table 2: Controlled truncation study on non-stationary modulated periodic data ($L_f=32, p=5.5$, GP Prior CRPS = 0.2565, Oracle CRPS = 0.0200).

| Basis | $k=6$ CRPS $\downarrow$ (Gap $\uparrow$) | $k=10$ CRPS $\downarrow$ (Gap $\uparrow$) | $k=16$ CRPS $\downarrow$ (Gap $\uparrow$) | $k=32$ (Full) CRPS $\downarrow$ (Gap $\uparrow$) |
|:---|:---:|:---:|:---:|:---:|
| Mercer (Ours) | 0\.1619 (40\.0%) | 0\.0768 (76\.0%) | 0\.0597 (83\.2%) | 0\.0531 (86.0%) |
| PCA | 0\.1660 (38.3%) | 0\.0884 (71.1%) | 0\.0613 (82.5%) | 0\.0547 (85.3%) |
| DCT | 0\.2315 (10.6%) | 0\.2140 (18.0%) | 0\.1834 (30.9%) | 0\.0528 (86\.1%) |
| Fourier | 0\.2304 (11.0%) | 0\.2167 (16.8%) | 0\.1839 (30.7%) | 0\.0543 (85.5%) |
| Raw | 0\.2331 (9.9%) | 0\.2180 (16.3%) | 0\.1819 (31.6%) | 0\.0546 (85.4%) |

At full rank ($k=32$), the CRPS values are close; under low-rank truncation ($k=10$), Mercer closes 76% of the prior-to-oracle gap versus 18% for DCT and 17% for Fourier. The basis matters here because it determines which corrections the low-rank model can express.

### W2 / Q4: OU, periodicity, and nonstationarity

We agree that OU is a weak setting for distinguishing Mercer from DCT. To address this, we added the nonstationary periodic experiment in Table 2, using $K(s,t) = (0.5 + c\min(s,t))K_{\text{Per}}(s,t)$ with period $p=5.5$ on a horizon of length 32. This tests whether the basis follows useful correction directions when the covariance departs from stationary OU structure.

We will narrow the periodic-basis discussion accordingly. Fourier truncation depends on the selected period and frequencies, and this experiment supports adaptation to the specified covariance structure. The broader real-data study below tests the practical value of GP correction across datasets.

---

## Part 3/4: Real-data benchmark and computation

This part presents the 54-task GIFT-Eval comparison, including forecast accuracy and training computation.

### W3 / Q3: Backbone attribution and forecasting comparisons

We agree that the $4.7\times$ memory and $4\times$ speed gains over TSFlow come from replacing the sequential S4 backbone with an MLP. We will rewrite the corresponding efficiency discussion to attribute these gains to the backbone change. To isolate the effect of representation, we now present matched corrections in the original time coordinates and the prior's eigenbasis, using the same MLP and training recipe, alongside the low-rank comparisons in Table 2. These experiments distinguish the small full-rank coordinate effect from the role of basis choice in selecting a low-rank correction subspace.

We agree with your concern that the original evaluation mixed backbone and basis effects and lacked a broad probabilistic comparison. In response, we expanded the real-data study to all 54 GIFT-Eval tasks whose datasets contain single-channel series, an established forecasting benchmark spanning datasets, frequencies, and short, medium, and long horizons. Its official splits, forecast origins, and scoring give the comparisons a common evaluation framework.

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

## Part 4/4: History conditioning and related work

This part clarifies access to history and explains the related-work connections.

### Q2: Historical information under the prior

We will clarify the history-conditioning path in the method description. In the synthetic horizon-only architecture, the observed history $y_p$ is fed directly into the MLP alongside the GP mean $\mu$, allowing the network to capture historical dependencies even when the GP kernel has a short memory.

For an exactly observed latent OU process, the Markov property makes the last state sufficient for the prior's conditional future. This limited summary motivates conditioning the MLP correction directly on the full observed history: the prior supplies a statistical starting forecast, and the network can learn historical dependencies beyond those captured by the prior. The real-data control comparison above supports the value of this correction: adding it to the fitted empirical Gaussian forecast reduces aggregate wQL by 13.90% relative to GP-only.

### Related work

Thank you for pointing out these connections. We will add both references and make this conceptual foundation explicit. Functional Flow Matching (Kerrigan et al., 2024) provides the function-space framework for learning transport from a Gaussian prior; our work builds on this perspective for conditional time-series forecasting, learning a lightweight correction to a statistical forecast in the prior's eigenbasis. Blurring Diffusion Models (Hoogeboom & Salimans, 2023) shows how structured, non-isotropic noise introduces an inductive bias suited to images. The corresponding principle for time series is to use a prior that reflects temporal structure, as TSFlow already demonstrates with Gaussian process priors. We build on that foundation by studying the representation of the learned correction and its accuracy and computational cost.
