---
title: answer
order: 2
---

## Part 1/4: Positioning and optimizer analysis

We thank you for identifying the error in our Adam analysis and asking us to separate the role of the representation from the efficiency of the neural model. Your comments have helped us refine the paper's positioning. TSFlow focuses on matching the prior to the temporal structure of the data. We ask whether the prior is a good probabilistic forecaster in its own right. When it is, how much can a very small neural network gain from the predictive information it already contains? We build on TSFlow's use of Gaussian process priors in flow matching to learn a small correction to that forecast. The goal is to make the most of that starting forecast with minimal training cost. For real data, we fit an empirical Gaussian prior following Lin et al., *Empirical Gaussian Processes* (ICML 2026; https://arxiv.org/abs/2602.12082).

Our results show that a small learned correction improves the statistical forecast and reaches accuracy competitive with larger task-trained forecasters, at substantially lower training cost in the profiled implementations.

We first correct the optimizer analysis, then examine when the choice of correction directions matters. The original paper studied optimization and stability. New truncation experiments explore whether the prior can also guide a smaller correction. Finally, the forecasting comparisons and matched controls show what the prior, correction, and neural architecture each contribute.

### W1 / Q1: Adam conditioning analysis

Let $K$ be a fixed positive-definite covariance and $\Phi$ an orthogonal basis. Then $A=\Phi^\top K\Phi$ is the covariance in that basis. We use $\kappa$ for the condition number and $\operatorname{diag}(A)$ for the diagonal matrix formed from $A$. The retained eigenvalues are $\lambda_1\geq\cdots\geq\lambda_k$.

The original analysis applied the second-moment scaling incorrectly. Thank you for catching this error. We have corrected the analysis and will replace Proposition 2 and its interpretation. Adam divides gradients by the square root of the second moment. Under the Kronecker model (Balles & Hennig, 2018), the resulting matrix is $M_{\text{Adam}} = D^{1/2} A D^{1/2}$, where $D = \operatorname{diag}(A)^{-1/2}$. The original analysis used $DAD$.
We use this matrix as a frozen input-curvature surrogate. Actual Adam also depends on residuals, momentum, and evolving second moments. In the eigenbasis, the corrected surrogate has condition number $\sqrt{\kappa(A)}$.

In time coordinates, the covariance can retain strong off-diagonal dependencies. In the Mercer basis, the prior covariance is diagonal. However, this does not imply that AdamW achieves optimal conditioning, especially on the conditional training interpolants.

In the Mercer basis, $\Phi^\top K\Phi=\Lambda$ is diagonal. The corrected scaling gives condition number $\sqrt{\kappa(A)}$ and zero off-diagonal prior covariance. If only $k$ directions are retained, the condition number is $\sqrt{\lambda_1/\lambda_k}$. The benefit therefore depends on the retained spectrum. Whitening also rescales the directions. It changes their loss weights unless the loss is adjusted.

We compare the prior's eigenbasis (Mercer) with a basis estimated from data (PCA), two fixed bases (DCT and Fourier), and the original time coordinates (Raw). The representation acts on the forecast horizon of length $L_f$. Here $c_0$ is the starting sample in the chosen coordinates, and $c_t$ is its flow-matching interpolation with the target. OffDiag measures off-diagonal covariance coupling; zero means the covariance is diagonal.

Table 1: First-layer interpolant geometry ($c_t$) and prior decorrelation ($c_0$) on Horizon-Only Matérn-3/2 ($L_f=32$).

| Basis | Prior OffDiag ($c_0$) $\downarrow$ | Interpolant OffDiag ($c_t$) $\downarrow$ | $\kappa(DAD)$ (Train / Test) $\downarrow$ | $\kappa(D^{1/2}AD^{1/2})$ (Train / Test) $\downarrow$ |
|:---|:---:|:---:|:---:|:---:|
| Mercer (Ours) | 0\.000 | 0\.087 | 3\.1 / 4.4 | 250 / 320 |
| PCA | 0\.181 | 0\.101 | 48\.0 / 49.0 | 330 / 450 |
| DCT | 0\.524 | 0\.538 | 640 / 870 | 900 / 1,100 |
| Fourier | 0\.545 | 0\.561 | 4,500 / 5,500 | 2,900 / 3,800 |
| Raw | 0\.966 | 0\.970 | 58,000 / 68,000 | 46,000 / 56,000 |

With the corrected scaling, Mercer reduces the surrogate's condition number by two orders of magnitude relative to Raw in this experiment. It also removes off-diagonal covariance coupling in the prior.

---

## Part 2/4: Synthetic representation experiments

### Q2: Conditional-horizon basis and low-rank correction

Let $p$ denote the observed past and $f$ the future. Their covariance blocks are $K_{pp}$ and $K_{ff}$, with cross-covariance $K_{fp}=K_{pf}^\top$. The conditional forecast has mean $\mu$ and covariance $\Sigma$.

Following your suggestion, we use the eigenbasis of the conditional future covariance in the synthetic low-rank experiments:
$\boldsymbol{\Sigma} = K_{ff} - K_{fp}K_{pp}^{-1}K_{pf} = \Phi \Lambda \Phi^\top \in \mathbb{R}^{L_f \times L_f}$.
The starting forecast remains a full-rank draw $y_0\sim\mathcal{N}(\mu,\Sigma)$. The multilayer perceptron (MLP) receives the history and prior mean $[y_p,\mu]$ through a separate conditioning branch. It learns a correction in only the top $k$ directions:

$$\hat{y}_f = y_0 + \Phi_k(\hat{c}_1 - c_0), \quad \Phi_k \in \mathbb{R}^{L_f \times k}.$$

Here $y_p$ is the observed history, and $\Phi_k$ contains the $k$ retained eigenvectors. The starting coordinates are $c_0=\Phi_k^\top y_0$. The network learns their endpoint $\hat c_1$.

At full rank ($k=L_f$), the orthonormal bases allow the same corrections and perform similarly. With fewer directions ($k\ll L_f$), the choice of basis determines what the model can correct. The differences become large in the tested example that departs from stationary Ornstein–Uhlenbeck (OU) structure.

We report the continuous ranked probability score (CRPS), which measures probabilistic forecast error. Lower is better. Percentages show how much of the prior-to-oracle gap the correction closes: (prior CRPS − model CRPS) / (prior CRPS − oracle CRPS). The oracle score is the synthetic benchmark's reference.

Table 2: Controlled truncation study on non-stationary modulated periodic data ($L_f=32, p=5.5$, GP Prior CRPS = 0.2565, Oracle CRPS = 0.0200).

| Basis | $k=6$ CRPS $\downarrow$ (Gap $\uparrow$) | $k=10$ CRPS $\downarrow$ (Gap $\uparrow$) | $k=16$ CRPS $\downarrow$ (Gap $\uparrow$) | $k=32$ (Full) CRPS $\downarrow$ (Gap $\uparrow$) |
|:---|:---:|:---:|:---:|:---:|
| Mercer (Ours) | 0\.1619 (40.0%) | 0\.0768 (76.0%) | 0\.0597 (83.2%) | 0\.0531 (86.0%) |
| PCA | 0\.1660 (38.3%) | 0\.0884 (71.1%) | 0\.0613 (82.5%) | 0\.0547 (85.3%) |
| DCT | 0\.2315 (10.6%) | 0\.2140 (18.0%) | 0\.1834 (30.9%) | 0\.0528 (86.1%) |
| Fourier | 0\.2304 (11.0%) | 0\.2167 (16.8%) | 0\.1839 (30.7%) | 0\.0543 (85.5%) |
| Raw | 0\.2331 (9.9%) | 0\.2180 (16.3%) | 0\.1819 (31.6%) | 0\.0546 (85.4%) |

At full rank ($k=32$), forecast errors are close across bases. With only ten correction directions, Mercer closes 76% of the prior-to-oracle gap. DCT closes 18% and Fourier 17%. The basis matters because it determines which corrections the smaller model can learn.

### W2 / Q4: OU, periodicity, and nonstationarity

OU is a weak setting for separating Mercer from DCT. We therefore added the nonstationary periodic experiment in Table 2. Its kernel is $K(s,t)=(0.5+c\min(s,t))K_{\text{Per}}(s,t)$, with period $p=5.5$ and horizon length 32. This tests whether the prior's directions remain useful for correction when the covariance departs from stationary OU structure.

We will revise the periodic-basis discussion to match this evidence. Fourier truncation depends on the chosen period and frequencies. This experiment shows the value of adapting to the specified covariance. The real-data benchmark below tests the broader value of correcting a statistical forecast.

---

## Part 3/4: Real-data benchmark and computation

### W3 / Q3: Backbone attribution and forecasting comparisons

The $4.7\times$ memory and $4\times$ speed gains over TSFlow come from replacing S4 with an MLP. We will attribute them to that architecture change. To measure the effect of coordinates separately, we use the same MLP and training recipe in time coordinates and in the prior's eigenbasis. Table 2 also compares corrections restricted to fewer directions. These controls distinguish a small full-rank coordinate effect from the larger effect of choosing which directions to correct.

The original evaluation mixed architecture and basis effects and offered limited probabilistic comparisons. We now cover all 54 GIFT-Eval tasks with single-channel series, spanning datasets, frequencies, and short, medium, and long horizons. All comparisons use official splits, forecast origins, and scoring.

### Prior and representation controls

We compare the statistical forecast alone with three learned models. GP-only is the fitted forecast without a correction. Raw-GP adds a correction in the original time coordinates. It uses the same Gaussian starting forecast and correction network as our model. Raw-noise uses that network in time coordinates but starts from independent Gaussian noise. Our model starts from the GP forecast and learns the correction in its eigenbasis. All neural models use the same training recipe.

The controls use the same 54 tasks and equal-weight geometric mean wQL.

| Method | Geom. mean wQL ↓ |
|---|---:|
| GP-only (statistical forecast) | 0\.163321 |
| Raw-noise (correction from independent noise) | 0\.156343 |
| Raw-GP (prior + correction in time coordinates) | 0\.141639 |
| Ours (prior + correction in its eigenbasis) | 0\.140626 |
| Ours, half input/output ranks (width 192) | 0\.142803 |

Halving the input and correction output ranks at width 192 increases aggregate wQL by only 1.55%, preserving most of the full-rank accuracy.

The main gain comes from learning a correction to a useful forecast. The correction improves on the GP alone, and starting from that forecast also improves on starting from independent noise. Changing to the prior's eigenbasis gives a smaller additional gain. The synthetic studies examine settings where the choice of correction directions matters more.

### Broader forecasting comparison

We compare against two groups of forecasters. DLinear, TiDE, PatchTST, and TFT are trained separately on each dataset. DLinear is linear, TiDE is a direct-forecast MLP, and PatchTST is a patch-based transformer. TFT is the Temporal Fusion Transformer. Chronos-2, TiRex, and TimesFM-2.5 are pretrained probabilistic models used without dataset-specific training (zero-shot). Our model fits an empirical Gaussian forecast to the training data, then learns a correction in its covariance eigenbasis.

We report weighted quantile loss (wQL) for probabilistic forecasts and mean absolute scaled error (MASE) for forecast medians. Training cost is measured in millions of floating-point operations (MFLOPs) per sample. Lower is better for all three.

| Method | Geom. mean wQL ↓ | Geom. mean MASE ↓ | Geom. mean training MFLOPs/sample ↓ |
|---|---:|---:|---:|
| Ours (empirical eigenbasis + GP) | 0\.1406 | 1\.1025 | 2\.763 |
| TSFlow (locally reproduced) | 0\.1578 | 1\.1926 | 663\.046 (estimated) |
| DLinear | 0\.2054 | 1\.3466 | 1\.456 |
| TiDE | 0\.1751 | 1\.3677 | 0\.341 |
| PatchTST | 0\.1449 | 1\.1488 | 11\.653 |
| TFT | 0\.1420 | 1\.1521 | 71\.051 |
| Chronos-2 (zero-shot) | 0\.1189 | 0\.9829 | -- |
| TiRex (zero-shot) | 0\.1205 | 0\.9936 | -- |
| TimesFM-2.5 (zero-shot) | 0\.1207 | 0\.9895 | -- |

All columns are geometric means over the same 54 tasks, with equal weight per task. Our results use one training seed. Context lengths and checkpoints are selected on validation data. Our forecasts use 20 velocity-field evaluations per trajectory and 100 draws. TSFlow is locally reproduced on all 54 tasks using seed 6432: we select among OU, squared-exponential, and periodic kernels by the best validation wQL through epoch 40, then continue the selected kernel to 400 total epochs. Its model, optimizer, EMA, gradient clipping, normalization, and sampling settings follow the original TSFlow recipe; early kernel selection and single-seed evaluation define this campaign's budget. The other external accuracy scores come from the published GIFT-Eval results.

Our correction achieves 10.86% lower aggregate wQL and 7.55% lower aggregate MASE than this TSFlow reproduction on the complete shared panel.

Training profiles cover a forward pass, loss computation, and backward pass for one univariate window at batch size one. They include our prior conditioning, sampling, and coordinate transforms. TSFlow uses its final selected kernels and native training loss, including GP conditioning and factorization; S4 arithmetic uses a dense Cauchy reference and analytical FFT estimates. Shared S4 kernel generation and GP factorization are counted once per window at batch one and amortize over larger batches. Optimizer updates, clipping, EMA, validation, and one-time preparation are excluded. Other task-trained baseline costs use profiled reference architectures and context lengths. Accuracy scores follow the published evaluation recipes. The pretrained models require no task-specific training, so their entries are blank in this column. They rely on substantial computation already invested in pretraining. That cost is separate from the task-specific training measured here.

A strong statistical forecast lets a small learned correction reach accuracy competitive with PatchTST and TFT at substantially lower profiled training cost. The aggregate accuracy gains are modest, and performance varies across tasks. The component controls give the clearest evidence for the approach: the correction improves the starting forecast itself. The pretrained models achieve stronger accuracy and require no downstream training.

---

## Part 4/4: History conditioning and related work

### Q2: Historical information under the prior

The correction has direct access to history. In the synthetic horizon-only model, the MLP receives both the observed history $y_p$ and the prior mean $\mu$. It can therefore learn dependencies that the prior misses.

For an exactly observed latent OU process, the last state is sufficient for the prior's conditional forecast. The correction receives the full history and can learn dependencies beyond this summary. The real-data controls show the value of adding a correction: it improves aggregate wQL by 13.90% over the fitted empirical Gaussian forecast.

### Related work

Thank you for these connections. We will add both references and clarify how our work builds on them. Functional Flow Matching (Kerrigan et al., 2024) provides a function-space framework for learning transport from a Gaussian prior. We use this perspective for conditional forecasting, with a small learned correction to a statistical forecast. Blurring Diffusion Models (Hoogeboom & Salimans, 2023) shows how structured noise can encode useful assumptions for images. TSFlow applies the corresponding idea to time series through Gaussian process priors. We build on this foundation by studying how to make the most of the starting forecast with a small correction, and how its coordinates affect learning.
