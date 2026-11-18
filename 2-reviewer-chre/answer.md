---
title: answer
order: 2
---

## Part 1/4: Positioning and non-Gaussian experiments

This part clarifies our contribution and presents the expanded experiments on non-Gaussian dynamics.

We thank you for asking for broader empirical coverage, probabilistic comparisons, and a clearer account of kernel and latent-space choices. The question motivating our work is how much neural computation probabilistic forecasting requires when a statistical model already provides a strong forecast. Building on flow matching with Gaussian process priors, as in TSFlow, we investigate whether that statistical forecast can support a small neural correction with accuracy competitive with larger supervised models, and how the choice of correction coordinates affects optimization and stability. The expanded benchmark and kernel studies examine the scope of this approach. Our empirical Gaussian prior follows the approach of estimating forecast structure from historical data developed in Lin et al., Empirical Gaussian Processes (ICML 2026; https://arxiv.org/abs/2602.12082).

The evidence below examines both the scope of this approach and the choices that make it effective. The controlled dynamical examples help identify when the prior's temporal structure and the representation of its correction are useful, including under non-Gaussian and nonstationary dynamics. The kernel and representation studies examine how the statistical structure guides that correction and when it can support a smaller model; the truncation experiments extend the original paper's focus on optimization and stability and were added for this rebuttal. The broader forecasting comparisons then assess the practical accuracy and computational cost, while the component controls identify what the statistical forecast and learned correction each contribute.

### W1 / Q1: Non-Gaussian dynamics and evaluation scope

To address your request for controlled dynamical examples, we added experiments on two strongly non-Gaussian synthetic dynamical systems: (A) a bimodal Double-Well Langevin stochastic differential equation (SDE) ($dy_t = 3(y_t - y_t^3)dt + \sigma dW_t$, where trajectories stochastically bifurcate into $\pm 1$ wells while the Gaussian prior predicts the unstable mean at $0$), and (B) a non-stationary modulated periodic system with quadratic harmonic skew ($y_t = x_t + a(t)(x_t^2 - K_{tt}) + \epsilon_t$).

In these comparisons, Mercer uses the conditional prior's covariance eigenvectors; principal component analysis (PCA) estimates a basis from data; the discrete cosine transform (DCT) and Fourier transform supply fixed bases; and Raw uses the original time coordinates. The Gaussian process (GP) draw remains full resolution, while the correction uses $k$ basis directions over a forecast horizon of length $L_f$. In the equations above, $W_t$ is Brownian motion, $\sigma$ controls noise strength, $x_t$ is a latent Gaussian process with covariance $K$, and $a(t)$ controls the quadratic distortion, and $\epsilon_t$ is observation noise.

We report the continuous ranked probability score (CRPS), which measures probabilistic forecast error; lower is better. Where percentages are shown, the fraction of the prior-to-oracle gap closed is (prior CRPS − model CRPS) / (prior CRPS − oracle CRPS), using the synthetic benchmark's oracle reference score.

Table 1: Performance on non-Gaussian synthetic dynamical systems ($L_f=32$; CRPS $\downarrow$ and % of Prior-to-Oracle Gap Closed $\uparrow$).

| System / Setting | GP Prior CRPS | Oracle CRPS | Mercer (Ours) | PCA | DCT | Fourier | Raw |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Bimodal Double-Well SDE ($k=12$, high learning rate) | 0\.3647 | 0\.2349 | 0\.2489 (89.2%) | 0\.2493 (88.9%) | 0\.2496 (88.7%) | 0\.2511 (87.5%) | 0\.3647 (0.0%) |
| Modulated Nonlinear Periodic ($k=10$) | 0\.2565 | 0\.0200 | 0\.0768 (76.0%) | 0\.0884 (71.1%) | 0\.2140 (18.0%) | 0\.2167 (16.8%) | 0\.2180 (16.3%) |

Flow matching substantially improves upon the Gaussian prior on bimodal and skewed dynamics (closing up to 89% of the Oracle gap), with the largest basis differences appearing in the nonstationary low-rank example. The double-well Raw failure describes stability at the stated high learning rate.

These controlled dynamics test how a Gaussian source can be corrected toward a non-Gaussian forecast. Our evaluated application is temporal forecasting, with broader real-data coverage provided by 54 GIFT-Eval tasks across datasets, frequencies, and horizons.

---

## Part 2/4: Kernel choice and learned representations

This part addresses kernel choice, smoothness, and comparisons with learned representations.

### W3 / Q1 / Q3: Kernel choice and smoothness

To broaden the kernel study beyond the Ornstein–Uhlenbeck (OU) kernel and examine the role of smoothness, we added radial basis function (RBF) and Matérn-5/2 kernels across training sample sizes ($N \in \{64, 512\}$), ranks ($k=6$ vs. full rank $k=32$), and spectral whitening, which rescales each retained coordinate by the inverse square root of its covariance eigenvalue and amplifies small-variance modes. For smooth kernels on $L_f=32$, eigenvalue decay is rapid: the top 6 modes of the conditional-future covariance $\boldsymbol{\Sigma}$ capture 99.97% of the variance for RBF and 99.67% for Matérn-5/2, leaving 26 near-degenerate tail modes at the noise floor.

Table 2: Smooth kernel ablation (`RBF` and `Matérn-5/2`, $L_f=32$; CRPS $\downarrow$ and % Oracle Gap Closed $\uparrow$).

| Kernel & Sample Size | Top-6 $\Sigma$ Energy | Mercer ($k=6$) | PCA ($k=6$) | DCT ($k=6$) | Raw ($k=6$) | Mercer ($k=32$ Full) | Mercer ($k=32$ Whitened) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RBF ($N=64$) | 99\.97% | 0\.3932 (+27.4%) | 0\.4078 (+24.3%) | 0\.4167 (+22.4%) | 0\.5173 (+0.9%) | 0\.4245 (+20.7%) | 0\.4852 (+7.7%) |
| RBF ($N=512$) | 99\.97% | 0\.3457 (+37.6%) | 0\.3494 (+36.8%) | 0\.3477 (+37.2%) | 0\.5095 (+2.5%) | 0\.3495 (+36.8%) | 0\.3618 (+34.1%) |
| Matérn-5/2 ($N=64$) | 99\.67% | 0\.3726 (+2.3%) | 0\.3750 (+1.5%) | 0\.3821 (-0.6%) | 0\.3817 (-0.5%) | 0\.3856 (-1.7%) | 0\.4291 (-15.1%) |
| Matérn-5/2 ($N=512$) | 99\.67% | 0\.3303 (+15.2%) | 0\.3321 (+14.7%) | 0\.3321 (+14.7%) | 0\.3758 (+1.3%) | 0\.3339 (+14.1%) | 0\.3331 (+14.4%) |

Truncation is most useful in the small-sample RBF setting, where the leading modes retain almost all prior variance and the reduced model has lower forecast error. The smaller differences at N=512 show that the benefit depends on sample size and spectrum. Covariance reconstruction and nonlinear correction are separate objectives; the forecast results test whether the retained modes are useful for correction.

Together with OU and the modulated periodic example, these experiments cover stationary, smooth, and nonstationary priors. Non-Gaussian target behavior is learned by the correction. Signature and sequential kernels are natural extensions.

### W4: Learned latent representations

We will revise the latent-space discussion to explain what the representation parameterizes. Our model preserves a full-resolution GP draw and learns an additive correction in a selected subspace. Autoencoder-based latent flow matching instead learns an encoder and decoder for generation. This distinction preserves prior variation outside the correction subspace.

For a specified kernel and its parameters, the Mercer basis requires no separate calibration sample; it still requires covariance construction and eigendecomposition. The empirical Gaussian basis used in our real-data study is fitted from training data. To isolate basis estimation in the synthetic study, we fixed the multilayer perceptron (MLP) training set ($N=512, k=8$) and varied only the PCA calibration data—the separate sample used to estimate its covariance eigenvectors.

Table 3: Robustness to basis-estimation scarcity and historical regime shift ($k=8$, CRPS $\downarrow$ and Gap $\uparrow$).

| Basis ($k=8$) | Basis Calibration Cost | Small Calibration ($N_{\text{basis}}=10$) | Matched Calibration ($N_{\text{basis}}=128$) | Shifted Historical Regime |
|:---|:---:|:---:|:---:|:---:|
| Mercer (Ours) | 0 separate calibration samples | 0\.1250 (55.6%) | 0\.1250 (55.6%) | 0\.1250 (55.6%) |
| PCA (Standard) | Empirical Cov($y_f$) | 0\.1387 (49.8%) | 0\.1304 (53.3%) | 0\.1608 (40.4%) |
| DCT / Fourier | 0 samples (Fixed) | 0\.2156 / 0.2184 (17%) | 0\.2156 / 0.2184 (17%) | 0\.2156 / 0.2184 (17%) |

PCA quality depends on its calibration data. With the correction's training set fixed, scarce or shifted calibration worsens PCA forecasts, while the specified-prior basis stays unchanged. This experiment isolates sensitivity to basis estimation.

---

## Part 3/4: Real-data benchmark and computation

This part presents the 54-task GIFT-Eval comparison, including forecast accuracy and training computation.

### W2 / W6: Probabilistic baselines and numerical results

We agree that five datasets and a rank-only summary gave too narrow a view of forecasting performance. Your requests for broader coverage, probabilistic baselines, and absolute metrics motivated an expanded study on all 54 GIFT-Eval tasks whose datasets contain single-channel series, spanning datasets, frequencies, and forecast horizons under official splits and forecast origins. We now report both probabilistic and point-forecast accuracy alongside training computation.

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

## Part 4/4: Controls, computational scaling, and channels

This part isolates the prior and correction effects and addresses computational scaling and correlated channels.

### Prior and representation controls

To identify what each component contributes, we compare three controls: GP-only is the fitted statistical forecast without a neural correction; Raw-GP uses the same Gaussian starting forecast and correction network in the original time coordinates; Raw-noise uses that network in the original coordinates but starts from independent Gaussian noise. These controls share the neural training recipe wherever a network is used.

The matched controls below use the same 54 tasks and equal-weight geometric mean wQL.

| Method | Geom. mean wQL ↓ |
|---|---:|
| GP-only (statistical forecast) | 0.163321 |
| Raw-noise (correction from independent noise) | 0.156343 |
| Raw-GP (prior + correction in time coordinates) | 0.141639 |
| Ours (prior + correction in its eigenbasis) | 0.140626 |

These controls explain where the gain comes from: correction improves wQL by 13.90% over GP-only and 10.05% over Raw-noise, while the full-rank eigenbasis improves Raw-GP by 0.72%. Thus the real-data evidence favors lightweight correction of a fitted prior, with the synthetic studies identifying when basis choice contributes more strongly. Performance varies across tasks: the aggregate gains over PatchTST and TFT accompany wins on 23/54 and 21/54 tasks, respectively.

### W5 / Q4: Computational scaling

We will expand the cost discussion to separate prior preparation, sampling, and neural correction. Writing $C$ for context length and $H$ for forecast horizon, the conditional covariance is prepared once for a given prior, context, and horizon. Dense preparation can cost $O((C+H)^3)$ time and $O((C+H)^2)$ storage. Reducing correction rank lowers the MLP's input/output projection cost: in the synthetic suite, parameters fall from 24.9k to 18.0k. Full-rank prior sampling remains part of the forecast cost. With fixed hidden width, projection and network costs grow with the retained dimensions; inference also scales with solver steps and forecast draws.

### Q2: Correlated channels

We will make the channel-independent scope explicit in the method and experimental descriptions. The evaluated model shares neural parameters across channels and conditions each forecast on that channel's history. It targets marginal probabilistic forecasts. Modeling joint multichannel outcomes additionally requires cross-channel dependence.
