---
title: answer
order: 2
---

## Part 1/4: Positioning and non-Gaussian experiments

We thank you for asking for broader evidence, probabilistic comparisons, and a clearer account of kernel and representation choices. Your comments have helped us refine the paper's positioning. TSFlow focuses on matching the prior to the temporal structure of the data. We ask whether the prior is a good probabilistic forecaster in its own right. When it is, how much can a very small neural network gain from the predictive information it already contains? We build on TSFlow's use of Gaussian process priors in flow matching to learn a small correction to that forecast. The goal is to make the most of that starting forecast with minimal training cost. Our empirical Gaussian prior follows Lin et al., *Empirical Gaussian Processes* (ICML 2026; https://arxiv.org/abs/2602.12082).

Our results show that a small learned correction improves the statistical forecast and reaches accuracy competitive with larger task-trained forecasters, at substantially lower training cost in the profiled implementations.

The controlled examples test whether the correction can improve a Gaussian forecast when the data are non-Gaussian or nonstationary. The kernel studies examine which correction directions are useful. New truncation experiments extend the original focus on optimization and stability to smaller corrections. The real-data benchmark and matched controls measure forecast quality and training cost.

### W1 / Q1: Non-Gaussian dynamics and evaluation scope

We added two controlled examples with strongly non-Gaussian dynamics. The first is a bimodal Double-Well Langevin stochastic differential equation (SDE). Trajectories move toward either of two wells at $\pm1$, while the Gaussian prior predicts their unstable mean at zero:
$dy_t=3(y_t-y_t^3)dt+\sigma dW_t$.
The second is a nonstationary periodic system with a quadratic distortion:
$y_t=x_t+a(t)(x_t^2-K_{tt})+\epsilon_t$.
These examples test whether the correction can recover forecast behavior that the Gaussian starting point misses.

Mercer uses the conditional prior's covariance eigenvectors. Principal component analysis (PCA) estimates directions from data. The discrete cosine transform (DCT) and Fourier transform use fixed directions. Raw uses time coordinates. The GP draw stays at full resolution. Only the learned correction is restricted to $k$ directions over a horizon of length $L_f$.

In the equations, $W_t$ is Brownian motion and $\sigma$ controls its strength. The latent process $x_t$ is Gaussian with covariance $K$. The function $a(t)$ controls the quadratic distortion, and $\epsilon_t$ is observation noise.

We report the continuous ranked probability score (CRPS), which measures probabilistic forecast error. Lower is better. Percentages show how much of the prior-to-oracle gap the correction closes: (prior CRPS − model CRPS) / (prior CRPS − oracle CRPS). The oracle score is the synthetic benchmark's reference.

Table 1: Performance on non-Gaussian synthetic dynamical systems ($L_f=32$; CRPS $\downarrow$ and % of Prior-to-Oracle Gap Closed $\uparrow$).

| System / Setting | GP Prior CRPS | Oracle CRPS | Mercer (Ours) | PCA | DCT | Fourier | Raw |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Bimodal Double-Well SDE ($k=12$, high learning rate) | 0\.3647 | 0\.2349 | 0\.2489 (89.2%) | 0\.2493 (88.9%) | 0\.2496 (88.7%) | 0\.2511 (87.5%) | 0\.3647 (0.0%) |
| Modulated Nonlinear Periodic ($k=10$) | 0\.2565 | 0\.0200 | 0\.0768 (76.0%) | 0\.0884 (71.1%) | 0\.2140 (18.0%) | 0\.2167 (16.8%) | 0\.2180 (16.3%) |

On the bimodal Double-Well SDE, Mercer, PCA, DCT, and Fourier achieve similar CRPS, with Mercer obtaining the lowest numerical value (0.2489). All four representations substantially improve the GP prior, closing approximately 88–89% of the prior-to-oracle gap. Raw shows no improvement under the high learning rate used in this experiment.

In contrast, the nonlinear periodic example shows much larger differences between representations. At k=10, Mercer achieves a CRPS of 0.0768, compared with 0.2140 for DCT and 0.2167 for Fourier.

These results suggest that the benefit of the Mercer basis depends on the structure of the dynamics and the selected correction subspace.

---

## Part 2/4: Kernel choice and learned representations

### W3 / Q1 / Q3: Kernel choice and smoothness

We added radial basis function (RBF) and Matérn-5/2 kernels to test smoother priors. We vary training sample size ($N\in\{64,512\}$) and correction rank ($k=6$ or full rank $k=32$). We also test whitening, which divides each coordinate by the square root of its covariance eigenvalue. This amplifies directions with small prior variance.

For these smooth kernels, six directions capture almost all conditional future variance: 99.97% for RBF and 99.67% for Matérn-5/2. The remaining 26 directions account for only a small fraction of the total prior variance.

Table 2: Smooth kernel ablation (`RBF` and `Matérn-5/2`, $L_f=32$; CRPS $\downarrow$ and % Oracle Gap Closed $\uparrow$).

| Kernel & Sample Size | Top-6 $\Sigma$ Energy | Mercer ($k=6$) | PCA ($k=6$) | DCT ($k=6$) | Raw ($k=6$) | Mercer ($k=32$ Full) | Mercer ($k=32$ Whitened) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RBF ($N=64$) | 99\.97% | 0\.3932 (+27.4%) | 0\.4078 (+24.3%) | 0\.4167 (+22.4%) | 0\.5173 (+0.9%) | 0\.4245 (+20.7%) | 0\.4852 (+7.7%) |
| RBF ($N=512$) | 99\.97% | 0\.3457 (+37.6%) | 0\.3494 (+36.8%) | 0\.3477 (+37.2%) | 0\.5095 (+2.5%) | 0\.3495 (+36.8%) | 0\.3618 (+34.1%) |
| Matérn-5/2 ($N=64$) | 99\.67% | 0\.3726 (+2.3%) | 0\.3750 (+1.5%) | 0\.3821 (-0.6%) | 0\.3817 (-0.5%) | 0\.3856 (-1.7%) | 0\.4291 (-15.1%) |
| Matérn-5/2 ($N=512$) | 99\.67% | 0\.3303 (+15.2%) | 0\.3321 (+14.7%) | 0\.3321 (+14.7%) | 0\.3758 (+1.3%) | 0\.3339 (+14.1%) | 0\.3331 (+14.4%) |

A smaller correction helps most in the small-sample RBF setting. Six directions retain almost all prior variance and give lower forecast error than the full-rank correction. The differences shrink at $N=512$. The benefit therefore depends on the sample size and covariance spectrum. Forecast scores test whether these directions are useful for correction, beyond how well they reconstruct the covariance.

Together with the Ornstein–Uhlenbeck (OU) kernel and the modulated periodic example, these studies cover stationary, smooth, and nonstationary priors. The correction learns the non-Gaussian target behavior. Signature and sequential kernels offer directions for future work.

### W4: Learned latent representations

Our representation describes a correction to a forecast. The model keeps the full-resolution GP draw and adds a correction in selected directions. Variation outside those directions is preserved. Autoencoder-based latent flow matching instead learns an encoder and decoder for generation. We will make this distinction clearer.

Once a kernel and its parameters are specified, the Mercer basis needs no separate calibration sample. Computing the covariance and its eigendecomposition still has a cost. The empirical Gaussian basis in our real-data study is fitted from training data. In the synthetic study, we isolate basis estimation by fixing the multilayer perceptron (MLP) training set ($N=512,k=8$). We change only the separate calibration sample used to estimate PCA's directions.

Table 3: Robustness to basis-estimation scarcity and historical regime shift ($k=8$, CRPS $\downarrow$ and Gap $\uparrow$).

| Basis ($k=8$) | Basis Calibration Cost | Small Calibration ($N_{\text{basis}}=10$) | Matched Calibration ($N_{\text{basis}}=128$) | Shifted Historical Regime |
|:---|:---:|:---:|:---:|:---:|
| Mercer (Ours) | 0 separate calibration samples | 0\.1250 (55.6%) | 0\.1250 (55.6%) | 0\.1250 (55.6%) |
| PCA (Standard) | Empirical Cov($y_f$) | 0\.1387 (49.8%) | 0\.1304 (53.3%) | 0\.1608 (40.4%) |
| DCT / Fourier | 0 samples (Fixed) | 0\.2156 / 0.2184 (17%) | 0\.2156 / 0.2184 (17%) | 0\.2156 / 0.2184 (17%) |

PCA's forecast quality depends on its calibration data. Scarce or shifted calibration worsens its results even though the correction's training set stays fixed. The specified-prior basis stays unchanged. This isolates the effect of estimating the basis.

---

## Part 3/4: Real-data benchmark and computation

### W2 / W6: Probabilistic baselines and numerical results

The original evaluation covered five datasets and summarized results by rank. Your requests for broader coverage, probabilistic baselines, and absolute scores motivated a larger evaluation. We now cover all 54 GIFT-Eval tasks whose datasets contain single-channel series. These span datasets, frequencies, and forecast horizons. We use official splits and forecast origins, and report probabilistic accuracy, point-forecast accuracy, and training cost.

We compare against two groups of forecasters. DLinear, TiDE, PatchTST, and TFT are trained separately on each dataset. DLinear is linear, TiDE is a direct-forecast MLP, and PatchTST is a patch-based transformer. TFT is the Temporal Fusion Transformer. Chronos-2, TiRex, and TimesFM-2.5 are pretrained probabilistic models used without dataset-specific training (zero-shot). Our model fits an empirical Gaussian forecast to the training data, then learns a correction in its covariance eigenbasis.

We report weighted quantile loss (wQL) for probabilistic forecasts and mean absolute scaled error (MASE) for forecast medians. Training cost is measured in millions of floating-point operations (MFLOPs) per sample. Lower is better for all three.

| Method | Geom. mean wQL ↓ | Geom. mean MASE ↓ | Geom. mean training MFLOPs/sample ↓ |
|---|---:|---:|---:|
| MercerFlow (empirical eigenbasis + GP) | 0\.1406 | 1\.1025 | 2\.763 |
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

## Part 4/4: Controls, computational scaling, and channels

### Prior and representation controls

We compare the statistical forecast alone with three learned models. GP-only is the fitted forecast without a correction. Raw-GP adds a correction in the original time coordinates. It uses the same Gaussian starting forecast and correction network as our model. Raw-noise uses that network in time coordinates but starts from independent Gaussian noise. Our model starts from the GP forecast and learns the correction in its eigenbasis. All neural models use the same training recipe.

The matched controls below use the same 54 tasks and equal-weight geometric mean wQL.

| Method | Geom. mean wQL ↓ |
|---|---:|
| GP-only (statistical forecast) | 0\.163321 |
| Raw-noise (correction from independent noise) | 0\.156343 |
| Raw-GP (prior + correction in time coordinates) | 0\.141639 |
| MercerFlow (prior + correction in its eigenbasis) | 0\.140626 |
| MercerFlow, half input/output ranks (same width) | 0\.142803 |

Halving both the input and correction output ranks while keeping the same network width increases aggregate wQL by only 1.55%, preserving most of the full-rank model's accuracy.

The main gain comes from learning a correction to a useful forecast. The correction improves on the GP alone, and starting from that forecast also improves on starting from independent noise. Changing to the prior's eigenbasis gives a smaller additional gain. The synthetic studies examine settings where the choice of correction directions matters more. Our model wins on 23/54 tasks against PatchTST and 21/54 against TFT.

### W5 / Q4: Computational scaling

We will separate the costs of preparing the prior, sampling a forecast, and learning its correction. Let $C$ be the context length and $H$ the forecast horizon. The conditional covariance is prepared once for a given prior, context, and horizon. Dense preparation can take $O((C+H)^3)$ time and $O((C+H)^2)$ storage. A smaller correction reduces the MLP's input and output projection cost. In the synthetic suite, parameters fall from 24.9k to 18.0k. The starting forecast still requires full-rank prior sampling. At fixed hidden width, network and projection costs grow with the retained dimensions. Inference cost also grows with solver steps and forecast draws.

### Q2: Correlated channels

The evaluated model forecasts each channel from its own history, with neural parameters shared across channels. It targets marginal probabilistic forecasts. Joint multichannel forecasting would also require modeling dependence between channels. We will state this scope clearly in the method and experiments.
