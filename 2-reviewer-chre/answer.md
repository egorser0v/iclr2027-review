---
title: answer
order: 2
---

We appreciate your constructive feedback. We have expanded our evaluation to cover nonlinear SDEs, smooth kernel families (`Matérn-5/2`, `RBF`), low-rank truncation, and robustness to limited/shifted basis-calibration data.

### 1\. Non-Gaussian Dynamical Systems / SDE Benchmarks

We evaluated the codecs on two strongly non-Gaussian synthetic dynamical systems: (A) a bimodal **Double-Well Langevin SDE** ($dy_t = 3(y_t - y_t^3)dt + \sigma dW_t$, where trajectories stochastically bifurcate into $\pm 1$ wells while the GP prior predicts the unstable mean at $0$), and (B) a **non-stationary modulated periodic system** with quadratic harmonic skew ($y_t = x_t + a(t)(x_t^2 - K_{tt}) + \epsilon_t$).

**Table 1: Performance on non-Gaussian synthetic dynamical systems ($L_f=32$; CRPS $\downarrow$ and % of Prior-to-Oracle Gap Closed $\uparrow$).**

| System / Setting | GP Prior CRPS | Oracle CRPS | **Mercer (Ours)** | **PCA** | **DCT** | **Fourier** | **Raw** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Bimodal Double-Well SDE** ($k=12$, High LR) | 0\.3647 | 0\.2349 | **0\.2489 (89.2%)** | 0\.2493 (88.9%) | 0\.2496 (88.7%) | 0\.2511 (87.5%) | 0\.3647 (0.0%) |
| **Modulated Nonlinear Periodic** ($k=10$) | 0\.2565 | 0\.0200 | **0\.0768 (76.0%)** | 0\.0884 (71.1%) | 0\.2140 (18.0%) | 0\.2167 (16.8%) | 0\.2180 (16.3%) |

**Takeaway:** Flow matching substantially improves upon the Gaussian prior on bimodal and skewed dynamics (closing up to 89% of the Oracle gap), and Mercer provides the strongest low-rank subspace when the underlying covariance is non-Markovian.



\[TO ADD: Additional spatiotemporal PDE benchmark if completed.\]

---

### 2 & 3. Kernel Ablation (`RBF` and `Matérn-5/2`), Smoothness, and Spectral Truncation

We evaluated `RBF` and `Matérn-5/2` kernels across sample sizes ($N \in \{64, 512\}$), ranks ($k=6$ vs. full rank $k=32$), and spectral whitening ($\Lambda_k^{-1/2}$, which amplifies tail modes). For smooth kernels on $L_f=32$, eigenvalue decay is rapid: the **top 6 modes of $\boldsymbol{\Sigma}$ capture 99.97% of the variance for RBF and 99.67% for Matérn-5/2**, leaving 26 near-degenerate tail modes at the noise floor.

**Table 2: Smooth kernel ablation (`RBF` and `Matérn-5/2`, $L_f=32$; CRPS $\downarrow$ and % Oracle Gap Closed $\uparrow$).**

| Kernel & Sample Size | Top-6 $\Sigma$ Energy | **Mercer ($k=6$)** | **PCA ($k=6$)** | **DCT ($k=6$)** | **Raw ($k=6$)** | **Mercer ($k=32$ Full)** | **Mercer ($k=32$ Whitened)** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RBF ($N=64$)** | 99\.97% | **0\.3932 (+27.4%)** | 0\.4078 (+24.3%) | 0\.4167 (+22.4%) | 0\.5173 (+0.9%) | 0\.4245 (+20.7%) | 0\.4852 (+7.7%) |
| **RBF ($N=512$)** | 99\.97% | **0\.3457 (+37.6%)** | 0\.3494 (+36.8%) | 0\.3477 (+37.2%) | 0\.5095 (+2.5%) | 0\.3495 (+36.8%) | 0\.3618 (+34.1%) |
| **Matérn-5/2 ($N=64$)** | 99\.67% | **0\.3726 (+2.3%)** | 0\.3750 (+1.5%) | 0\.3821 (-0.6%) | 0\.3817 (-0.5%) | 0\.3856 (-1.7%) | 0\.4291 (-15.1%) |
| **Matérn-5/2 ($N=512$)** | 99\.67% | **0\.3303 (+15.2%)** | 0\.3321 (+14.7%) | 0\.3321 (+14.7%) | 0\.3758 (+1.3%) | 0\.3339 (+14.1%) | 0\.3331 (+14.4%) |

**Takeaway:** On smooth kernels, retaining or whitening the 26 near-zero tail modes ($k=32$) causes overfitting in small-sample regimes ($N=64$), whereas truncating Mercer to $k=6$ removes the noise tail and statistically outperforms full-rank $k=32$ (paired 95% CI $[-0.0535, -0.0091]$ on RBF).

---

### 4 & 6. Mercer as an Analytical Zero-Cost Alternative to Learned Latent Spaces / PCA

Unlike autoencoder-based Latent Flow Matching or empirical PCA--which require sufficient, stationary training data to learn a low-dimensional projection--Mercer derives the optimal Karhunen–Loève subspace analytically from $\boldsymbol{\Sigma}$ (**0 training samples, 0 encoder parameters**). To isolate basis quality from MLP training, we fixed the operational MLP training set ($N=512, k=8$) and varied only the calibration data used to fit PCA.

**Table 3: Robustness to basis-estimation scarcity and historical regime shift ($k=8$, CRPS $\downarrow$ and Gap $\uparrow$).**

| Basis ($k=8$) | Basis Calibration Cost | Small Calibration ($N_{\text{basis}}=10$) | Matched Calibration ($N_{\text{basis}}=128$) | **Shifted Historical Regime** | Shift Proj. Loss $\downarrow$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Mercer (Ours)** | **0 samples (Data-Free)** | **0\.1250 (55.6%)** | 0\.1250 (55.6%) | **0\.1250 (55.6%)** | **0\.229** |
| **PCA (Standard)** | Empirical Cov($y_f$) | 0\.1387 (49.8%) | 0\.1304 (53.3%) | 0\.1608 (40.4%) | 0\.414 |
| **PCA (Residual)** | Empirical Cov($y_f - \mu$) | 0\.1315 (52.8%) | **0\.1150 (59.8%)** | 0\.2131 (18.3%) | 0\.753 |
| **DCT / Fourier** | 0 samples (Fixed) | 0\.2156 / 0.2184 (17%) | 0\.2156 / 0.2184 (17%) | 0\.2156 / 0.2184 (17%) | 0\.742 / 0.748 |

**Takeaway:** Data-dependent bases (both standard and residual PCA) degrade significantly under small calibration samples ($N=10$) or regime shifts (where residual PCA loses $75\%$ of the signal energy), whereas Mercer remains completely invariant.
\[TO ADD: Direct empirical comparison against an Autoencoder + Latent Flow Matching baseline at matched latent dimension $k$.\]

---

### 5, 7, 8 & 9. Probabilistic Baselines, Scaling, Main Table Numbers, and Multivariate Scope

- **Main Table Numbers:** We will replace the rank-only summary in Table 1 of the main text with the exact numerical CRPS and MAE values (currently in Appendix Tables 14–15).
- **Computational Scaling:** Truncating the horizon representation from $L_f=32$ to $k=6..10$ reduces the MLP's input/output projection parameters by $\sim 28\%$ (from 24.9k to 18.0k parameters in our synthetic suite) while eigendecomposition of $\boldsymbol{\Sigma} \in \mathbb{R}^{L_f \times L_f}$ is computed only **once** prior to training ($< 2$ ms).
  \[TO ADD: Scaling table across sequence lengths $L \in \{192, 336, 720\}$ and comparison with LS4 / Probabilistic Transformer.\]
  \[TO ADD: Multivariate correlated-channel experiment or explicit scope statement.\]
