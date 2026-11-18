---
title: answer
order: 2
---

We appreciate your candid assessment. We agree that a full-rank orthogonal rotation does not alter the model class and yields modest differences when fully converged. In response, we formulated and tested a **low-rank horizon-only variant** where the choice of basis fundamentally determines the subspace of learnable corrections.

### 1. Methodological Contribution: Low-Rank Residual Flow on the Horizon Covariance $\boldsymbol{\Sigma}$
In our revised formulation, the analytical GP posterior sample $y_0 \sim \mathcal{N}(\mu, \Sigma)$ is preserved at full resolution, while the MLP velocity field operates strictly within the top-$k$ eigenvectors $\Phi_k \in \mathbb{R}^{L_f \times k}$ ($k \ll L_f$) of the conditional horizon covariance $\boldsymbol{\Sigma}$:
$$\hat{y}_f = y_0 + \Phi_k (\hat{c}_1 - c_0).$$
Unlike a full-rank rotation ($k=L_f$), truncating to $k \ll L_f$ restricts the expressive subspace of the flow correction and acts as a spectral regularizer.

**Table 1: Effect of truncation rank $k$ on non-stationary modulated periodic data ($L_f=32$, GP Prior CRPS = 0.2565, Oracle = 0.0200).**
| Basis | $k=4$ CRPS $\downarrow$ | $k=6$ CRPS $\downarrow$ | $k=10$ CRPS $\downarrow$ (Gap $\uparrow$) | $k=10$ Proj. Loss $\downarrow$ | $k=32$ (Full) CRPS $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mercer (Ours)** | **0.1926** | **0.1619** | **0.0768 (76.0%)** | **0.075** | 0.0531 |
| **PCA** | 0.1959 | 0.1660 | 0.0884 (71.1%) | 0.103 | 0.0547 |
| **DCT** | 0.2385 | 0.2315 | 0.2140 (18.0%) | 0.728 | **0.0528** |
| **Fourier** | 0.2421 | 0.2304 | 0.2167 (16.8%) | 0.732 | 0.0543 |
| **Raw** | 0.2411 | 0.2331 | 0.2180 (16.3%) | 0.882 | 0.0546 |

**Takeaway:** While all bases converge to the same CRPS at full rank ($k=32$), Mercer reduces CRPS by nearly $3\times$ relative to DCT, Fourier, and Raw at $k=10$ (`0.0768` vs. `0.2140–0.2180`) because it retains $92.5\%$ of the target correction energy compared to $<28\%$ for fixed frequency bases.

---

### 2. Advantage Over Empirical PCA Under Sample Scarcity, Regime Shift, and Smooth Kernels
We further isolated two regimes where data-free low-rank Mercer outperforms empirical PCA and full-rank models:
1. **Basis Scarcity & Regime Shift ($k=8$):** Holding the MLP training data fixed ($N=512$) and varying only the calibration data used to fit PCA, Mercer outperforms standard PCA when calibration data are scarce ($N_{\text{basis}}=10$) or subject to a historical regime shift.
2. **Small-Sample Regularization on Smooth Kernels ($N=64$):** On smooth `RBF` and `Matérn-5/2` priors (where 6 modes capture $>99.6\%$ of $\boldsymbol{\Sigma}$ variance), truncating Mercer to $k=6$ prevents the MLP from overfitting to the 26 near-zero tail modes, beating full-rank $k=32$.

**Table 2: Paired comparisons and 95% confidence intervals (5 seeds, 256 fixed test windows; negative $\Delta\text{CRPS}$ favors the first method).**
| Experiment / Regime | Comparison | $\text{CRPS}_1$ vs. $\text{CRPS}_2$ | Mean $\Delta\text{CRPS}$ | Paired 95% CI ($p < 0.05$) |
| :--- | :--- | :---: | :---: | :---: |
| **Modulated Periodic ($k=10$)** | Mercer ($k=10$) $-$ Fourier ($k=10$) | `0.0768` vs `0.2167` | `-0.1399` | **`[-0.1415, -0.1384]`** |
| **Modulated Periodic ($k=10$)** | Mercer ($k=10$) $-$ PCA ($k=10$) | `0.0768` vs `0.0884` | `-0.0116` | **`[-0.0161, -0.0071]`** |
| **Small Basis Sample ($N_{\text{basis}}=10, k=8$)** | Mercer ($k=8$) $-$ PCA ($k=8$) | `0.1250` vs `0.1387` | `-0.0137` | **`[-0.0191, -0.0083]`** |
| **Historical Regime Shift ($k=8$)** | Mercer ($k=8$) $-$ PCA ($k=8$) | `0.1250` vs `0.1608` | `-0.0359` | **`[-0.0409, -0.0308]`** |
| **Smooth RBF Small Sample ($N=64$)** | Mercer ($k=6$) $-$ Mercer ($k=32$ Full) | `0.3932` vs `0.4245` | `-0.0313` | **`[-0.0535, -0.0091]`** |

**Takeaway:** Across all tested low-rank, small-sample, and shifted regimes, Mercer achieves statistically significant improvements ($p < 0.05$, 95% CIs strictly excluding zero) over competing codecs and full-rank baselines.

---

### 3, 4, 5 & 6. Efficiency Attribution, Optimizer Diagnostics, and Real-Data Benchmarks
- **Efficiency Attribution:** We will revise the text to state explicitly that the $4.7\times$ memory and $4\times$ runtime improvements over TSFlow stem from the MLP backbone, whereas the Mercer basis provides parameter and dimensionality reduction via low-rank truncation ($k \ll L_f$).
- **Optimizer Diagnostics:** In our revised analysis, we measure both $\kappa(DAD)$ and the Adam-consistent $\kappa(D^{1/2}AD^{1/2})$ directly on training interpolants $c_t$ alongside actual Adam second-moment statistics, confirming that while Mercer improves first-layer conditioning by two orders of magnitude over Raw, low-rank subspace selection is the primary driver of final CRPS gains.
- **Statistical Thresholds:** We have replaced the $p < 0.2$ ranking threshold with exact numerical CRPS tables and standard 95% confidence intervals ($p < 0.05$, Table 2).
[TO ADD: Updated real-world benchmark table with low-rank Horizon-Only MercerFlow and $p < 0.05$ paired confidence intervals.]
