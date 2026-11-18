---
title: answer
order: 2
---

We appreciate your constructive feedback and theoretical insights, which directly motivated our revised horizon-only low-rank formulation and corrected Adam analysis.

### 1\. Adam Conditioning Analysis

Under the Kronecker model (Balles & Hennig, 2018), Adam scales gradients by the square root of the second moment, making the effective preconditioned matrix $M_{\text{Adam}} = D^{1/2} A D^{1/2}$ (with $D = \text{diag}(A)^{-1/2}$) rather than $D A D$.
In the revised Proposition 2:

- **Raw ($\Phi = I$):** Since $\text{diag}(A)$ is constant on a stationary grid, $D \propto I$, so $\kappa(M_{\text{Adam}}) = \kappa(A)$ and off-diagonal coupling remains dense.
- **Mercer ($\Phi^\top K \Phi = \Lambda$):** The covariance is strictly diagonalized, reducing the condition number to the square root $\kappa(M_{\text{Adam}}) = \sqrt{\kappa(A)}$ while eliminating off-diagonal gradient coupling entirely ($\text{OffDiag} = 0$). Furthermore, under low-rank truncation ($k \ll L_f$) or spectral whitening ($\Lambda_k^{-1/2}$), $\kappa(M_{\text{Adam}})$ is bounded by $\sqrt{\lambda_1 / \lambda_k} = \mathcal{O}(1)$ or $1$, respectively.

**Table 1: First-layer interpolant geometry ($c_t$) and prior decorrelation ($c_0$) on Horizon-Only Matérn-3/2 ($L_f=32$).**

| Basis | Prior OffDiag ($c_0$) $\downarrow$ | Interpolant OffDiag ($c_t$) $\downarrow$ | $\kappa(DAD)$ (Train / Test) $\downarrow$ | $\kappa(D^{1/2}AD^{1/2})$ (Train / Test) $\downarrow$ |
|:---|:---:|:---:|:---:|:---:|
| **Mercer (Ours)** | **0\.000** | **0\.087** | **3\.1 / 4.4** | **250 / 320** |
| **PCA** | 0\.181 | 0\.101 | 48\.0 / 49.0 | 330 / 450 |
| **DCT** | 0\.524 | 0\.538 | 640 / 870 | 900 / 1,100 |
| **Fourier** | 0\.545 | 0\.561 | 4,500 / 5,500 | 2,900 / 3,800 |
| **Raw** | 0\.966 | 0\.970 | 58,000 / 68,000 | 46,000 / 56,000 |

**Takeaway:** Even under the exact $D^{1/2} A D^{1/2}$ scaling, Mercer reduces the condition number by two orders of magnitude over Raw and eliminates off-diagonal coordinate coupling on the prior.

---

### 2 & 3. Horizon-Only $\Sigma$\-Basis & Low-Rank Truncation vs. DCT, Fourier, and PCA

Following your suggestion in Question 2, we replaced the joint-window projection with the **eigenbasis of the conditional horizon covariance** $\boldsymbol{\Sigma} = K_{ff} - K_{fp}K_{pp}^{-1}K_{pf} = \Phi \Lambda \Phi^\top \in \mathbb{R}^{L_f \times L_f}$. The GP prior sample $y_0 \sim \mathcal{N}(\mu, \Sigma)$ is kept at full rank, history $[y_p, \mu]$ enters the MLP via a separate conditioning branch, and the flow learns a **low-rank residual correction** restricted to the top-$k$ modes:

$$\hat{y}_f = y_0 + \Phi_k(\hat{c}_1 - c_0), \quad \Phi_k \in \mathbb{R}^{L_f \times k}.$$

While full-rank orthonormal bases ($k=L_f$) parameterize the same function class and perform similarly, low-rank truncation ($k \ll L_f$) separates the bases dramatically when the kernel departs from a stationary OU process.

**Table 2: Controlled truncation study on non-stationary modulated periodic data ($L_f=32, p=5.5$, GP Prior CRPS = 0.2565, Oracle CRPS = 0.0200).**

| Basis | $k=6$ CRPS $\downarrow$ (Gap $\uparrow$) | $k=10$ CRPS $\downarrow$ (Gap $\uparrow$) | $k=10$ Proj. Loss $\downarrow$ | $k=16$ CRPS $\downarrow$ (Gap $\uparrow$) | $k=32$ (Full) CRPS $\downarrow$ (Gap $\uparrow$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Mercer (Ours)** | **0\.1619** (**40\.0%**) | **0\.0768** (**76\.0%**) | **0\.075** | **0\.0597** (**83\.2%**) | 0\.0531 (86.0%) |
| **PCA** | 0\.1660 (38.3%) | 0\.0884 (71.1%) | 0\.103 | 0\.0613 (82.5%) | 0\.0547 (85.3%) |
| **DCT** | 0\.2315 (10.6%) | 0\.2140 (18.0%) | 0\.728 | 0\.1834 (30.9%) | **0\.0528** (**86\.1%**) |
| **Fourier** | 0\.2304 (11.0%) | 0\.2167 (16.8%) | 0\.732 | 0\.1839 (30.7%) | 0\.0543 (85.5%) |
| **Raw** | 0\.2331 (9.9%) | 0\.2180 (16.3%) | 0\.882 | 0\.1819 (31.6%) | 0\.0546 (85.4%) |

**Takeaway:** At full rank ($k=32$), all codecs tie, confirming our theoretical invariance analysis; under low-rank truncation ($k=10$), fixed low-frequency bases (DCT/Fourier) discard $>72\%$ of the signal energy, whereas Mercer captures the dominant kernel modes, closing $4\times$ more of the Oracle gap and statistically outperforming PCA (paired 95% CI $[-0.0161, -0.0071]$).

---

### 4\. Non-OU, Periodic, and Non-Stationary Priors

Table 2 above already uses a **non-stationary modulated periodic kernel** $K(s,t) = (0.5 + c \min(s,t)) K_{\text{Per}}(s,t)$ with a non-divisor period ($p=5.5$ on $L_f=32$), demonstrating how Mercer adapts to both non-stationarity and non-circulant periodicity.
\[TO ADD: Results on a second real-world periodic dataset and a period-matched Fourier baseline.\]

---

### 5 & 6. Separating Backbone Efficiency from Basis Choice, TSFlow Protocol, and Baselines

We agree that the $4.7\times$ memory and $4\times$ speed gains over TSFlow come from replacing the sequential S4 backbone with an MLP, whereas the specific benefit of the Mercer basis lies in enabling **low-rank flow corrections** ($k \ll L_f$, reducing MLP input/output parameters while preserving accuracy, as shown in Table 2).

\[TO ADD: Comparison on TSFlow's original short-horizon GluonTS benchmarks and additional probabilistic baselines (LS4 / Probabilistic Transformer).\]

---

### 7\. Role of History Length Under OU vs. Separate Context Conditioning

In our revised Horizon-Only architecture, the observed history $y_p$ is fed directly into the MLP alongside the GP mean $\mu$, allowing the network to capture historical dependencies even when the GP kernel has a short memory.

\[TO ADD: History-length ablation comparing full context vs. last-point conditioning.\]

---

### 8\. Positioning Relative to Functional Flow Matching & Blurring Diffusion

We will cite Kerrigan et al. (2024) and Hoogeboom & Salimans (2023) prominently. While Functional Flow Matching defines Gaussian measures in function spaces and Blurring Diffusion uses fixed DCT projections for stationary Gaussian blur, our work uses the **conditional posterior eigenbasis ($\boldsymbol{\Sigma}$)** of arbitrary time-series kernels to learn low-rank residual corrections on top of a full-rank GP prior.
