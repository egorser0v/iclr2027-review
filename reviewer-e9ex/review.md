---
title: review
order: 1
---

**Critical Weaknesses:**

1. **The central mechanism rests on a mis-specified model of Adam.** The analysis treats Adam as Jacobi preconditioning with  and studies . Adam, however, divides gradients by the *square root* of the second moment (cf. Balles & Hennig \[2\]). Under the Kronecker model the paper adopts, this applies  once rather than twice, so the relevant matrix is . Under this model a diagonalized covariance is not optimally conditioned (for the Mercer basis the condition number is , not 1), so the claimed conditioning advantage does not follow. Prop. 2 also gives the same bound for Fourier and DCT, so it does not separate the bases it is meant to compare.
2. **Under the OU prior used in all main experiments, the Mercer basis is essentially the DCT basis, so the claimed advantage and adaptivity are weakly supported.** The paper itself notes that DCT is an asymptotic Karhunen–Loève basis for OU processes, so Mercer and DCT should behave almost identically in the main benchmark. The benefit of Mercer is shown only in a single periodic-kernel experiment on one dataset, and the claim of adapting to non-stationary priors is not tested.
3. **The gains over TSFlow cannot be attributed to the proposed basis.** In Table 1 even the plain Raw MLP clearly outranks TSFlow, and by Prop. 1 the choice of basis does not change the model class. The accuracy and efficiency gains therefore come from replacing S4 with an MLP rather than from the Mercer basis. Moreover, TSFlow \[1\] was originally evaluated on GluonTS datasets with short horizons, whereas here all models are run on different benchmarks with long horizons, so the comparison is not anchored to TSFlow's own setting. Apart from TSFlow, the baselines are deterministic forecasters, which cannot be judged on calibration; probabilistic baselines are missing.

**Questions For Authors:**

1. Could the conditioning analysis be redone with the Adam-consistent matrix ? Which conclusions survive?
2. Decorrelation is exact only for the centred joint prior. Since the conditional covariance  of the horizon does not depend on the context, why not use its eigenbasis, which decorrelates the source noise exactly for every window? (Relatedly, for an OU kernel the conditional mean is essentially determined by the last context point, so how much history does the prior actually carry?)
3. Can the method be compared with TSFlow in its original protocol, and with at least one additional probabilistic baseline?
4. For the periodic-prior study, can a Fourier baseline matched to the period be included, and can the benefit be shown on more than one dataset?

Little tip: you can position the work relative to Functional Flow Matching \[3\] and Blurring Diffusion Models \[4\].

**References**

- \[1\] Marcel Kollovieh, Marten Lienen, David Lüdke, Leo Schwinn, Stephan Günnemann. *Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting.* ICLR 2025. <https://arxiv.org/abs/2410.03024>
- \[2\] Lukas Balles, Philipp Hennig. *Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients.* ICML 2018. <https://proceedings.mlr.press/v80/balles18a.html>
- \[3\] Gavin Kerrigan, Giosue Migliorini, Padhraic Smyth. *Functional Flow Matching.* AISTATS 2024. <https://proceedings.mlr.press/v238/kerrigan24a.html>
- \[4\] Emiel Hoogeboom, Tim Salimans. *Blurring Diffusion Models.* ICLR 2023. <https://arxiv.org/abs/2209.05557>
