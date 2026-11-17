---
title: review
order: 1
---

**Critical Weaknesses:**

- Empirical evaluations are rather limited, focusing on the five standard time series datasets. Evaluations on spatio-temporal datasets such as videos and PDEs are missing. I believe that evaluations on PDE datasets (rather than only on real-wold time series datasets) whose temporal structure is well understood can provide further insights into the advantages of the proposed method.
- There are no empirical comparisons to other probabilistic forecasters such as LS4 \[1\] and Probabilistic Transformer \[2\].
- There are no comparison/ablation studies for other choices of kernel such as the Matern kernel, signature kernel \[5\], and other sequential kernels.
- It is not clear how the performance of the proposed method compares to latent flow matching, where an autoencoder is first pretrained and then flow matching is performed on the latent space (see, e.g., the approach in \[3\], \[4\]), to see how the use of Mercer basis compares to learned basis/latent representations.
- Although a brief discussion on computational efficiency is provided, it is not clear how the computational cost scales with dimension and sequence length.
- Table 1 is not informative enough as the main result table. Quantitative metrics such as CRPS would be better to clearly demonstrate the differences between the compared baselines.

---

\[1\] Zhou, Linqi, et al. "Deep latent state space models for time-series generation." International Conference on Machine Learning. PMLR, 2023.

\[2\] Tang, Binh, and David S. Matteson. "Probabilistic transformer for time series analysis." Advances in neural information processing systems 34 (2021): 23592-23608.

\[3\] Lim, S. H., Wang, Y., Yu, A., Hart, E., Mahoney, M. W., Li, X. S., & Erichson, N. B. (2024). Elucidating the design choice of probability paths in flow matching for forecasting. arXiv preprint arXiv:2410.03229.

\[4\] Carvajal Reyes, C., & Tobar, F. (2026). Time series generation with spectrally aligned latent flow matching. arXiv e-prints, arXiv-2609.

\[5\] Király, F. J., & Oberhauser, H. (2019). Kernels for sequentially ordered data. Journal of Machine Learning Research, 20(31), 1-45.

**Questions For Authors:**

- The paper focuses on the exponential (OU) kernel as the primary kernel used for implementation. How will this work for non-Gaussian datasets? Will other kernels such as the Matern kernel and signature kernel help improving the empirical results?
- The channels are treated independently in the method, which is fine for simplifying the method. How would the method fare in situations where we have multivariate time series data that are highly correlated (across the channels)?
- How would the performance of the method depends on the smoothness of the dataset?
- How would the computational cost scales with dimension and sequence length?
