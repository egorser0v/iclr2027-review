---
title: General response
order: 0
---

We thank the reviewers for helping us distinguish the benefits of the Gaussian process prior, the lightweight correction network, and its representation. We agree that the original presentation placed too much emphasis on Mercer-specific gains and attributed backbone savings too broadly.

The expanded evaluation gives a clearer picture. Across 54 native-univariate GIFT-Eval tasks, our neural correction improves aggregate weighted quantile loss (wQL) by 13.90% over the fitted Gaussian forecast. Learning the correction in the original time coordinates achieves close full-rank accuracy, with a 0.72% difference from using the fitted covariance's eigenvectors. The main practical result is therefore that a small flow-matching network can materially improve a structured probabilistic forecast.

The synthetic experiments explain the complementary role of representation. Full-rank bases perform similarly, while low-rank bases determine which corrections the model can express. Nonstationary and smooth-kernel examples show when this choice improves accuracy or regularization. We also correct the Adam analysis and attribute the measured efficiency gains from replacing the S4 sequence model with a multilayer perceptron to the backbone change.

Together, these results sharpen the contribution: lightweight correction of a Gaussian forecast, with controlled evidence for when representation matters. The individual responses below present the relevant experiments and numerical comparisons.
