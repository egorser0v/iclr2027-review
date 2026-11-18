---
title: General response
order: 0
---

We thank the reviewers for the opportunity to clarify the question motivating our work: how much neural computation does probabilistic forecasting require when a statistical model already provides a strong forecast? This raises a design question: how should that forecast be integrated into a learned model so that its quality translates into lower training cost? Building on flow matching with Gaussian process priors, as in TSFlow, we investigate how far a small MLP correction can go, and how aligning its coordinates with the prior affects optimization and stability.

Across 54 native-univariate GIFT-Eval tasks, our neural correction improves aggregate weighted quantile loss (wQL) by 13.90% over the fitted Gaussian forecast and achieves accuracy competitive with larger task-trained forecasters, using 4.2× fewer training operations than PatchTST and 25.7× fewer than TFT in the profiled implementations. These results show how a strong statistical forecast can support accurate probabilistic forecasting with a small supervised network. The full-rank eigenbasis improves aggregate wQL by 0.72% over the same correction in time coordinates.

The original paper studied full-rank rotations, focusing on optimization and stability. For this rebuttal, we additionally test truncation in synthetic experiments: the basis then determines which corrections the model can express, allowing us to examine whether the correction can be made smaller. Nonstationary and smooth-kernel examples show when this choice improves accuracy or regularization. We also correct the Adam analysis and attribute the measured efficiency gains from replacing the S4 sequence model with a multilayer perceptron to the backbone change.

The individual responses below clarify this contribution and present the relevant experiments and numerical comparisons.
