---
title: General response
order: 0
---

We thank the reviewers for their constructive feedback, which has helped us refine the positioning of our paper. TSFlow focuses on matching the prior to the temporal structure of the data. We ask whether the prior is a good probabilistic forecaster in its own right. When it is, how much can a very small neural network gain from the predictive information it already contains? Building on TSFlow, we learn a small correction to that forecast. The challenge is to build on that strong starting point as effectively as possible, with minimal training cost.

Our results support this approach. A small learned correction improves the statistical forecast and reaches accuracy competitive with larger task-trained forecasters, at substantially lower training cost in the profiled implementations. Expressing the correction in the prior's eigenbasis gives a smaller additional improvement. These results show the value of combining a strong statistical forecast with a small learned correction.

The reviews also prompted us to clarify what the choice of coordinates contributes. The original paper studied full-rank rotations and their effect on optimization and stability. In the rebuttal, we also test smaller corrections restricted to selected directions in synthetic experiments. The basis then determines which corrections the model can express. Nonstationary and smooth-kernel examples show when this choice improves accuracy or regularization. We have also corrected the Adam analysis. The measured efficiency gains from replacing the S4 sequence model with an MLP come from that architecture change.

The individual responses below develop these points and provide the supporting experiments and numerical comparisons.
