---
title: review
order: 1
---

**Critical Weaknesses:**

1 The method primarily combines a classical kernel eigendecomposition with standard linear flow matching. It changes the optimization coordinates rather than the underlying transport construction.

2 The main computational comparison replaces a 186k-parameter S4 with a 55k-parameter MLP. Since FM already supports lightweight networks, this does not establish how much efficiency the proposed basis itself contributes.

**Questions For Authors:**

1. How large are Mercer’s absolute accuracy gains over Raw, DCT, and Fourier under individually tuned optimization and matched computational budgets?
2. How reliably does prior-based Jacobi conditioning explain actual AdamW behavior on conditional interpolants?
3. Do the conclusions remain convincing when reporting effect sizes, confidence intervals, and rankings under a more conventional significance threshold?
