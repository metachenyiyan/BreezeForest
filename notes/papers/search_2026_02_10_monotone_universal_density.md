# Paper Search: Monotone Neural Networks, Universal Density Estimation, and Normalizing Flows
**Date**: 2026-02-10

## Summary
Searched for papers at the intersection of monotone neural network transformations, universal density approximation, and normalizing flow architectures — the core techniques underlying BreezeForest. Found 12 relevant papers spanning foundational works (2018) through cutting-edge results (Jan 2026), with key themes including transformer-based autoregressive flows achieving state-of-the-art image generation, new analytic bijection families, and theoretical insights on Jacobian determinant optimality.

## Papers Found

### [Relevance: High] Transformer Neural Autoregressive Flows (T-NAFs)
- **Authors**: Massimiliano Patacchiola, Aliaksandra Shysheya, Katja Hofmann, Richard E. Turner
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2401.01855
- **Key contribution**: Treats each dimension as a separate token with attention masking for autoregressive constraint. Matches or outperforms NAF/BNAF with an order of magnitude fewer parameters, without composing multiple flows.
- **Relation to BreezeForest**: Directly comparable architecture. T-NAF replaces BreezeForest's tree+breeze connections with transformer attention. The attention-masked approach could inform BreezeForest's conditioning mechanism. Key question: does attention-based conditioning outperform BreezeForest's explicit breeze connections?
- **Techniques**: Autoregressive flow, transformer attention, monotone neural networks

### [Relevance: High] TarFlow: Normalizing Flows are Capable Generative Models
- **Authors**: Shuangfei Zhai, Ruixiang Zhang, Preetum Nakkiran, et al.
- **Year**: 2024 (ICML 2025)
- **arXiv**: https://arxiv.org/abs/2412.06329
- **Key contribution**: Transformer-based MAF variant (TarFlow) achieves state-of-the-art likelihood estimation and sample quality comparable to diffusion models on images. Uses Gaussian noise augmentation, post-training denoising, and guidance.
- **Relation to BreezeForest**: Demonstrates that autoregressive flows can scale to high-dimensional image data — validates the architectural family BreezeForest belongs to. The noise augmentation and denoising techniques could help BreezeForest with regularization beyond MultiBF.
- **Techniques**: Autoregressive flow, transformer, noise augmentation, guidance

### [Relevance: High] Neural Autoregressive Flows (NAF)
- **Authors**: Chin-Wei Huang, David Krueger, Alexandre Lacoste, Aaron Courville
- **Year**: 2018 (ICML 2018)
- **arXiv**: https://arxiv.org/abs/1804.00779
- **Key contribution**: Replaces affine transformations with monotone neural networks (Deep Sigmoidal Flows, Deep Dense Sigmoidal Flows). First proof that autoregressive flows with monotone neural network transformations are universal density approximators. Transformer fits target inverse conditional CDF.
- **Relation to BreezeForest**: Direct intellectual ancestor. BreezeForest's conditional CDF approach via sigmoid-activated trees is a variant of NAF's monotone transformation idea. BreezeForest's innovation is the tree structure that achieves O(N^2) instead of NAF's quadratic parameter growth.
- **Techniques**: Autoregressive flow, monotone neural network, conditional CDF

### [Relevance: High] Invertible Monotone Operators for Normalizing Flows
- **Authors**: Byeongkeun Ahn, Chiyoon Kim, Youngjoon Hong, Hyunwoo J. Kim
- **Year**: 2022 (NeurIPS 2022)
- **arXiv**: https://arxiv.org/abs/2210.08176
- **Key contribution**: Uses monotone operators to overcome Lipschitz constant limitations in ResNet-based flows. Introduces CPila activation function for better gradient flow. "Monotone Flows" achieve strong results on MNIST, CIFAR-10, ImageNet32/64.
- **Relation to BreezeForest**: Different notion of "monotone" — monotone operators (contractive maps) vs BreezeForest's monotone (increasing) transformations. However, the CPila activation and gradient flow analysis could inform BreezeForest's activation function choice (currently Sigmoid).
- **Techniques**: Residual flow, monotone operator, Lipschitz constraints

### [Relevance: High] Analytic Bijections for Smooth and Interpretable Normalizing Flows
- **Authors**: Mathis Gerdes, Miranda C. N. Cheng
- **Year**: 2026 (January 2026)
- **arXiv**: https://arxiv.org/abs/2601.10774
- **Key contribution**: Introduces cubic rational, sinh, and cubic polynomial bijections that are globally smooth and analytically invertible. Radial flows achieve comparable quality with 1000x fewer parameters on radially-structured targets.
- **Relation to BreezeForest**: Highly relevant — BreezeForest uses bisection for inversion because its sigmoid-based transformation lacks an analytic inverse. These analytic bijections could replace BreezeForest's sigmoid activation, eliminating the need for bisection entirely and enabling one-pass sampling.
- **Techniques**: Coupling flow, analytic inversion, scalar bijections

### [Relevance: High] Unconstrained Monotonic Neural Networks (UMNN)
- **Authors**: Antoine Wehenkel, Gilles Louppe
- **Year**: 2019 (NeurIPS 2019)
- **arXiv**: https://arxiv.org/abs/1908.05164
- **Key contribution**: Models monotonicity by describing a positive-output neural network and numerically integrating it (Clenshaw-Curtis quadrature). First monotonic architecture inverted for sample generation (UMNN-MAF).
- **Relation to BreezeForest**: Alternative to BreezeForest's constrained-weight approach. UMNN's "integrate positive derivative" is more flexible than BreezeForest's "squared weights for monotonicity." Could be combined with BreezeForest's tree structure.
- **Techniques**: Autoregressive flow, numerical integration, unconstrained monotonicity

### [Relevance: Medium] Jacobian Determinant of Normalizing Flows
- **Authors**: Huadong Liao, Jiawei He
- **Year**: 2021
- **arXiv**: https://arxiv.org/abs/2102.06539
- **Key contribution**: Proves the Jacobian determinant mapping is unique for given distributions, establishing a unique global optimum for likelihood. Shows likelihood relates to eigenvalues of auto-correlation matrices. Proposes design principles based on balancing expansion/contraction.
- **Relation to BreezeForest**: Theoretical foundation for understanding BreezeForest's numerical Jacobian approximation. The balance between expansion and contraction could inform BreezeForest's sap_w parameter tuning. Confirms that the Jacobian computation is the critical bottleneck to get right.
- **Techniques**: Theoretical analysis, Jacobian determinant, training stability

### [Relevance: Medium] Block Neural Autoregressive Flow (BNAF)
- **Authors**: Nicola De Cao, Ivan Titov, Wilker Aziz
- **Year**: 2019 (UAI 2019)
- **arXiv**: https://arxiv.org/abs/1904.04676
- **Key contribution**: Single feed-forward network with block-triangular masked weights. Universal approximator with orders of magnitude fewer parameters than NAF. Analytical Jacobian tracking through layers.
- **Relation to BreezeForest**: Most direct competitor. BreezeForest decomposes BNAF's monolithic structure into per-dimension trees, achieving O(N^2) vs O(N^3) Jacobian computation. See detailed comparison in `notes/comparisons/bf_vs_bnaf_2026_02_10.md`.
- **Techniques**: Autoregressive flow, block-triangular masks, weight normalization

### [Relevance: Medium] Mixture Modeling with Normalizing Flows for Spherical Density Estimation
- **Authors**: Tin Lok James Ng, Andrew Zammit-Mangion
- **Year**: 2023
- **arXiv**: https://arxiv.org/abs/2301.06404
- **Key contribution**: Mixture of normalizing flows on spheres using EM algorithm. Demonstrates mixture approach yields better density representation than single components.
- **Relation to BreezeForest**: Validates the MultiBF mixture approach. Different domain (spherical vs Euclidean) but same principle: mixture of flow components captures complex multi-modal densities. EM training could be an alternative to BreezeForest's joint gradient training.
- **Techniques**: Mixture model, normalizing flows, EM algorithm, spherical manifold

### [Relevance: Medium] Stable Training of Normalizing Flows for High-dimensional Variational Inference
- **Authors**: Daniel Andrade
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2402.16408
- **Key contribution**: Identifies high-variance stochastic gradients as cause of training instability in high-dimensional flows. Proposes soft-thresholding of scale parameter and bijective soft log transformation (LOFT) for stable training.
- **Relation to BreezeForest**: Directly relevant if BreezeForest scales to high dimensions. The soft-thresholding technique could complement BreezeForest's sap_w regularization. LOFT transformation could be applied to BreezeForest's outputs.
- **Techniques**: Real NVP, variance reduction, soft-thresholding, high-dimensional

### [Relevance: Medium] Sum-of-Squares Polynomial Flow
- **Authors**: Priyank Jaini, Kira A. Selby, Yaoliang Yu
- **Year**: 2019 (ICML 2019)
- **arXiv**: https://arxiv.org/abs/1905.02325
- **Key contribution**: Uses positive polynomials (sum-of-squares) as monotone transformations in triangular maps. Theoretically grounded in probability theory. Interpretable and parameter-efficient.
- **Relation to BreezeForest**: Alternative monotonicity approach. SOS polynomials are more structured than BreezeForest's neural network trees. Could potentially replace tree layers with polynomial transformations for better theoretical guarantees.
- **Techniques**: Autoregressive flow, polynomial transformation, triangular maps

### [Relevance: Low] Self Normalizing Flows
- **Authors**: T. Anderson Keller, Jorn W.T. Peters, Priyank Jaini, Emiel Hoogeboom, Patrick Forré, Max Welling
- **Year**: 2021 (ICML 2021)
- **arXiv**: https://proceedings.mlr.press/v139/keller21a.html
- **Key contribution**: Approximates log Jacobian determinant using the inverse of the Jacobian itself. Flow components learn to approximate their own inverse through self-supervised reconstruction loss.
- **Relation to BreezeForest**: Interesting alternative to BreezeForest's finite-difference Jacobian. Self-supervised inversion learning could replace bisection for sampling.
- **Techniques**: Self-supervised, Jacobian approximation, invertible networks

## Key Takeaways

1. **T-NAFs are BreezeForest's closest modern competitor**: Published Jan 2024, T-NAFs use transformers for the same problem BreezeForest solves with trees. Direct empirical comparison on benchmarks would be the highest-impact next step.

2. **TarFlow validates autoregressive flows at scale**: The Dec 2024 result showing autoregressive flows matching diffusion models on images demonstrates the architectural family is viable at scale, not just on toy problems.

3. **Analytic bijections could eliminate bisection**: The Jan 2026 paper on analytic bijections (cubic rational, sinh, cubic polynomial) offers drop-in replacements for BreezeForest's sigmoid that would enable analytic inversion, eliminating the bisection bottleneck entirely.

4. **Monotonicity enforcement is converging**: The field is moving toward more flexible monotonicity (UMNN's positive derivatives, SOS polynomials, analytic bijections) rather than constrained weights (BreezeForest, NAF, BNAF). BreezeForest could benefit from adopting these approaches.

5. **Mixture of flows is validated but underexplored**: MultiBF's approach is validated by the spherical mixture paper, but few works combine mixture models with universal autoregressive flows. This is a distinctive strength of BreezeForest.

6. **High-dimensional stability is a known challenge**: The stable training paper confirms that scaling flows to high dimensions requires specific regularization techniques (soft-thresholding, LOFT). BreezeForest will face these challenges when scaling beyond 2D.

## Suggested Follow-up
- **Read T-NAFs paper in full**: Most directly comparable to BreezeForest; understand their attention masking and benchmark results
- **Experiment with analytic bijections**: Replace sigmoid with cubic rational bijection to test if bisection can be eliminated
- **Implement UMNN-style monotonicity**: Test if unconstrained positive derivatives outperform squared weights in BreezeForest's tree layers
- **Run BreezeForest on standard benchmarks**: POWER, GAS, HEPMASS, MINIBOONE, BSDS300 for direct comparison with NAF/BNAF/T-NAF results
- **Search for "tree-structured neural networks" + "density estimation"**: No direct results found for tree-structured flow architectures — BreezeForest's tree structure may be genuinely novel
