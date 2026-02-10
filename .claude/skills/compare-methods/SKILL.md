---
description: Compare BreezeForest with other normalizing flow methods on multiple dimensions
allowed-tools: WebSearch, WebFetch, Read, Write, Glob, Grep, Bash
argument-hint: <method name or paper, e.g. "BNAF" or "Neural Spline Flows" or "RealNVP vs Glow">
---

# Compare Methods

Compare BreezeForest with other normalizing flow methods.

**Method(s) to compare**: $ARGUMENTS

## Task: Multi-Dimensional Method Comparison

### 1. Read BreezeForest Implementation

First, read the BreezeForest source code to extract current implementation details:

- `model/BreezeForest.py` — Main model: autoregressive flow, conditional CDF approach, Jacobian-free determinant via numerical differentiation (2 forward passes), batched bisection for inversion, `sap_w` regularization parameter
- `model/TreeLayer.py` — Tree architecture: each dimension has a "Tree", previous dimensions send "breeze" connections to later dimensions, strictly increasing output via softplus weights
- `model/MultiBF.py` — Mixture model: combines multiple BreezeForest components with learnable mixing weights for regularization against sample memorization
- `model/tools.py` — Utilities: bisection algorithm, Sigmoid/logit transforms, numerical tools

Key BreezeForest characteristics to extract:
- Architecture: autoregressive, tree-structured per dimension, breeze connections
- Jacobian: lower triangular (autoregressive), computed via numerical differentiation (delta=0.0001) or exact `torch.func.jacrev`
- Universality: single hidden layer is universal density estimator (like BNAF)
- Sampling: batched bisection algorithm to invert the flow
- Regularization: `sap_w` parameter + MultiBF mixture model to avoid sample memorization
- Complexity: O(N^2) vs BNAF's O(N^3) for Jacobian determinant

### 2. Research the Comparison Target

Use `WebSearch` to find information about "$ARGUMENTS":
- Original paper (arXiv)
- GitHub repository (if available)
- Key technical details from the paper abstract and related resources

If a GitHub repo is found, use `gh` CLI or `WebFetch` to understand the implementation.

### 3. Compare on These Dimensions

#### 3.1 Architecture
| Aspect | BreezeForest | [Target Method] |
|---|---|---|
| Flow type | Autoregressive | ? |
| Building block | Tree + breeze connections | ? |
| Coupling mechanism | Conditional CDF via breeze | ? |
| Depth requirement | Few layers (universal with 1 hidden layer) | ? |
| Parameterization | Softplus-constrained weights (monotonicity) | ? |

#### 3.2 Jacobian Determinant
| Aspect | BreezeForest | [Target Method] |
|---|---|---|
| Jacobian structure | Lower triangular | ? |
| Computation method | Numerical diff (2 forward passes) / exact jacrev | ? |
| Complexity | O(N^2) per sample | ? |
| Numerical stability | delta=0.0001 approximation | ? |

#### 3.3 Expressiveness
| Aspect | BreezeForest | [Target Method] |
|---|---|---|
| Universal approximation | Yes (1 hidden layer) | ? |
| Density modeling | Conditional CDF decomposition | ? |
| Multi-modal support | MultiBF mixture model | ? |

#### 3.4 Training
| Aspect | BreezeForest | [Target Method] |
|---|---|---|
| Loss function | Negative log-likelihood | ? |
| Regularization | sap_w + mixture model | ? |
| Anti-memorization | MultiBF Gaussian-like constraint | ? |
| Optimizer | Adam (typical) | ? |

#### 3.5 Sampling / Inversion
| Aspect | BreezeForest | [Target Method] |
|---|---|---|
| Inversion method | Batched bisection | ? |
| Sampling source | Uniform distribution | ? |
| Sampling speed | Iterative (bisection) | ? |

#### 3.6 Practical Considerations
| Aspect | BreezeForest | [Target Method] |
|---|---|---|
| Code availability | This repo | ? |
| Scalability (dimension) | Tested on 2D | ? |
| Maturity | Research prototype | ? |

### 4. Analysis and Insights

Provide:
- **Advantages of BreezeForest** over the target method
- **Advantages of the target method** over BreezeForest
- **Complementary ideas**: Techniques from the target that could enhance BreezeForest
- **Open questions**: Areas where comparison is unclear and needs empirical testing

### 5. Output Format

```markdown
# Method Comparison: BreezeForest vs [Target Method]
**Date**: YYYY-MM-DD

## Overview
[1 paragraph summary of both methods and key differences]

## Comparison Table
[Combined table with all dimensions]

## Detailed Analysis

### Architecture Differences
[Discussion]

### Computational Efficiency
[Discussion with complexity analysis]

### Expressiveness and Universality
[Discussion]

### Training and Regularization
[Discussion]

### Sampling and Generation
[Discussion]

## Key Insights
1. [Insight about relative strengths]
2. [Insight about complementary techniques]
3. ...

## Potential Improvements for BreezeForest
- [Concrete suggestion inspired by the comparison]
- ...

## References
- [Paper 1]
- [Paper 2]
```

### 6. Save Results

Save to `notes/comparisons/bf_vs_[method_slug]_YYYY_MM_DD.md`.
Create the directory if it doesn't exist.
