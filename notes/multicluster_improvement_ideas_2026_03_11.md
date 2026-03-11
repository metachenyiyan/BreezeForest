# BreezeForest Multi-Cluster Improvement Ideas
**Date**: 2026-03-11  
**Problem**: MultiBF generates points **between clusters** (inter-cluster artifacts) when trained on multi-modal datasets (e.g., 8 Gaussians). Longer training and LR tuning do not resolve this.

---

## Root Cause Analysis

### 1. Topological Mismatch (Fundamental)
Each `BreezeForest` component is a bijective map from a **connected** base space `Uniform(0.01, 0.99)^d` to the real space. Since the uniform base is topologically connected (a cube), and a bijection preserves topology, the image of each component must also be topologically connected — it **cannot** represent a disconnected support.

For a multi-cluster target (e.g., 8 well-separated Gaussians), each BreezeForest component **must** assign non-zero density to the "corridor" between clusters to maintain a valid bijection. This is a mathematical certainty, not a training failure.

**Evidence in code**:
- `MultiBF.inverse_map`: samples `z ~ Uniform(0.01, 0.99)^d` uniformly, then inverts. Every z value must map to some x — including z values that map to inter-cluster regions.
- `bisection` in `tools.py`: Stage 1 uses `Normal(0,1)` as distribution space, Stage 2 refines in real space. Neither is component-specific.

### 2. Component Non-Specialization (Training)
`MultiBF.train_forward` computes:
```
log p(x) = logsumexp_k( log π_k + log|det J_k(x)| )
```
The gradient of this loss flows to **all K components** for every training sample, regardless of which component should "own" that sample. This prevents components from specializing to individual clusters — instead, each component is nudged to explain ALL clusters, so inter-cluster density is not punished.

**Evidence in code** (`MultiBF.py`, lines 115–138):
- `stacked = torch.stack(component_log_probs, dim=0)` — all components contribute
- `log_prob = torch.logsumexp(stacked, dim=0)` — gradient distributes to all via softmax-like weighting

### 3. Sigmoid Smoothness (Expressiveness)
The `Sigmoid` activation in `TreeLayer` creates smooth, bounded outputs in `(0, 1)`. Because sigmoid saturates gradually, each component assigns non-trivially small but non-zero density everywhere — including inter-cluster regions. The smooth sigmoid cannot create sharp "walls" of zero density at cluster boundaries.

**Evidence in code** (`TreeLayer.py`, line 69):
- `self.acti_func = Sigmoid()` — bounded smooth activation throughout all tree layers

---

## Top 3 Improvement Ideas

---

### Idea 1 (BEST — Topological Fix): Per-Component Learnable Gaussian Base Distribution

**Core idea**: Replace each component's implicit `Uniform(0.01, 0.99)^d` base with a **learnable Gaussian** `N(μ_k, diag(σ_k²))`. Component k learns `μ_k` and `σ_k` that center its sampling region on its target cluster.

**Why this fixes the problem**:
- With Gaussian base `N(μ_k, σ_k)`, sampling `z ~ N(μ_k, σ_k)` draws almost all z values near the cluster center → the inverse map `f_k^{-1}(z)` maps these concentrated z values into the cluster region
- The bijection still exists, but the probability mass of the base is now concentrated where we want it
- Components learn "where to look" — component k's μ_k drifts toward cluster k's centroid during training

**Implementation sketch** (without changing project code):
```python
# In MultiBF, add per-component learnable base params:
self.base_means = nn.Parameter(torch.zeros(n_components, dim))  # μ_k
self.base_log_stds = nn.Parameter(torch.zeros(n_components, dim))  # log σ_k

# In inverse_map, for component k:
mu = self.base_means[k]
std = torch.exp(self.base_log_stds[k])
z = torch.randn(n_k, self.dim) * std + mu  # N(μ_k, σ_k²)
# Then pass to bisection with component-specific distribution:
dis = Normal(mu[dim_i], std[dim_i])  # used in stage-1 bisection
```

**Supporting literature**:
- **StiCTAF** (arXiv:2510.07965, ICLR 2025): Stick-Breaking Mixture Normalizing Flows with component-wise base distributions explicitly designed to address multimodality. Directly proves per-component bases enable multi-modal coverage without mode-seeking bias.
- **Topological universality results** (arXiv:2402.06578): Formalizes that flow bijectivity + base topology constrains the target distribution's topology — disconnected support requires disconnected base or mixture.
- The existing `bisection` function already accepts a `distribution` parameter — it just needs to be made component-specific and learnable.

**Difficulty**: Medium. Requires adding 2×K×dim parameters and changing `inverse_map` sampling + bisection call sites.

**Expected impact**: ★★★★★ — Directly addresses the topological root cause. Even with a fixed (non-learnable) Gaussian base initialized near cluster centroids, this would virtually eliminate inter-cluster generation.

---

### Idea 2 (STRONG — Training Fix): EM-Style Hard/Soft Assignment Training for Component Specialization

**Core idea**: Replace the `logsumexp` training objective with **responsibility-weighted EM training**. In the E-step, compute per-sample component responsibilities `r_k(x_i) = P(k | x_i)`. In the M-step, scale each component's gradient by its responsibility.

**Why this fixes the problem**:
- Current log-sum-exp: gradient for component k on sample x_i is proportional to `exp(log π_k + log|J_k(x_i)|) / sum_j(...)` — still non-zero for all k even if component k fits x_i poorly
- EM training: component k only receives strong gradient from samples in cluster k → components specialize → each component has zero responsibility for off-cluster samples → no incentive to model inter-cluster regions

**Two variants**:

**Soft EM** (recommended): Weight component k's loss by `r_k(x_i)`:
```python
# E-step
with torch.no_grad():
    log_r = stacked - log_prob.unsqueeze(0)  # (K, batch)  log responsibilities
    r = torch.exp(log_r)  # (K, batch)  soft assignments

# M-step: weighted loss per component
loss = -torch.sum(r * stacked) / batch_size
```

**Hard EM** (simpler): Assign each sample to its argmax component:
```python
assignments = stacked.argmax(dim=0)  # (batch,) — hardest component per sample
for k in range(K):
    mask = (assignments == k)
    if mask.sum() > 0:
        loss_k = -stacked[k, mask].mean()
        loss_k.backward()
```

**Supporting literature**:
- **AMF-VI** (arXiv:2510.02056, 2024): Adaptive Mixture Flow Variational Inference — sequential expert training followed by likelihood-driven weight updates. Demonstrates that independent expert training (as in hard EM) prevents cross-contamination between components.
- **End-to-End Gaussian Mixture Priors for Diffusion** (arXiv:2503.00524, 2025): Uses EM to learn per-component base distributions that avoid mode collapse — iteratively adds components and trains via M-step coordinate ascent.
- **EM for Mixture of Experts** (OpenReview 2025): Scalable EM scheme where M-step enables parallel expert specialization. Convergence is guaranteed by EM's monotone improvement property.

**Difficulty**: Low-to-Medium. The main change is in `MultiBF.train_forward` — no architectural changes needed. Soft EM can be done with 2 extra lines after the logsumexp.

**Expected impact**: ★★★★☆ — Strong improvement in component specialization. Works synergistically with Idea 1. Convergence is slower than gradient-based training but more principled. Hard EM may cause "dead" components if poorly initialized; soft EM is safer.

---

### Idea 3 (STRONG — Expressiveness Fix): Replace Sigmoid with Rational-Quadratic Spline (Neural Spline Flow) Activation

**Core idea**: Replace the `Sigmoid()` activation in `TreeLayer` with a **monotone rational-quadratic spline** transform. The spline is parameterized by K knot positions, heights, and derivatives, allowing exact analytical inversion and much sharper density profiles.

**Why this fixes the problem**:
- Sigmoid assigns non-zero density everywhere due to its smooth exponential tails → even a perfectly specialized component cannot create true zero-density corridors between clusters
- Rational-quadratic splines have **compact support** within the spline domain. Outside the knot range, density is exactly zero (or near-zero with linear tails). This allows a component to be highly concentrated in a small region.
- Splines also allow asymmetric, multi-humped transformations within a single activation — more expressive than sigmoid at the same width
- **Bonus**: The spline has analytical inverse (no bisection needed) → sampling becomes O(1) per dimension per layer instead of O(log(1/ε)) bisection iterations

**Rational-quadratic spline** (as in arXiv:1906.04032):
```
f(x; {x_k, y_k, d_k}) = rational quadratic interpolant between knots
```
where `{x_k}` are input knot positions, `{y_k}` are output values, `{d_k}` are derivatives at knots. This satisfies: strictly monotone, analytically invertible, smooth (C¹).

**Implementation sketch** for `TreeLayer`:
```python
# Replace Sigmoid with learnable rational-quadratic spline
class RQSplineActivation(nn.Module):
    def __init__(self, n_bins=8, range_=(-3, 3)):
        # Learnable: widths (W), heights (H), derivatives (D) per bin
        self.widths = nn.Parameter(torch.zeros(n_bins))   # -> softmax -> bin widths
        self.heights = nn.Parameter(torch.zeros(n_bins))  # -> softmax -> bin heights
        self.derivatives = nn.Parameter(torch.zeros(n_bins + 1))  # -> softplus
    def forward(self, x):  # rational-quadratic interpolation, exact
        ...
    def inverse(self, y):  # analytical inverse of RQ spline
        ...
```

**Supporting literature**:
- **Neural Spline Flows** (arXiv:1906.04032, NeurIPS 2019): Original paper introducing rational-quadratic splines for normalizing flows. Shows dramatic improvement over affine/sigmoid coupling on multi-modal benchmarks.
- **A-RQS vs. sigmoid autoregressive flows** (arXiv:2302.12024, 2023 comparative study): Autoregressive rational-quadratic spline (A-RQS) outperforms sigmoid MAF/RealNVP on ALL tested multi-modal distributions (4–400 dimensions), with superior accuracy AND training speed.
- **Flexible Tails for Normalizing Flows** (ICML 2025): Extends spline flows with tail adaptation, directly relevant to BreezeForest's bounded sigmoid output issue.

**Difficulty**: High. Requires re-implementing `TreeLayer.forward_helper` and `BreezeForest.inverse_map` with spline arithmetic. The reward is potentially eliminating bisection entirely (exact analytical inversion).

**Expected impact**: ★★★★☆ — Strong expressiveness improvement. Does not fully solve the topological issue (Idea 1) but dramatically reduces the "softness" of inter-cluster density. Best combined with Idea 1 or 2.

---

## Summary Table

| | Idea 1: Per-Component Gaussian Base | Idea 2: EM Assignment Training | Idea 3: RQ Spline Activation |
|---|---|---|---|
| Root cause addressed | Topological (fundamental) | Component non-specialization | Activation smoothness |
| Implementation difficulty | Medium | Low-Medium | High |
| Expected impact | ★★★★★ | ★★★★☆ | ★★★★☆ |
| Requires arch change | Minor (2K×d params) | No (training loop only) | Yes (new layer type) |
| Key paper | StiCTAF (arXiv:2510.07965) | AMF-VI (arXiv:2510.02056) | NSF (arXiv:1906.04032) |
| Synergistic with | Idea 2 | Idea 1 | Ideas 1+2 |

## Recommended Implementation Order
1. **Start with Idea 2** (EM training) — zero architecture change, immediate improvement
2. **Then add Idea 1** (learnable Gaussian base) — works with existing `bisection` API, medium effort, biggest impact
3. **Add Idea 3** (RQ splines) as a longer-term enhancement to fully eliminate inter-cluster density

---

## References

1. **StiCTAF** — Stick-Breaking Mixture Normalizing Flows with Component-Wise Tail Adaptation (2024): https://arxiv.org/abs/2510.07965
2. **Neural Spline Flows** — Durkan et al., NeurIPS 2019: https://arxiv.org/abs/1906.04032
3. **AMF-VI** — Adaptive Mixture Flow Variational Inference (2024): https://arxiv.org/pdf/2510.02056
4. **End-to-End GMM Priors for Diffusion** (2025): https://arxiv.org/abs/2503.00524
5. **EM for Mixture of Experts** (2025): openreview.net/pdf/e1468f6e6b92d46c4eb13c3dba142fb92f8447a0
6. **Annealing Flow** — Sampling Multi-Modal High-Dimensional Distributions (ICML 2025): https://arxiv.org/abs/2409.20547
7. **Flow Universality / Topological Constraints** (2024): https://arxiv.org/abs/2402.06578
8. **A-RQS vs Affine/Sigmoid Comparison** (2023): https://arxiv.org/abs/2302.12024
9. **TarFlow** — Normalizing Flows are Capable Generative Models (ICML 2025): https://arxiv.org/abs/2412.06329
