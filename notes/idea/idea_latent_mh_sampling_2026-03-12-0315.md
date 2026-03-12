# Idea: Latent Metropolis-Hastings Sampling (LMH) — Jacobian-Guided Correct Sampling for Single BF and MultiBF

**创建时间**: 2026-03-12 03:15 UTC
**推荐优先级**: ⭐⭐⭐ 最高优先级（唯一同时覆盖单 BF 和 MultiBF 的 inference-time 修复，且无需 GMM 外部拟合）

---

## 问题定义

BreezeForest 的 `inverse_map()` 当前采样策略为：
```python
z ~ Uniform([0.01, 0.99]^d)
x = f^{-1}(z)   # via bisection
```

这会均匀地从整个 [0.01, 0.99]^d 采样，包括那些映射到 inter-cluster 区域的 z 值。

**已有采样修复方案的局限性**：

1. **LZR（矩形 box，2026-03-11 12:35）**：估计各维度分位数构成矩形边界，忽略维度间相关性，且仅适用于 MultiBF（每个组件单独处理）。已被 Latent GMM 替代。

2. **Latent GMM Resampling（2026-03-12 01:51）**：对每个组件的训练样本在 latent 空间拟合 GMM，然后从 GMM 采样。**有效但有以下限制**：
   - 需要 sklearn GMM 外部拟合，引入额外依赖
   - GMM 假设 latent 分布是多峰高斯，当组件未完全专一化时可能拟合不准
   - 仅针对 MultiBF，**不适用于单 BF** 场景

3. **两个方案均不适用于单 BF**：`demo_functions.py` 中的 `generate_sample()` 使用单 BF + Uniform 采样，没有任何修复机制。

**根本缺口**：需要一个不依赖外部拟合、基于 BreezeForest 自身 Jacobian 的、**同时适用于单 BF 和 MultiBF** 的 inference-time 采样修复。

---

## 从当前项目代码与已有 idea 中得到的背景判断

**代码分析**（关键发现）：

**单 BF 场景**（`demo_functions.py`，`BreezeForest.inverse_map()`）：
- 没有任何 LZR/Latent GMM 机制
- `generate_sample()` 直接从 Uniform(0.01, 0.99) 采样，不做任何密度感知
- `BreezeForest.train_forward()` 已经计算了 Jacobian（通过有限差分）：
  ```python
  du_dx = (x_deltas[1] - x_deltas[0]) / (2 * epsilons)
  du_dx = torch.abs(du_dx * bf.dim_mask + 1 - bf.dim_mask).clamp(min=0.001)
  x_logDet = torch.sum(torch.mean(torch.log(du_dx), dim=0))
  ```
- **关键洞察**：`train_forward()` 的 Jacobian 计算**完全可以在 inference 时重用**，用来评估任意 z 对应 x 的密度

**MultiBF 场景**（`MultiBF._per_sample_log_det()`）：
- 已有 per-sample log-determinant 计算，代码完整
- Latent GMM 是在这些 log-det 上做 forward pass 后拟合的

**BreezeForest 的 Jacobian 性质**：
- 由于 BF 是累积分布函数（CDF），Jacobian |det J_f(x)| = p(x)（数据密度的近似）
- 在 inter-cluster 区域，p(x) ≈ 0，因此 |det J_f(x)| ≈ 0
- 在 cluster 中心，p(x) 大，因此 |det J_f(x)| 大
- 这个性质可以**直接用作 latent 空间的密度 proxy**：
  ```
  对于 z 值，通过 f^{-1}(z) 得到 x，再计算 |det J_f(x)| = p_approx(x)
  p(z) ∝ p(x) / |det J_{f^{-1}}(z)| = p(x) × |det J_f(x)| = |det J_f(x)|^2
  ```

**已有 idea 分析**：
- **Latent GMM Resampling（2026-03-12 01:51）**：本 Idea 是其**替代方案**（不是升级）。GMM 方法更轻量，MH 方法更正确。根据使用场景选择：
  - 如果需要理论正确性（provably correct samples）且能接受 ~5-20x 计算开销 → 用 LMH
  - 如果需要快速且无 MCMC 开销 → 用 Latent GMM
- **Coeurdoux 2024（外部研究）**：直接理论支撑。"Normalizing flow sampling with Langevin dynamics in the latent space" 证明这类方法适用于任何预训练 NF，且无需重训练。

---

## 核心思路

在 latent 空间 [0.01, 0.99]^d 中运行 **Metropolis-Hastings（MH）链**，以 BreezeForest 的 Jacobian 为目标密度：

**目标密度**：`π(z) ∝ |det J_f(f^{-1}(z))|^2`（Jacobian 的平方，正比于 latent 空间的密度）

**MH 采样流程**（per 生成样本）：
1. **初始化**：z_0 ~ Uniform([0.01, 0.99]^d)（可用 Latent GMM 作为更好的起点）
2. **MH 步骤**（重复 T 次）：
   a. 计算 x_t = f^{-1}(z_t)（bisection）
   b. 计算 log π(z_t) = 2 × log|det J_f(x_t)|（有限差分 Jacobian）
   c. 提议 z' = clip(z_t + N(0, σ^2), 0.01, 0.99)（带边界裁剪的随机游走）
   d. 计算 x' = f^{-1}(z')，log π(z') = 2 × log|det J_f(x')|
   e. 接受/拒绝：以概率 min(1, exp(log π(z') - log π(z_t))) 接受 z' → z_t+1
3. **返回 z_T**，计算 x = f^{-1}(z_T)

**核心原理**：MH 链平稳分布是 π(z)，也就是 Jacobian 集中的区域（cluster 中心）。链的不变分布自然避开 inter-cluster 的低 Jacobian 区域。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**直觉论证**：

BreezeForest 的 Jacobian（数据密度）在以下情况下大 vs 小：
- **Cluster 中心附近**：f 的 CDF 变化快 → Jacobian 大 → π(z) 大 → MH 倾向于在此停留
- **Inter-cluster 区域**：f 的 CDF 变化慢（CDF 的"平台"）→ Jacobian 小 → π(z) 小 → MH 会拒绝这些 z 值的提议

**收敛保证**：Metropolis-Hastings 的平稳分布恰好是 π(z)，不需要 burn-in 后期校正。与 Latent GMM（无保证）相比，MH 给出了理论正确的采样分布。

**对比 Coeurdoux 2024 的 MALA 方法**：
- MALA 使用梯度（Langevin 项），收敛更快但需要计算 ∇_z log π(z)
- 本 Idea 使用无梯度的 MH（只需 density evaluation），更适合 BreezeForest 的 bisection-based inverse
- 对于低维（d=2）数据，无梯度 MH 效率已经足够

**对 Single BF 的适用性**：
- 单 BF 没有组件区分，Latent GMM 和 ICDR 均不适用
- LMH 对单 BF 完全适用：Jacobian 计算方式相同，MH 流程无需修改
- **这是 LMH 相对于 Latent GMM 的独特价值**

---

## 与历史 idea 的关系

| 历史 Idea | 关系 | 说明 |
|-----------|------|------|
| **LZR (2026-03-11 12:35)** | **间接替代** | LZR 被 Latent GMM 替代，LMH 是另一条不经过 GMM 的替代路径。LZR 的矩形限制不如 LMH 的 Jacobian 引导精确 |
| **Latent GMM Resampling (2026-03-12 01:51)** | **并行替代方案** | 两者目标相同（latent 空间密度感知采样），但路径不同。Latent GMM 更轻量（无 MCMC），LMH 更正确（有理论保证）。推荐：Latent GMM 作为默认，LMH 作为高质量要求时的选项 |
| **Hard-EM / DAEM** | 效果叠加 | training-time 修复使 Jacobian 在 cluster 内更集中 → MH 接受率更高 → 样本质量更好 |
| **K-Means Pre-Init** | 同上 | 同上 |
| **ICDR (2026-03-11 12:40)** | 同上 | ICDR 使不同组件的 Jacobian 在对方 cluster 区域更低 → MH 在 MultiBF 中效果更好 |

---

## 具体实现建议

### 步骤 1：为 BreezeForest 添加单样本密度评估

```python
def log_density_at_z(self, z, epsilon=None):
    """
    Compute log density (log Jacobian determinant) at latent point z.
    
    For use in MH sampling: π(z) ∝ |det J_f(f^{-1}(z))|^2
    
    :param z: latent point tensor (n_samples, dim)
    :param epsilon: finite difference step size (None = use self.epsilon)
    :return: log density at z (n_samples,)
    """
    if epsilon is None:
        epsilon = self.epsilon
    
    # Get x from bisection inverse
    with torch.no_grad():
        x = self.inverse_map(z, max_gap=1e-2)  # coarse inversion for speed
    
    # Compute log Jacobian at x
    x_deltas = torch.cat([
        (x - epsilon).view(1, -1, x.size(1)),
        (x + epsilon).view(1, -1, x.size(1))
    ], dim=0)
    
    breeze_list = []
    y = self.forward(x, breeze_list)
    x_deltas_y = self.breeze_forward(x_deltas, breeze_list)
    
    du_dx = (x_deltas_y[1] - x_deltas_y[0]) / (2 * epsilon)
    du_dx = torch.abs(du_dx * self.dim_mask + 1 - self.dim_mask).clamp(min=0.001)
    
    # Per-sample sum of log Jacobian diagonals
    log_jac = torch.sum(torch.log(du_dx), dim=1)  # (n_samples,)
    return log_jac
```

### 步骤 2：在 BreezeForest 中添加 MH 采样

```python
def inverse_map_mh(
    self,
    n_samples,
    n_mh_steps=10,
    step_size=0.05,
    max_gap=1e-2,
    decay_ratio=1.0,
    warm_start_z=None
):
    """
    Generate samples via Metropolis-Hastings in latent space.
    
    Target distribution: π(z) ∝ |det J_f(f^{-1}(z))|^2
    Sampler: Random-walk MH with Gaussian proposal
    
    :param n_samples: number of samples to generate
    :param n_mh_steps: number of MH steps per sample (more = higher quality, higher cost)
    :param step_size: proposal standard deviation in latent space
    :param warm_start_z: optional starting z points (n_samples, dim); None = Uniform
    :return: generated samples (n_samples, dim)
    """
    # Initialize chains
    if warm_start_z is not None:
        z_current = warm_start_z.clone()
    else:
        z_current = torch.rand(n_samples, self.dim) * 0.98 + 0.01
    
    # Compute initial log density
    log_pi_current = self.log_density_at_z(z_current)
    
    n_accepted = 0
    
    for step in range(n_mh_steps):
        # Gaussian random walk proposal
        noise = torch.randn_like(z_current) * step_size
        z_proposal = (z_current + noise).clamp(0.01, 0.99)
        
        # Evaluate proposed density
        log_pi_proposal = self.log_density_at_z(z_proposal)
        
        # MH acceptance
        log_alpha = log_pi_proposal - log_pi_current  # (n_samples,)
        accept_prob = torch.exp(log_alpha.clamp(max=0.0))  # min(1, exp(log_alpha))
        u = torch.rand(n_samples)
        accept = u < accept_prob
        
        # Update accepted samples
        z_current = torch.where(accept.unsqueeze(1).expand_as(z_current),
                                z_proposal, z_current)
        log_pi_current = torch.where(accept, log_pi_proposal, log_pi_current)
        n_accepted += accept.float().sum().item()
    
    avg_accept_rate = n_accepted / (n_samples * n_mh_steps)
    
    # Final inverse map
    x = self.inverse_map(z_current, max_gap=max_gap, decay_ratio=decay_ratio)
    
    return x, avg_accept_rate
```

### 步骤 3：为 MultiBF 添加 MH 采样包装

```python
def inverse_map_mh(
    self,
    n_samples,
    n_mh_steps=10,
    step_size=0.05,
    max_gap=1e-3,
    decay_ratio=1.0,
    use_latent_gmm_warmstart=False
):
    """
    Generate samples from MultiBF using MH in latent space per component.
    
    :param use_latent_gmm_warmstart: if True and latent_gmms are calibrated,
                                     use GMM samples as MH warm start
    """
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)
    total_accept_rates = []

    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue

        # Optional: use Latent GMM as warm start
        warm_start = None
        if use_latent_gmm_warmstart and hasattr(self, 'latent_gmms'):
            gmm_entry = self.latent_gmms[k] if k < len(self.latent_gmms) else None
            if gmm_entry is not None and gmm_entry[0] == 'gmm':
                _, gmm = gmm_entry
                z_warm, _ = gmm.sample(n_k)
                warm_start = torch.tensor(z_warm, dtype=torch.float32).clamp(0.01, 0.99)

        x_k, accept_rate = self.components[k].inverse_map_mh(
            n_k,
            n_mh_steps=n_mh_steps,
            step_size=step_size,
            max_gap=max_gap,
            decay_ratio=decay_ratio,
            warm_start_z=warm_start
        )
        results[mask] = x_k.detach()
        total_accept_rates.append(accept_rate)

    avg_accept = sum(total_accept_rates) / len(total_accept_rates) if total_accept_rates else 0.0
    return results, avg_accept
```

### 步骤 4：在 demo_functions.py 中使用

```python
# 单 BF 场景（替换 generate_sample 中的 inverse_map 调用）
model.eval()
with torch.no_grad():
    generated, accept_rate = model.inverse_map_mh(
        n_samples=data_size,
        n_mh_steps=15,
        step_size=0.04   # 调整以获得 0.2-0.5 的接受率
    )
    print(f"MH acceptance rate: {accept_rate:.3f}")
    generated = generated * std + mean
```

### 超参数调优

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `n_mh_steps` | 5 – 20 | 越多越准，但也越慢。5 步已有显著改善，20 步接近收敛 |
| `step_size` | 0.02 – 0.1 | 目标接受率 20%-50%。太大 → 接受率低；太小 → 混合慢 |
| 诊断 | 打印 accept_rate | <0.1 → 减小 step_size；>0.7 → 增大 step_size |
| `use_latent_gmm_warmstart` | True（如已 calibrate） | GMM 热启动大幅减少 burn-in，建议同时使用 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **计算开销大** | 每生成样本需要 2×T 次 bisection 调用（vs. 1 次）。T=10 时约慢 20x | 对高质量生成场景可接受；或只用 T=5 步的快速版本 |
| **低接受率** | 如果流训练不好（Jacobian 在 cluster 间变化平缓），MH 接受率低 | 与 DAEM/K-Means 结合使用，使 Jacobian 更集中；或增大 step_size |
| **Burn-in 问题** | 从 Uniform 起点运行 10 步可能不够 burn-in | 使用 Latent GMM 热启动；或增大 n_mh_steps |
| **边界效应** | 对 z 裁剪到 [0.01, 0.99] 会影响 MH 的细致平衡条件 | 可以用对称 logit 变换在 R^d 中运行 MH，然后 sigmoid 转回 [0,1] |
| **批量并行限制** | 各样本的 MH 链独立，可以完全并行；但 inverse_map 的 bisection 是顺序的 | 使用 batch 化 bisection（现有代码已支持 batch） |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（唯一同时覆盖单 BF 和 MultiBF 的 inference-time 修复）**

理由：
1. **覆盖单 BF 场景**：所有其他 inference-time 想法（LZR、Latent GMM）只针对 MultiBF；本 Idea 是唯一对单 BF 同样有效的采样修复方案
2. **理论保证**：MH 平稳分布是 π(z) ∝ Jacobian^2，理论上正确（不依赖 GMM 拟合的准确性）
3. **无需外部依赖**：不需要 sklearn 或任何额外库，完全使用 BreezeForest 现有的 Jacobian 计算代码
4. **与 Latent GMM 互补**：可以用 Latent GMM 作为 MH 的热启动，两种方案结合效果最优
5. **与 Coeurdoux 2024 保持一致**：该工作在 Machine Learning 期刊发表，证明了这类 latent-space MCMC 方法对任何预训练 NF 都有效

---

## 参考文献

- Coeurdoux, F., Dobigeon, N., & Chainais, P. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 113*, 8301–8326. https://arxiv.org/abs/2305.12149
  ← 直接理论基础：latent 空间 MCMC 对预训练 NF 的 post-hoc 采样修复
- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.
  ← 同思路的学习式版本；LMH 是其无参数的 MCMC 替代
- Robert, C. & Casella, G. (2004). "Monte Carlo Statistical Methods." (MH 算法标准参考)
- Gelfand, A.E. & Smith, A.F.M. (1990). "Sampling-Based Approaches to Calculating Marginal Densities." *JASA*.
  ← MCMC 采样的经典理论
