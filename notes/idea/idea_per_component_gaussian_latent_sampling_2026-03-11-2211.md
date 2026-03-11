# Idea: Per-Component Gaussian Latent Sampling (PGLS) — 用拟合高斯替换均匀 z 采样

**创建时间**: 2026-03-11 22:11 UTC  
**推荐优先级**: ⭐⭐ 高优先级（对 LZR 的结构性升级，无需重训练，比矩形 zone 更准确）

---

## 问题定义

`MultiBF.inverse_map()` 的 z 采样策略：

```python
z = torch.rand(n_k, self.dim) * 0.98 + 0.01  # Uniform([0.01, 0.99]^d)
```

这里存在两个假设，二者均不成立：

**假设 1：组件 k 的 latent space 中，cluster k 占据了 [0.01, 0.99]^d 的大部分体积**  
实际：cluster k 在 z 空间只占据一个子区域，其余区域对应其他 cluster 或 inter-cluster 区域。

**假设 2：cluster k 在 z 空间的分布接近均匀**  
实际：cluster k 在 z 空间的分布通常是一个有偏的、各向异性的团（受 CDF 映射的 warping 影响），远非均匀分布。

**LZR（已有 idea_latent_zone_restriction_2026-03-11-1235.md）**尝试通过估计矩形 box `[a_k^1, b_k^1] × ... × [a_k^d, b_k^d]` 来缩小 z 的采样范围。但矩形 box 有如下问题：
1. **维度独立性假设**：矩形 box 假设各维度的 z 分布独立，忽略了维度间的相关性（协方差）
2. **形状误差**：若 cluster k 在 z 空间是斜椭圆形，矩形 box 会同时"遗漏"合法区域和"包含"非法区域
3. **无法适应 z 分布的偏斜**：矩形 box 的中心和范围由百分位数决定，对偏斜分布效果差

**本 Idea 的出发点**：用一个**参数化的多元高斯分布** N(μ_k, Σ_k) 拟合 cluster k 在 z 空间的分布，替代均匀采样。这样既保留了分布的中心和形状信息，又自然处理了维度间的相关性。

---

## 从当前项目代码与已有 idea 中得到的背景判断

**BreezeForest 的 z 空间特性**：

从 `BreezeForest.forward()` 的实现可以看出，z = f_k(x) 的值域是 (0, 1)^d（由 Sigmoid 激活函数保证）。因此 z 空间是 (0, 1)^d 中的一个有界子集。

**z 空间中 cluster 分布的形态**：
- `TreeLayer.forward_helper()` 中：`x = x @ tree_matrix`（线性变换）+ `breeze_bias`（条件偏移）+ `actinorm_init_scale/bias`（标准化）→ `sigmoid(x)`
- 对于一个球形 Gaussian cluster，经过这系列变换后，其在 z 空间的投影大致是一个**有相关结构的椭球形**（被 sigmoid 边界 warp）
- 因此，用**多元高斯**拟合这个椭球形是合理的第一近似（比矩形 box 准确）

**LZR 的实现参考**：LZR 已经建立了 responsibility-based 样本筛选和 forward-map 统计的基础代码框架，PGLS 可以复用此框架，只是在统计量上从"1D 百分位数 box"升级为"多维高斯参数"。

**从 Stimper et al. (2022) 的角度看**：
Stimper 的 Resampled Base Distributions 是一个**可学习的**非均匀 base distribution（通过学习 rejection sampling 概率）。PGLS 是其**无需学习、直接从数据拟合**的实用替代：用训练数据在 z 空间的经验统计（mean + covariance）直接构造一个参数化的 base distribution。

---

## 核心思路

**Per-Component Gaussian Latent Sampling（PGLS）**：

**校准阶段（Post-Training, 一次性）**：
1. 将训练数据 {x_i} 通过组件 k 的正向映射 f_k 得到 z 表示：z_i^k = f_k(x_i)
2. 用 responsibility r_k(x_i) 作为权重，计算 z^k 的加权均值和协方差：
   ```
   μ_k = Σ_i r_k(x_i) * z_i^k / Σ_i r_k(x_i)
   Σ_k = weighted_cov(z^k, weights=r_k(x))
   ```
3. 构造截断多元高斯：z ~ TruncatedGaussian(μ_k, Σ_k, bounds=(0.01, 0.99)^d)

**生成阶段（替换 Uniform 采样）**：
```python
# 原来：z = torch.rand(n_k, self.dim) * 0.98 + 0.01
# 替换为：
z = sample_truncated_gaussian(mu_k, Sigma_k, n_k, bounds=(0.01, 0.99))
```

**关键直觉**：从 N(μ_k, Σ_k) 中采样的 z 集中在 cluster k 的 latent centroid 附近，自然避开了 inter-cluster 区域（这些区域在 z 空间偏离 μ_k 较远，被高斯分布赋予极低概率）。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**从 CDF 映射的角度分析**：

BreezeForest 的 f_k 是一个 CDF 型映射，将数据分布 p_k(x) 映射为 [0,1]^d 上的分布 q_k(z)。理想情况下，若 p_k 是纯 cluster k 的数据（如单高斯），则 q_k(z) 应该近似均匀分布。

但实际（soft-EM 训练）：
- p_k 是所有 K 个 cluster 的混合，f_k 将 cluster k 的数据映射到 q_k(z) 的一个子区域 Z_k
- cluster j (j≠k) 的数据映射到 Z_j^k（Z_k 的补集中的某区域）
- inter-cluster 数据映射到 Z_between^k（通常在 Z_k 和 Z_j^k 之间）

当 z ~ Uniform([0.01, 0.99]^d) 时，会命中 Z_k、Z_j^k 和 Z_between^k 三类区域。**PGLS 通过将采样分布集中在 Z_k（cluster k 的 latent centroid）附近，大幅减少命中 Z_j^k 和 Z_between^k 的概率**。

**与 LZR 的精确对比**：

| 维度 | LZR（矩形 box） | PGLS（多元高斯） |
|------|----------------|----------------|
| 形状 | 轴对齐矩形，忽略维度相关性 | 椭球形，捕获维度相关性（协方差） |
| 对偏斜分布的处理 | 差（百分位数 box 会偏大） | 好（高斯可拟合偏斜分布的大致形态） |
| 采样密度分布 | 在 box 内均匀 | 在中心附近密集，边缘稀疏（类似自然分布） |
| 对 cluster 中心区域的聚焦 | 一般（整个 box 等权） | 好（高斯峰值在 μ_k，有自然的密度梯度） |
| 计算开销（校准） | O(N * d) | O(N * d²)（需要计算协方差矩阵） |
| 截断处理 | 天然（box 边界即截断） | 需要截断/拒绝（z ∈ (0,1)^d 边界）|
| 适用维度 | 任意 d | d ≤ ~10（高维协方差估计需要更多数据） |

对于 BreezeForest 的典型 demo（dim=2），d=2 时多元高斯的协方差估计非常准确，PGLS 显著优于 LZR。

**对比 Stimper et al. (2022) Resampled Base Distributions**：
Stimper 的方法学习一个神经网络来参数化 z 空间的 rejection probability，需要额外的训练步骤。PGLS 用简单的统计估计（加权均值 + 协方差）替代神经网络估计，无需额外训练，计算成本极低（O(Nd²) 一次性）。

---

## 与历史 idea 的关系

**结构性升级 LZR（idea_latent_zone_restriction_2026-03-11-1235.md）**：

LZR 的核心思路（从数据的 z 表示中估计合法采样区域）与 PGLS 完全一致，区别仅在于表示能力：
- LZR：每维度独立的百分位数范围 → 矩形 box
- PGLS：多维高斯（均值 + 协方差）→ 椭球形，更准确

PGLS 可视为 LZR 的"高斯版本"。建议：
- **如果 cluster 在 z 空间是近似球形的**：LZR（box）效果接近 PGLS，开销更低
- **如果 cluster 在 z 空间有明显的各向异性或相关结构**：PGLS 显著优于 LZR

**与 RFS（idea_responsibility_filtered_sampling_2026-03-11-2211.md）的关系**：

PGLS（z 空间采样优化）和 RFS（x 空间过滤）是**两个不同阶段的互补优化**：
- PGLS：在生成 z 时就引导采样到高质量区域（主动防御）
- RFS：在生成 x 后过滤低质量样本（被动过滤）
- **最佳组合**：用 PGLS 替换 Uniform z 采样 → 再用 RFS 过滤残余的 inter-cluster 样本

**与 PLT/Hard-EM 的关系**：
PGLS 在 soft-EM 训练模型上也有效（因为即使组件不完全专一，其在 z 空间也有偏向 cluster k 的统计中心）。PLT 训练后，PGLS 的效果更好（z 空间分布更集中）。

---

## 具体实现建议

### 步骤 1：添加 `calibrate_gaussian_zones()` 到 `MultiBF`

```python
def calibrate_gaussian_zones(self, x_train, clamp_bounds=(0.01, 0.99)):
    """
    Fit per-component Gaussian distribution in latent z-space.
    Uses responsibility-weighted statistics.
    
    :param x_train: training data (N, dim)
    :param clamp_bounds: valid z-range for clamping
    :return: sets self.gaussian_zones = [(mu_k, Sigma_k), ...]
    """
    self.gaussian_zones = []
    lo_bound, hi_bound = clamp_bounds
    
    with torch.no_grad():
        # Compute responsibilities
        log_pi = self.get_mixture_log_weights()
        component_log_probs = []
        for k, bf in enumerate(self.components):
            per_sample_ld = self._per_sample_log_det(bf, x_train)
            component_log_probs.append(log_pi[k] + per_sample_ld)
        
        stacked = torch.stack(component_log_probs, dim=0)  # (K, N)
        log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
        responsibilities = torch.exp(log_resp)  # (K, N)
        
        for k, bf in enumerate(self.components):
            # Forward map all training data through component k
            breeze_list = []
            z_all = bf.forward(x_train, breeze_list)  # (N, dim), values in (0, 1)
            
            # Responsibility weights for component k
            w_k = responsibilities[k]  # (N,)
            w_k_sum = w_k.sum() + 1e-8
            
            # Weighted mean
            mu_k = (w_k.unsqueeze(1) * z_all).sum(0) / w_k_sum  # (dim,)
            
            # Weighted covariance
            z_centered = z_all - mu_k.unsqueeze(0)  # (N, dim)
            Sigma_k = (w_k.unsqueeze(1) * z_centered).T @ z_centered / w_k_sum  # (dim, dim)
            
            # Add small regularization to ensure positive definite
            Sigma_k = Sigma_k + 1e-4 * torch.eye(self.dim)
            
            self.gaussian_zones.append((mu_k, Sigma_k))
            print(f"Component {k}: mu={mu_k.numpy().round(3)}, "
                  f"std_diag={torch.diagonal(Sigma_k).sqrt().numpy().round(3)}")
```

### 步骤 2：添加截断高斯采样工具

```python
def _sample_truncated_gaussian(self, mu, Sigma, n_samples, bounds=(0.01, 0.99), max_attempts=10):
    """
    Sample from Truncated Multivariate Gaussian via rejection sampling.
    Valid z must be in (bounds[0], bounds[1])^d.
    
    :param mu: mean (dim,)
    :param Sigma: covariance (dim, dim)
    :param n_samples: number of samples needed
    :return: samples (n_samples, dim)
    """
    lo, hi = bounds
    samples = []
    remaining = n_samples
    
    # Cholesky decomposition for efficient sampling
    try:
        L = torch.linalg.cholesky(Sigma)
    except Exception:
        # Fallback: use diagonal only
        L = torch.diag(torch.diagonal(Sigma).sqrt())
    
    for _ in range(max_attempts):
        # Sample standard normal and transform
        eps = torch.randn(remaining * 3, self.dim)  # Oversample
        z_candidates = mu.unsqueeze(0) + eps @ L.T  # (remaining*3, dim)
        
        # Keep samples within bounds
        valid_mask = (z_candidates >= lo).all(dim=1) & (z_candidates <= hi).all(dim=1)
        valid_samples = z_candidates[valid_mask]
        
        if len(valid_samples) >= remaining:
            samples.append(valid_samples[:remaining])
            remaining = 0
            break
        else:
            samples.append(valid_samples)
            remaining -= len(valid_samples)
    
    if remaining > 0:
        # Fallback: uniform sampling for remaining slots
        fallback = torch.rand(remaining, self.dim) * (hi - lo) + lo
        samples.append(fallback)
    
    return torch.cat(samples, dim=0)[:n_samples]
```

### 步骤 3：修改 `inverse_map()` 使用 Gaussian z 采样

```python
def inverse_map_with_gaussian_zones(self, n_samples, max_gap=1e-3, decay_ratio=1.0, n_sigma=2.0):
    """
    Generate samples using per-component Gaussian latent zones.
    Requires calibrate_gaussian_zones() to be called first.
    
    :param n_sigma: number of standard deviations to cover (controls spread)
    """
    assert hasattr(self, 'gaussian_zones'), "Call calibrate_gaussian_zones() first"
    
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)

    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        mu_k, Sigma_k = self.gaussian_zones[k]
        
        # Scale covariance by n_sigma (larger = more spread, smaller = tighter)
        Sigma_k_scaled = Sigma_k * (n_sigma / 2.0) ** 2
        
        # Sample z from Gaussian zone
        z = self._sample_truncated_gaussian(mu_k, Sigma_k_scaled, n_k)
        
        x_k = self.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
        results[mask] = x_k

    return results
```

### 步骤 4：在 demo 中集成

```python
# 训练完成后
all_batch = ...  # 全量训练数据
all_batch_norm = (all_batch - mean) / std

with torch.no_grad():
    # 校准 Gaussian zones（约 0.5 秒）
    mbf.calibrate_gaussian_zones(all_batch_norm)
    
    # 生成（使用 Gaussian z 采样）
    samples = mbf.inverse_map_with_gaussian_zones(n_samples=data_size, n_sigma=2.0)
    samples = samples * std + mean
```

### 超参数调优指南

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `n_sigma` | 1.5 – 3.0 | 控制生成的扩散程度。1.5：紧凑（只生成 cluster 中心区域）；3.0：宽松（包含 cluster 边缘） |
| `clamp_bounds` | (0.01, 0.99) | 不需要调整，与原有 Uniform 采样的边界一致 |
| 高斯 vs. 矩形 | 低维（d≤5）用高斯 | 高维时协方差估计噪声大，考虑对角化（独立维度） |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **高斯近似误差** | 若 cluster 在 z 空间的真实形状是非高斯的（如月牙形），高斯拟合会有误差 | 对于低维数据（2D），可用 GMM（sklearn.mixture.GaussianMixture）替代单高斯拟合 |
| **协方差矩阵奇异** | 当组件专一化不足，z 空间的 cluster 数据在某维度方差极小，Σ 可能几乎奇异 | 已添加正则化 `Σ + 1e-4*I`；可增大正则化系数 |
| **截断高斯接受率低** | 若 μ_k 接近 (0,1)^d 的边界，截断会丢弃大量样本 | μ_k 通常在 (0.3, 0.7) 附近（cluster 数据的 CDF 中间区域），不太可能接近边界；如出现则用 fallback uniform |
| **z 空间 cluster 重叠** | 当模型组件重叠时，两个组件的 μ_k 可能很近，高斯区域重叠 | 结合 PLT 训练减少组件重叠；或使用 RFS 做额外过滤 |
| **高维时协方差估计噪声** | dim > 5 时，需要更多数据才能得到稳定的协方差估计 | 使用对角协方差（`Σ_k = diag(var_k^1, ..., var_k^d)`）；BreezeForest demo 通常是 d=2，不受此影响 |

---

## 推荐优先级

**⭐⭐ 高优先级（LZR 的有效替代，在 2D 场景下有明确优势）**

理由：
1. **比 LZR 更准确**：多元高斯捕获了维度间相关性，对非轴对齐 cluster 效果更好
2. **无需重训练**：与 LZR 一样，是 post-training calibration 方法，可立即在已有模型上验证
3. **理论更扎实**：加权协方差估计是统计中的标准方法，比百分位数 box 有更清晰的理论解释
4. **与 Stimper et al. (2022) 的连接**：PGLS 是 Stimper 的 Resampled Base Distribution 的参数化（数据驱动）简化版，核心思路有文献支撑
5. **可与 RFS 叠加**：PGLS（限制 z 采样范围）+ RFS（过滤低质量 x）是双层防护，比任何单一方法更强

**与 LZR 的选择建议**：
- 如果数据维度 d=2（BreezeForest 主要 demo 场景）：**优先选择 PGLS**（协方差准确，实现不复杂）
- 如果 cluster 在 z 空间接近轴对齐矩形：**LZR 足够**（更简单）
- 如果需要最强效果：**PGLS + RFS 组合**

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.  
  (Core motivation: replacing Uniform latent sampling with a data-driven base distribution)
- Handley, W. et al. (2023). "Piecewise Normalizing Flows." *arxiv 2305.02930*.  
  (Pre-clustering + per-cluster flow; cluster-specific z statistics are the natural calibration data)
- Mardia, K.V., Kent, J.T., and Bibby, J.M. (1979). *Multivariate Analysis.* Academic Press.  
  (Weighted covariance estimation; theoretical foundation for Gaussian z-space fitting)
- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 2024*.  
  (Latent-space sampling strategies for normalizing flows; related idea of non-uniform latent sampling)
