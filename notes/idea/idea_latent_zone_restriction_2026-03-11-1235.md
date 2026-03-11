# Idea: Responsibility-Guided Latent Zone Restriction (LZR)

**创建时间**: 2026-03-11 12:35 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（可立即实施，无需重训练）

---

## 问题定义

MultiBF 在**生成阶段**存在结构性缺陷：

对组件 k 做 inverse_map 时，采样 z ~ Uniform([0.01, 0.99]^d)，然后 x = f_k^{-1}(z)。

问题在于 f_k 是整个数据空间到 [0.01, 0.99]^d 的**全局双射**，因此其逆映射 f_k^{-1} 将 [0.01, 0.99]^d 中的**每一个** z 值都映射回某个 x。

关键洞察：
- f_k 会将 cluster k 的数据（高密度区域）映射到 [0.01, 0.99]^d 的某个**子区域 Z_k**（因为 cluster k 的 CDF 变化快，z 值集中在某处）
- f_k 还会将 cluster j≠k 的数据和 cluster 之间的区域映射到 Z_k 的**补集 Z_k^c**
- 当前生成策略从整个 [0.01, 0.99]^d 采样，包括了 Z_k^c，导致生成出 cluster j 或 inter-cluster 的点

**修复方向**：通过后处理（calibration）步骤识别每个组件 k 的 **latent cluster zone Z_k**，并在生成时将采样限制在 Z_k 内。

---

## 核心思路

**训练后校准（Post-Training Calibration）**：
1. 对训练数据中分配给组件 k 的样本，通过 f_k 正向传播，得到其 latent 表示 z_i^k = f_k(x_i)
2. 统计这些 z_i^k 的分布范围，确定 Z_k = [a_k, b_k]^d（各维度的百分位数边界）
3. 将 Z_k 作为组件 k 的"latent 合法采样区域"

**生成时约束**：
- 不再从 Uniform([0.01, 0.99]^d) 采样
- 改为从 Uniform(Z_k) = Uniform([a_k^1, b_k^1] × ... × [a_k^d, b_k^d]) 采样
- 然后用 f_k^{-1} 做 inverse_map

由于 Z_k 就是 cluster k 数据的 latent representation 所在区域，从 Z_k 采样再反演，必然会生成接近 cluster k 的样本。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**直觉论证**：

假设 cluster A 和 cluster B 是两个分离的高斯团：
- f_k（组件 k 对应 cluster A）会将 cluster A 的点映射到 Z_k，而 cluster B 的点映射到 Z_k^c
- 如果我们的 z 只从 Z_k 中采样，那么 f_k^{-1}(z) 几乎不可能产生 cluster B 的点（因为 cluster B 的点没有对应 Z_k 内的 z）
- inter-cluster 的点的 latent 表示也在 Z_k^c 附近，从 Z_k 采样不会映射到那里

**数学严格性**：

设 h_k(x) = f_k(x) 是组件 k 的正向映射（data space → [0,1]^d）。由单射性：

```
{x : f_k(x) ∈ Z_k} = {cluster k 的数据区域}（近似）
```

因此 f_k^{-1}(Z_k) ≈ {cluster k 的数据区域}。从 Z_k 采样 z，再取 f_k^{-1}(z)，会限制在 cluster k 区域内。

**对比 Resampling Base Distributions（Stimper et al., 2022）**：
Stimper et al. 通过学习一个 rejection sampling 的 base distribution 来修复 topology 问题。本 Idea 的 LZR 是一个更简单的等价方案：直接从数据中估计 Z_k 的范围，无需额外的学习步骤。

---

## 与历史 idea 的关系

**全新 idea**（首次提出）。现有文献和历史 notes 都没有提到针对 MultiBF 的 latent zone restriction 方法。

与 Idea 1（Hard-EM）的关系：**互补**。
- Idea 1 是 training-time 修复
- 本 Idea 是 inference-time 修复
- 两者可以叠加使用：Hard-EM 使组件专一化，LZR 进一步约束生成区域
- **即使不用 Hard-EM，本 Idea 也能单独改善生成质量**

与 Stimper et al. (2022) "Resampling Base Distributions" 的关系：同一思路的简化实现版本，无需额外学习。

---

## 具体实现建议

### 步骤 1：添加 calibrate_latent_zones() 方法到 MultiBF

```python
def calibrate_latent_zones(self, x_train, percentile_low=5.0, percentile_high=95.0):
    """
    Compute per-component latent zones from training data.
    
    :param x_train: training data tensor (N, dim)
    :param percentile_low: lower percentile for zone boundary (default 5%)
    :param percentile_high: upper percentile for zone boundary (default 95%)
    """
    self.latent_zones = []
    
    with torch.no_grad():
        # Compute responsibilities
        log_pi = self.get_mixture_log_weights()
        component_log_probs = []
        for k, bf in enumerate(self.components):
            per_sample_ld = self._per_sample_log_det(bf, x_train)
            component_log_probs.append(log_pi[k] + per_sample_ld)
        
        stacked = torch.stack(component_log_probs, dim=0)   # (K, N)
        log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
        responsibilities = torch.exp(log_resp)              # (K, N)
        
        for k, bf in enumerate(self.components):
            # Get latent representations for high-responsibility samples
            resp_k = responsibilities[k]  # (N,)
            
            # Select top-50% by responsibility (or use threshold > 1/K)
            threshold = 1.0 / self.n_components  # uniform threshold
            mask = resp_k > threshold
            
            if mask.sum() < 10:
                # Fallback: use top 20% of samples by responsibility
                topk = int(0.2 * len(resp_k))
                _, idx = torch.topk(resp_k, topk)
                mask = torch.zeros_like(resp_k, dtype=torch.bool)
                mask[idx] = True
            
            # Forward pass selected samples through component k
            x_k = x_train[mask]
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list)  # shape: (n_k, dim)
            
            # Compute zone boundaries (percentiles per dimension)
            lo = torch.tensor([
                torch.quantile(z_k[:, d], percentile_low / 100.0).item()
                for d in range(self.dim)
            ])
            hi = torch.tensor([
                torch.quantile(z_k[:, d], percentile_high / 100.0).item()
                for d in range(self.dim)
            ])
            
            # Clamp to valid range [0.01, 0.99]
            lo = lo.clamp(min=0.01)
            hi = hi.clamp(max=0.99)
            
            self.latent_zones.append((lo, hi))
    
    print(f"Calibrated latent zones for {len(self.latent_zones)} components:")
    for k, (lo, hi) in enumerate(self.latent_zones):
        print(f"  Component {k}: lo={lo.numpy().round(3)}, hi={hi.numpy().round(3)}")
```

### 步骤 2：修改 inverse_map() 使用 latent zones

```python
def inverse_map_with_zones(self, n_samples, max_gap=1e-3, decay_ratio=1.0):
    """
    Generate samples using per-component latent zone restriction.
    Requires calibrate_latent_zones() to be called first.
    """
    assert hasattr(self, 'latent_zones'), "Call calibrate_latent_zones() first"
    
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)

    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue

        lo_k, hi_k = self.latent_zones[k]
        # Sample from component k's zone only
        z = torch.rand(n_k, self.dim) * (hi_k - lo_k) + lo_k

        x_k = self.components[k].inverse_map(
            z, max_gap=max_gap, decay_ratio=decay_ratio
        )
        results[mask] = x_k

    return results
```

### 步骤 3：在 demo_multi_bf.py 中添加校准步骤

```python
# 训练完成后：
# 1. 校准 latent zones
all_data_loader = DataLoader(distribution, batch_size=3000, shuffle=True)
all_batch, _ = next(iter(all_data_loader))
all_batch = (all_batch - mean) / std
with torch.no_grad():
    mbf.calibrate_latent_zones(all_batch, percentile_low=5.0, percentile_high=95.0)

# 2. 使用 zone-restricted 生成
with torch.no_grad():
    samples = mbf.inverse_map_with_zones(n_samples=data_size)
    samples = samples * std + mean
```

### Zone 边界调优建议

| percentile_low | percentile_high | 效果 |
|---------------|----------------|------|
| 5% | 95% | 保守（避免 outlier latent 值，生成更紧凑） |
| 10% | 90% | 激进（更严格限制，可能截断部分合法样本） |
| 2% | 98% | 宽松（近似原始生成，修复有限但不截断） |

推荐从 5%-95% 开始，根据可视化效果调整。

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **Zone 估计不准** | 如果组件没有专一化（soft-EM 训练），Z_k 可能包含多个 cluster 的点 | 与 Idea 1（Hard-EM）结合使用；或使用更严格的 responsibility 阈值选取"纯"样本 |
| **截断合法样本** | 过严格的百分位数会截断 cluster 边缘的合法样本 | 用宽松百分位数（2%-98%）或对 zone 边界做轻微膨胀 |
| **Zone 重叠** | 如果两个组件的 Z_k 重叠，仍然会产生 inter-cluster 样本 | 可以对重叠区域做额外处理（去重叠），或接受少量重叠 |
| **多维度 Zone 估计** | 各维度独立估计边界，忽略维度间的相关性 | 可以升级到用 convex hull 或 PCA 主成分上的边界 |
| **需要校准数据** | 需要一批训练数据做 calibration forward pass | 使用训练集本身即可，无需额外数据 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（与 Idea 1 并列，且无需重训练）**

理由：
1. **零成本实施**：不需要修改训练代码，不需要重训练，只需在已训练模型上运行一次 calibration
2. **即时可验证**：可以在现有已训练的 MultiBF 模型上立即验证效果
3. **直接针对症状**：从根本上限制了生成时的 latent 采样范围，直接阻断 inter-cluster 路径
4. **可与 Idea 1 叠加**：Hard-EM 训练 + LZR 采样是最强的组合方案
5. **有理论支撑**：与 Stimper et al. (2022) 的 resampled base distribution 思路一致，是其数据驱动的简化版

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a.html
- Pires, G. & Figueiredo, M. (2020). "Variational Mixture of Normalizing Flows." *ESANN 2020*. Latent space partitioning idea.
- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 2024*. https://arxiv.org/abs/2305.12149
