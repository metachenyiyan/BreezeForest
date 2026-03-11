# Idea: Per-Component Latent KDE Sampling for MultiBF Inference

**创建时间**: 2026-03-11 23:40 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（LZR 的有原则替代，即插即用，无需重训练）

---

## 问题定义

MultiBF 在**生成阶段**存在根本性缺陷：

```python
# 当前 MultiBF.inverse_map() 中：
z = torch.rand(n_k, self.dim) * 0.98 + 0.01  # z ~ Uniform([0.01, 0.99]^d)
x_k = self.components[k].inverse_map(z, ...)
```

核心问题：`z ~ Uniform([0.01, 0.99]^d)` 均匀覆盖整个 latent 空间。但训练数据只占据 latent 空间的**一个子区域**（cluster k 的数据映射到的区域）。大量 z 值对应的是：
- 其他 cluster（cluster j≠k）在组件 k 的 latent 空间中的投影区域
- cluster 之间的空区域在 latent 空间中的对应位置

均匀采样意味着大量生成样本落在 cluster 之外或 cluster 之间。

---

## 从当前代码与已有 idea 中得到的背景判断

阅读 `model/BreezeForest.py` 后：
- `inverse_map()` 在 [0.01, 0.99]^d 上均匀采样 z，然后通过 bisection 反演
- `forward()` 将数据 x 映射到 [0.01, 0.99]^d（每维输出都是 Sigmoid → (0,1)）
- 训练数据通过 `forward()` 得到的 latent 表示 **不均匀分布在 [0,1]^d 中**；它们集中在某些子区域

已有 **Idea 2（LZR，12:35）** 发现了这个问题并提出了轴对齐包围盒（bounding box）方案：
- 计算训练数据在各组件 latent 空间中的分位数，得到 `[a_k, b_k]^d`
- 生成时限制 `z ~ Uniform([a_k, b_k]^d)`

**LZR 的局限性（本 Idea 所解决）**：
1. **轴对齐矩形**：仅按各维度独立截断，忽略了 latent 维度之间的相关性。例如，如果 z1 和 z2 高度相关（latent 表示沿某对角方向分布），矩形会包含大量低密度区域
2. **包围盒内部仍有空洞**：多模态 latent 分布（如 cluster A 的 latent 在左上角，cluster B 的在右下角）会使轴对齐矩形包含中间的低密度区域
3. **无法表示 latent 密度的形状**：只用边界，不知道内部哪些 z 密度高、哪些低

**本 Idea 用 KDE（核密度估计）替代轴对齐矩形**，直接建模 latent 空间中训练数据的实际密度分布。

---

## 核心思路

**训练后校准（Post-Training Calibration with KDE）**：

1. 对训练数据，通过每个组件 k 做正向传播，得到 latent 表示 `z_i^k = f_k(x_i)`
2. 根据 responsibility 选出"属于"组件 k 的样本子集（高 responsibility 样本）
3. 在这个子集的 latent 表示上**拟合一个 KDE（核密度估计）**
4. 生成时：从 KDE_k 采样 z，然后 `x = f_k^{-1}(z)`

**KDE 采样方法**（针对 [0,1]^d 的有界空间）：
- 用 rejection sampling：先从 KDE 采样 z，拒绝不在 [0.01, 0.99]^d 内的样本（自动 boundary safe）
- 或用 importance resampling（SIR）：从 Uniform(0.01, 0.99)^d 生成候选 z，按 KDE 密度重采样

**核选择**：
- 推荐 **Epanechnikov 核**（在有界支撑上效率最高）
- 或简单高斯核（带宽用 Silverman's rule 自动确定）

对于高维 d > 5 的情况，可以退化为**对角协方差多变量高斯**（比完整 GMM 更鲁棒）：
```
KDE_k ≈ N(μ_k, Σ_k_diag)  # 用 latent 样本的均值和对角协方差
```

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**完整机制**：

1. 组件 k 的训练数据（cluster k）通过 `f_k` 映射到 latent 空间，形成一个紧凑的 latent cluster
2. KDE_k 精确地建模这个 latent cluster 的形状（包括各维度相关性）
3. 从 KDE_k 采样 z，保证 z 在 latent cluster 的高密度区域内
4. `f_k^{-1}(z)` 映射回数据空间，输出在 cluster k 附近

**对比 LZR（Idea 2）**：

| 方面 | LZR（Idea 2） | KDE Sampling（本 Idea） |
|------|--------------|----------------------|
| latent 区域估计 | 轴对齐矩形（分位数） | KDE（核密度估计） |
| 捕获维度相关性 | 否 | 是 |
| 处理 latent 内部空洞 | 否 | 是 |
| 计算开销 | 低（仅分位数） | 中（KDE拟合+采样） |
| 需要 responsibility 计算 | 是 | 是 |
| 理论支撑 | Stimper 2022（近似） | GMM/KDE 密度估计（精确） |

**支持文献**：
- Stimper et al. (2022)「Resampling Base Distributions」：直接支持用学习到的密度替代均匀基分布
- BMVC 2024 「Multimodal base distributions in conditional flow matching」：验证了 GMM 基分布显著减少 mode 之间的误生成
- arxiv 2512.04954 (2024)：「GMM initialization significantly improves reconstruction fidelity for multi-modal posteriors」

---

## 与历史 idea 的关系

- **替代/升级 Idea 2（LZR，12:35）**：KDE Sampling 是 LZR 的严格超集。当 KDE 退化为均匀分布时，等价于原始采样；当 KDE 退化为轴对齐矩形时，等价于 LZR。KDE 提供了更精确的 latent 区域建模。
- **与 Idea 1（Hard-EM，12:30）或 DA-EM（本轮 Idea 1）互补**：训练时专一化 + 采样时 KDE 约束是最强组合
- **与 Idea 3（ICDR，12:40）的关系**：若 ICDR 训练后各组件更专一，则 KDE_k 的 latent cluster 更紧凑，KDE 估计更准确
- **与历史 notes 的关系**：LZR 引用的 Stimper et al. (2022) 与本 Idea 思路一致，是学习版本（需要神经网络）；本 Idea 是数据驱动的简单版本（KDE 无需额外学习）

---

## 具体实现建议

### 步骤 1：添加 calibrate_latent_kde() 方法到 MultiBF

```python
def calibrate_latent_kde(self, x_train, bandwidth='silverman', resp_threshold=None):
    """
    Fit per-component KDE on latent representations of training data.
    
    :param x_train: training data (N, dim)
    :param bandwidth: 'silverman' for automatic, or float for fixed bandwidth
    :param resp_threshold: responsibility threshold for sample selection
                          (default: 1/K, i.e., above-average responsibility)
    """
    from sklearn.neighbors import KernelDensity
    import numpy as np
    
    if resp_threshold is None:
        resp_threshold = 1.0 / self.n_components
    
    self.latent_kdes = []
    self.latent_kde_samples = []  # Store samples for importance resampling
    
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
            resp_k = responsibilities[k]  # (N,)
            
            # Select high-responsibility samples
            mask = resp_k > resp_threshold
            if mask.sum() < 20:
                topk = max(20, int(0.15 * len(resp_k)))
                _, idx = torch.topk(resp_k, topk)
                mask = torch.zeros(len(resp_k), dtype=torch.bool)
                mask[idx] = True
            
            # Forward pass to get latent representations
            x_k = x_train[mask]
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list)  # (n_k, dim)
            z_k_np = z_k.cpu().numpy()  # (n_k, dim)
            
            # Fit KDE on latent representations
            if bandwidth == 'silverman':
                # Silverman's rule of thumb: h = (4/(d+2))^(1/(d+4)) * n^(-1/(d+4)) * sigma
                n, d = z_k_np.shape
                sigma_avg = np.std(z_k_np, axis=0).mean()
                bw = (4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4)) * sigma_avg
                bw = max(bw, 0.01)  # Minimum bandwidth
            else:
                bw = bandwidth
            
            kde = KernelDensity(kernel='gaussian', bandwidth=bw)
            kde.fit(z_k_np)
            
            self.latent_kdes.append(kde)
            self.latent_kde_samples.append(z_k_np)
            
            print(f"Component {k}: {mask.sum().item()} samples, "
                  f"latent range=[{z_k_np.min(axis=0).round(3)}, {z_k_np.max(axis=0).round(3)}], "
                  f"bandwidth={bw:.4f}")
```

### 步骤 2：添加 KDE-guided inverse_map

**方法 A：直接从 KDE 采样（rejection sampling 确保在 [0.01, 0.99]^d 内）**

```python
def inverse_map_with_kde(self, n_samples, max_gap=1e-3, decay_ratio=1.0,
                          n_candidates_multiplier=5):
    """
    Generate samples using per-component KDE-guided latent sampling.
    
    :param n_candidates_multiplier: oversample by this factor, then reject out-of-bounds
    """
    assert hasattr(self, 'latent_kdes'), "Call calibrate_latent_kde() first"
    
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)
    
    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        # Sample from KDE with rejection to stay in [0.01, 0.99]^d
        valid_z = []
        n_needed = n_k
        while len(valid_z) < n_needed:
            n_try = (n_needed - len(valid_z)) * n_candidates_multiplier
            z_candidates = self.latent_kdes[k].sample(n_try)  # (n_try, dim)
            # Filter to valid range
            valid_mask = np.all((z_candidates >= 0.01) & (z_candidates <= 0.99), axis=1)
            valid_z.extend(z_candidates[valid_mask].tolist())
        
        z = torch.tensor(valid_z[:n_k], dtype=torch.float32)  # (n_k, dim)
        x_k = self.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
        results[mask] = x_k
    
    return results
```

**方法 B：Importance Resampling（SIR，更稳定，无需 rejection loop）**

```python
def inverse_map_with_kde_sir(self, n_samples, n_sir_candidates=2000, **kwargs):
    """
    Sample via Sequential Importance Resampling (SIR) from KDE.
    
    For each component k:
    1. Generate n_sir_candidates from Uniform([0.01, 0.99]^d)
    2. Compute KDE density at each candidate
    3. Resample according to KDE weights
    4. Apply inverse_map on resampled z
    """
    assert hasattr(self, 'latent_kdes'), "Call calibrate_latent_kde() first"
    
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)
    
    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        # SIR: generate candidates from uniform, resample by KDE weights
        z_uniform = np.random.rand(n_sir_candidates, self.dim) * 0.98 + 0.01
        log_weights = self.latent_kdes[k].score_samples(z_uniform)  # (n_sir_candidates,)
        log_weights -= log_weights.max()  # numerical stability
        weights_normalized = np.exp(log_weights)
        weights_normalized /= weights_normalized.sum()
        
        idx = np.random.choice(n_sir_candidates, size=n_k, replace=True, p=weights_normalized)
        z = torch.tensor(z_uniform[idx], dtype=torch.float32)  # (n_k, dim)
        
        x_k = self.components[k].inverse_map(z, **kwargs)
        results[mask] = x_k
    
    return results
```

**推荐：优先用方法 B（SIR），因为它总能返回 n_k 个样本，无需 rejection loop。**

### 步骤 3：demo 集成

```python
# 训练完成后的校准 + KDE 采样
import numpy as np

# 1. 收集所有训练数据
all_data_loader = DataLoader(distribution, batch_size=3000, shuffle=True)
all_batch, _ = next(iter(all_data_loader))
all_batch = (all_batch - mean) / std

# 2. 校准 KDE
with torch.no_grad():
    mbf.calibrate_latent_kde(all_batch, bandwidth='silverman')

# 3. 生成样本
with torch.no_grad():
    samples = mbf.inverse_map_with_kde_sir(n_samples=data_size, n_sir_candidates=3000)
    samples = samples * std + mean
```

### 超参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `bandwidth` | 'silverman' | 自动带宽（数据驱动），通常最优 |
| `resp_threshold` | 1/K | 选择 responsibility 高于平均值的样本 |
| `n_sir_candidates` | 1000~5000 | SIR 候选数。越多越精确，5000 时效果接近理论最优 |
| `n_candidates_multiplier` | 5（方法A） | 过采样倍数，确保足够有效样本 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **KDE 在高维退化** | d > 5 时 KDE 需大量样本才可靠 | 对高维用对角高斯代替 KDE，或 PCA 降维后在低维拟合 |
| **KDE 覆盖不足** | 训练数据量少时 KDE 低估 cluster 边缘密度 | 增大带宽（稍宽松），或用轻微向外扩展的高斯核 |
| **sklearn 依赖** | 生产环境可能无 sklearn | 提供 numpy 实现的高斯 KDE 备选（几十行代码） |
| **需要 calibration 数据** | 需要对训练集做一次 forward pass | 用训练集本身即可；对内存有限场景可分批处理 |
| **SIR 有效样本不足** | 若 KDE 非常集中，大量候选被低权重，SIR 有效数少 | 增大 `n_sir_candidates`；或用方法 A（rejection） |
| **组件未专一时 KDE 不准** | Soft-EM 训练的组件 latent 可能包含多个 cluster → KDE 不集中 | 与 DA-EM 训练结合；或使用更严格的 responsibility 阈值（如 top 30%） |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（替代 LZR，立即可验证，无需重训练）**

理由：
1. **即插即用**：只需在已训练模型上运行 calibrate_latent_kde()，无需重训练
2. **直接解决根因**：从数据中学习每个组件的 latent 密度形状，生成时只从高密度区域采样
3. **严格优于 LZR（Idea 2）**：KDE 捕获维度相关性和内部密度结构，LZR 的轴对齐矩形是 KDE 的粗略近似
4. **有充分文献支持**：multimodal base distribution 文献（BMVC 2024, arxiv 2512.04954, Stimper 2022）均验证了这类方法的有效性
5. **与 DA-EM 训练完美互补**：DA-EM 使组件专一化 → KDE 估计的 latent cluster 更紧凑 → 生成质量进一步提升

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a/stimper22a.pdf [KDE/rejection 采样作为基分布的先驱工作]
- BMVC 2024. "Multimodal base distributions in conditional flow matching generative models." https://bmvc2024.org/proceedings/492/ [验证 GMM 基分布减少 mode 间误生成]
- arxiv 2512.04954 (2024). "Amortized Inference of Multi-Modal Posteriors using Likelihood-Weighted Normalizing Flows." [GMM 初始化显著改善多模态重建保真度]
- Silverman, B.W. (1986). "Density Estimation for Statistics and Data Analysis." Chapman & Hall. [Silverman's rule，自动带宽估计]
- arxiv 2305.02930 (2023). "Piecewise Normalizing Flows." [分段流，与 per-component KDE 思路一致]
