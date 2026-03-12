# Idea: Per-Component Multivariate Gaussian Latent Base Distribution

**创建时间**: 2026-03-12 01:00 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（升级版 LZR，推断时直接可用）

---

## 问题定义

MultiBF 在生成阶段对每个组件 k 采样 z ~ Uniform(0.01, 0.99)^d，然后通过 bisection 计算 x = f_k^{-1}(z)。问题在于：

**Uniform(0.01, 0.99)^d 是 f_k 的整个值域，而 cluster k 的数据只映射到其中一个小的子区域 Z_k。**

从全 uniform 采样 z 时，大量 z 值落在 Z_k 以外，映射到其他 cluster 或 cluster 之间的区域。

现有 Idea 2（LZR, 2026-03-11-1235）提出用 **axis-aligned 矩形边界** 估计 Z_k（各维度独立估计百分位数范围），这是正确方向。但矩形边界有显著缺陷：

1. **忽略维度间协方差**：如果 cluster k 在 latent space 中形成一个斜长的椭圆（维度之间有相关性），矩形边界会包含椭圆之外的大量无效区域，仍然会采样到椭圆外的 z 值。
2. **不是概率分布**：从矩形内均匀采样不反映 Z_k 内的密度差异（cluster 中心的 z 值密度更高，边缘密度更低），导致生成的 x 在 cluster 内分布不均匀。
3. **无法捕捉旋转结构**：BreezeForest 的 autoregressive 结构会在 latent space 产生有方向性的映射，矩形框可能严重高估 Z_k 的范围。

---

## 从项目代码和已有 Idea 中得到的背景判断

- BreezeForest 的 `forward()` 将 x 映射到 [0,1]^d 的紧致空间（最后一层是 Sigmoid，输出在 (0,1) 之间）。
- 训练后，对训练数据做 forward pass，可以得到每个样本的 latent 表示 z_i = f_k(x_i)。
- `inverse_map()` 中 bisection 使用 `distribution` 参数（默认 Normal(0,1)）在 CDF 空间做粗搜索，然后在实数空间精细搜索。**关键**：bisection 中已经有 `distribution` 参数，可以传入自定义分布用于引导搜索范围。
- Idea 2（LZR）的 `calibrate_latent_zones()` 已经实现了从训练数据收集 latent 表示并估计边界的逻辑，本 Idea 在此基础上升级为拟合完整 MVN。
- `tools.py` 的 `bisection()` 函数中的 `distribution` 参数允许自定义，将非 Normal 分布传入可改变粗搜索的 CDF 变换方式。

**外部调研关键发现**：
- **HGAD（Liu et al., 2024, arXiv:2403.13349）**：提出 Hierarchical Gaussian Mixture 作为 normalizing flow 的 latent prior，强制不同类别的 latent 表示映射到不同的 Gaussian 组件（而非共享同一个 Standard Normal prior），使用 mutual information maximization 来结构化 latent space。这验证了为每个流组件使用专属 Gaussian base 分布的有效性。
- **VampPrior Mixture Model（Tomczak & Welling, 2018）**：VAE 中使用混合高斯作为 prior，而非标准 Normal，在密度估计和聚类上均有改善。同一思路可应用于 normalizing flows 的 base distribution 设计。
- **Piecewise Normalizing Flows（2023）**：每个 component flow 的 base 就是 Standard Normal，但数据被预聚类后，每个 cluster 内部数据接近 Normal，从而 Standard Normal 作为 base 是合适的。本 Idea 是对未做预聚类的情况下改善 base distribution 的方案。

---

## 核心思路

**训练后校准**（无需重训练）：对每个组件 k，收集其高 responsibility 训练样本的 latent 表示，拟合一个 **Multivariate Normal（MVN）分布 N(μ_k, Σ_k)**，并在生成时从该 MVN 采样 z（截断到 [0.01, 0.99]^d），代替原来的 Uniform 采样。

**数学直觉**：
- 设 X_k = {x_i : component k 负责} 是 cluster k 的数据
- Z_k = {f_k(x_i) : x_i ∈ X_k} 是 X_k 在 latent space 的映射
- 由 BreezeForest 的归一化特性，Z_k ⊂ [0,1]^d，且在 [0,1]^d 内形成一个紧致分布
- 拟合 MVN N(μ_k, Σ_k) 到 Z_k 后，从 N(μ_k, Σ_k) 截断采样的 z 几乎都落在 Z_k 内
- f_k^{-1}(z) 对这些 z 映射到 X_k 所在区域，避免 inter-cluster 生成

**与 LZR（矩形边界）的核心区别**：

| 方面 | LZR（Idea 2）矩形边界 | 本 Idea（MVN 椭圆边界） |
|------|----------------------|------------------------|
| 形状 | 轴对齐矩形（无协方差） | 椭圆（捕捉协方差/旋转） |
| 密度分布 | Uniform（等权重） | 高斯（中心权重高，边缘低） |
| 生成质量 | 中等（矩形可能包含无效区域） | 高（椭圆紧贴 cluster 形状） |
| 实现复杂度 | 简单（quantile 计算） | 中等（MVN 拟合，约 10 行代码） |
| 是否概率分布 | 否（均匀采样） | 是（有正式密度） |

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**直接因果链**：

1. 当前问题：z ~ Uniform(0.01, 0.99)^d 包含大量"坏 z"，这些 z 对应 inter-cluster 区域
2. 根因：Uniform 分布对 [0,1]^d 所有点等权，不区分 cluster 内外
3. MVN 修复：N(μ_k, Σ_k) 在 Z_k 中心有高概率，在 Z_k 外（即其他 cluster 或 inter-cluster 的 latent 区域）概率指数级衰减
4. 结果：从 N(μ_k, Σ_k) 采样 z 几乎不可能落在 inter-cluster 的 latent 区域

**定量估计**：
- 若 Z_k 在某维度的宽度为 0.3（例如从 [0.3, 0.6]），而 latent space 总宽度为 0.98（从 0.01 到 0.99）：
  - Uniform 采样时，落在 Z_k 维度范围内的概率 ≈ 0.3/0.98 ≈ 30%
  - 对 2D，两个维度都落在 Z_k 的概率 ≈ 9%（仅约 1/10 的点有效）
  - MVN 采样（标准差=Z_k宽度/4）时，95% 的样本落在 Z_k 范围内
  - 对 2D，两个维度都有效的概率 ≈ 90%（提升约 10 倍）

**对单个 BreezeForest（非 MultiBF）的适用性**：
- 单 BF 没有组件分配，但可以用 K-Means 将训练数据聚类，对每个 cluster 估计其 latent 表示的 MVN
- 生成时，先选一个 cluster（按 cluster 大小比例），再从该 cluster 的 MVN 采样 z，然后通过 BF 的 inverse_map 生成 x
- 这是对单 BF 的 inference-time 修复，不需要改变 BF 结构

---

## 与历史 Idea 的关系

**升级并替代 Idea 2（Latent Zone Restriction, 2026-03-11-1235）**

| 维度 | Idea 2（LZR 矩形边界） | 本 Idea（MVN 椭圆） |
|------|------------------------|---------------------|
| Latent 空间估计 | 每维度独立百分位数（矩形） | 完整 MVN（均值+协方差） |
| 协方差捕捉 | 否 | 是 |
| 生成分布 | 截断 Uniform | 截断 MVN（更真实） |
| 适用于单 BF | 否（需要 MultiBF） | 是（可结合 K-Means） |
| 实现复杂度 | 低 | 中等 |

**结论**：本 Idea 是 Idea 2 的明确升级。如果实现资源有限，可以先实现 LZR（Idea 2），然后升级到本 Idea 以获得更好的 latent 结构捕捉。

与 **Idea 1（Hard-EM）** / **K-Means Pre-Assign（本轮 Idea A）** 的关系：**互补**。  
- 训练阶段：K-Means Pre-Assign 使组件专一化  
- 推断阶段：本 Idea 通过 MVN 采样进一步确保 z 只落在 cluster k 的 latent 区域

与 **Idea 3（ICDR）** 的关系：**ICDR 是训练时的间接修复，本 Idea 是推断时的直接修复**。两者可以并用，但本 Idea 无需重训练，优先级更高。

---

## 具体实现建议

### 步骤 1：收集 per-component latent 表示

```python
def fit_component_mvn(mbf, x_train, device='cpu'):
    """
    For each component k in MultiBF, fit a Multivariate Normal to the
    latent representations of samples assigned to component k.
    
    :param mbf: trained MultiBF
    :param x_train: normalized training data (N, dim)
    :return: list of (mean, cov_matrix) per component
    """
    mbf.eval()
    mvn_params = []

    with torch.no_grad():
        # Compute responsibilities
        log_pi = mbf.get_mixture_log_weights()  # (K,)
        component_log_probs = []
        for k, bf in enumerate(mbf.components):
            per_sample_ld = mbf._per_sample_log_det(bf, x_train)  # (N,)
            component_log_probs.append(log_pi[k] + per_sample_ld)

        stacked = torch.stack(component_log_probs, dim=0)  # (K, N)
        log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
        responsibilities = torch.exp(log_resp)  # (K, N)

        for k, bf in enumerate(mbf.components):
            resp_k = responsibilities[k]  # (N,)
            
            # Hard-assign samples with responsibility above threshold
            threshold = 1.0 / mbf.n_components
            mask = resp_k > threshold
            if mask.sum() < max(10, mbf.dim + 1):  # Need at least dim+1 for covariance
                # Fallback: top 30% by responsibility
                topk = max(int(0.3 * len(resp_k)), mbf.dim + 1)
                _, idx = torch.topk(resp_k, topk)
                mask = torch.zeros_like(resp_k, dtype=torch.bool)
                mask[idx] = True
            
            x_k = x_train[mask]  # (n_k, dim)
            
            # Get latent representations
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list)  # (n_k, dim), values in (0, 1)
            z_k = z_k.detach().cpu()
            
            # Fit MVN: mean and covariance
            mu_k = z_k.mean(dim=0)  # (dim,)
            # Unbiased covariance
            z_centered = z_k - mu_k.unsqueeze(0)
            n_k = z_k.shape[0]
            cov_k = (z_centered.T @ z_centered) / (n_k - 1)  # (dim, dim)
            
            # Add small diagonal regularization for numerical stability
            cov_k = cov_k + torch.eye(mbf.dim) * 1e-4
            
            mvn_params.append((mu_k, cov_k))
            print(f"Component {k}: {n_k} samples, mean={mu_k.numpy().round(3)}, "
                  f"std_diag={cov_k.diag().sqrt().numpy().round(3)}")

    return mvn_params
```

### 步骤 2：MVN 截断采样

```python
def sample_truncated_mvn(mu, cov, n_samples, low=0.01, high=0.99, max_tries=10):
    """
    Sample from Multivariate Normal, reject samples outside [low, high]^d.
    Uses rejection sampling with a fallback to rejection with clipping.
    
    :param mu: mean (dim,)
    :param cov: covariance (dim, dim)
    :param n_samples: number of samples
    :param low: lower bound (scalar)
    :param high: upper bound (scalar)
    :return: samples (n_samples, dim)
    """
    L = torch.linalg.cholesky(cov)  # Cholesky for sampling
    
    collected = []
    total_collected = 0
    
    for _ in range(max_tries):
        needed = n_samples - total_collected
        if needed <= 0:
            break
        # Sample more than needed to account for rejection
        oversample = needed * 4
        z = torch.randn(oversample, len(mu)) @ L.T + mu.unsqueeze(0)
        # Reject out-of-bound samples
        valid = ((z >= low) & (z <= high)).all(dim=1)
        z_valid = z[valid]
        if len(z_valid) > 0:
            take = min(len(z_valid), needed)
            collected.append(z_valid[:take])
            total_collected += take
    
    if total_collected < n_samples:
        # Fallback: clamp remaining samples
        needed = n_samples - total_collected
        z = torch.randn(needed, len(mu)) @ L.T + mu.unsqueeze(0)
        z = z.clamp(min=low, max=high)
        collected.append(z)
    
    return torch.cat(collected, dim=0)[:n_samples]
```

### 步骤 3：替换 inverse_map 的采样策略

```python
def inverse_map_with_mvn(mbf, n_samples, mvn_params, max_gap=1e-3, decay_ratio=1.0):
    """
    Generate samples using per-component MVN latent distribution.
    
    :param mbf: MultiBF model
    :param n_samples: number of samples to generate
    :param mvn_params: list of (mu_k, cov_k) from fit_component_mvn()
    :return: generated samples (n_samples, dim)
    """
    mbf.eval()
    weights = mbf.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, mbf.dim)

    for k in range(mbf.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        mu_k, cov_k = mvn_params[k]
        # Sample z from component k's MVN (in [0.01, 0.99]^d)
        z_k = sample_truncated_mvn(mu_k, cov_k, n_k, low=0.01, high=0.99)
        
        # Inverse map z -> x via bisection
        x_k = mbf.components[k].inverse_map(z_k, max_gap=max_gap, decay_ratio=decay_ratio)
        results[mask] = x_k

    return results
```

### 步骤 4：适用于单 BreezeForest（无 MultiBF）的版本

```python
def inverse_map_single_bf_with_mvn(bf, x_train_normalized, n_samples,
                                    n_clusters=None, max_gap=1e-3):
    """
    For a single BreezeForest, pre-cluster training data, fit per-cluster MVN,
    and use MVN-guided sampling to avoid inter-cluster generation.
    
    :param bf: single BreezeForest model
    :param x_train_normalized: normalized training data (N, dim)
    :param n_samples: number of samples
    :param n_clusters: number of clusters (if None, estimated)
    :return: generated samples
    """
    from sklearn.cluster import KMeans
    
    x_np = x_train_normalized.detach().cpu().numpy()
    n_k = n_clusters or estimate_n_clusters(x_np)  # e.g., using silhouette score
    
    kmeans = KMeans(n_clusters=n_k, n_init=10)
    labels = kmeans.fit_predict(x_np)
    
    # Get latent representations of training data
    with torch.no_grad():
        breeze_list = []
        z_all = bf.forward(x_train_normalized, breeze_list).detach()
    
    # Fit MVN per cluster
    mvn_params = []
    cluster_sizes = []
    for k in range(n_k):
        mask = torch.tensor(labels == k)
        z_k = z_all[mask]
        mu_k = z_k.mean(dim=0)
        cov_k = torch.cov(z_k.T) + torch.eye(z_k.shape[1]) * 1e-4
        mvn_params.append((mu_k, cov_k))
        cluster_sizes.append(mask.sum().item())
    
    cluster_weights = torch.tensor([s / sum(cluster_sizes) for s in cluster_sizes])
    component_indices = torch.multinomial(cluster_weights, n_samples, replacement=True)
    
    results = torch.zeros(n_samples, bf.dim)
    for k in range(n_k):
        mask = (component_indices == k)
        n_k_samples = mask.sum().item()
        if n_k_samples == 0:
            continue
        mu_k, cov_k = mvn_params[k]
        z_k_samples = sample_truncated_mvn(mu_k, cov_k, n_k_samples)
        x_k = bf.inverse_map(z_k_samples, max_gap=max_gap)
        results[mask] = x_k
    
    return results
```

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **协方差矩阵奇异** | 样本数少于维度时，协方差矩阵不可逆 | 添加对角正则化（1e-4 * I），或使用对角协方差（独立 MVN） |
| **MVN 尾部采样到边界外** | MVN 尾部可能超出 [0.01, 0.99]，需要截断 | 使用 rejection sampling 或 clamp，如实现建议中所示 |
| **责任分配不准（soft-EM 训练的 MultiBF）** | 如果 MultiBF 组件不专一（soft-EM 训练），latent Z_k 混杂多个 cluster，拟合的 MVN 偏差大 | 结合 K-Means Pre-Assign（本轮 Idea A）先改善训练，再用本 Idea 做 inference |
| **维度高时 MVN 估计困难** | 高维 MVN 需要大量样本估计协方差 | 使用对角协方差（各维度独立高斯），退化为升级版 LZR |
| **生成分布截断后不再严格是 MVN** | 截断 MVN 在边界处有截断效应 | 对 BreezeForest 2D 场景影响极小；高维可考虑 normalizing flow 作为 latent prior |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（升级版 LZR，直接可用于已训练模型）**

理由：
1. **无需重训练**：只需在已训练 MultiBF 上做一次 forward pass 收集 latent 表示，拟合 MVN，约 5-10 分钟即可完成
2. **直接针对问题**：MVN 采样的 z 值几乎只落在 cluster k 的 latent 区域内，从根本上减少 inter-cluster 生成
3. **对 LZR（Idea 2）的明确升级**：在相同实现成本下，MVN 比矩形边界更准确地描述 latent 区域形状
4. **适用于单 BF 和 MultiBF**：无需架构改动，通用性强
5. **理论支撑**：HGAD（2024）、VampPrior 等均验证了为不同 cluster 使用专属 Gaussian base 分布的有效性
6. **可作为 MALA Latent Sampling（Idea C）的初始分布**，两者可组合使用

---

## 参考文献

- Liu, Z. et al. (2024). "Hierarchical Gaussian Mixture Normalizing Flow Modeling for Unified Anomaly Detection." *arXiv:2403.13349*. https://arxiv.org/abs/2403.13349  
  (为不同类别使用专属 Gaussian latent 先验，互信息最大化结构化 latent space)
- Tomczak, J.M. & Welling, M. (2018). "VAE with a VampPrior." *AISTATS 2018*.  
  (混合 Gaussian prior 在生成模型中的有效性)
- Bevins, H. & Handley, W.J. (2023). "Piecewise Normalizing Flows." *arXiv:2305.02930*.  
  (pre-clustering 后各 component 的 base distribution 近似为 Normal，与本 Idea 互补)
- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.  
  (更复杂的 learned base distribution 方案；本 Idea 是其数据驱动的简化版本)
- Pires, G. & Figueiredo, M. (2020). "Variational Mixture of Normalizing Flows." *ESANN 2020*.  
  (Latent space partitioning in mixture flows)
