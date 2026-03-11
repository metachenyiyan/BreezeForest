# Idea: Covariance-Aware Latent Zone Sampling (CALZS)

**创建时间**: 2026-03-11 19:33 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（升级旧 Idea 2 LZR，更准确的 latent 采样）

---

## 问题定义

MultiBF 在生成时从 Uniform([0.01, 0.99]^d) 均匀采样 z，再通过 f_k^{-1}(z) 得到样本 x。由于 f_k 是从整个数据空间到 (0,1)^d 的全局双射，均匀采样 z 会映射到**所有** cluster 甚至 cluster 间区域。

旧 Idea 2（LZR, 1235）通过轴对齐矩形 zone [a_k^d, b_k^d] 来限制采样范围。这是正确方向，但存在**关键缺陷**：

1. **忽略维度间协方差**：如果 cluster k 在 latent 空间是斜着的椭圆，轴对齐矩形要么过大（包含其他 cluster 区域）要么过小（截断合法样本）
2. **均匀采样在矩形内**：即使 zone 估计准确，cluster 在 zone 内的分布也不是均匀的，而是有峰值的；均匀采样低密度边缘区域的比例过高
3. **百分位数超参数敏感**：percentile_low/percentile_high 选 5% 还是 10% 效果差异明显
4. **依赖 Z_k 是矩形**：现实中 latent cluster 是椭球形，不是超矩形

---

## 从项目代码与已有 idea 得到的背景判断

**代码分析**：
- `BreezeForest.forward()` 将 x 映射到 (0,1)^d 的 CDF 值（sigmoid 输出）
- 对专一化的组件 k，cluster k 的数据在 (0,1)^d 中的分布应该是某个有限支撑的分布（不是全 (0,1)^d）
- `BreezeForest.inverse_map()` 中的 bisection：先在 CDF 分布空间搜索，再在实数空间精化
- bisection 使用的 `distribution` 参数决定了搜索的粗范围；当前用 Normal(mean, std)，对多 cluster 数据是个粗近似

**LZR 的结构性限制**（轴对齐矩形）：
```
Z_k = [a_k^1, b_k^1] × [a_k^2, b_k^2] × ... × [a_k^d, b_k^d]
```
这个矩形的角落区域（(a_k^1, b_k^1) × (a_k^2, a_k^2) 的边角部分）很可能是空的或低密度的，但 LZR 会均匀采样这些角落。

**已有 idea 状态**：
- Idea 2（LZR, 1235）：percentile 矩形，均匀采样，正确方向但精度不足
- 本 Idea（CALZS）：多元高斯，尊重协方差，更准确地描述 latent cluster 形状

---

## 核心思路

**两步改进**：

### 改进 1：用多元高斯替代轴对齐矩形
对每个专一化组件 k：
1. 正向映射：z_i^k = f_k(x_i) for 属于 cluster k 的训练数据 x_i，得到 z_i^k ∈ (0,1)^d
2. 拟合多元高斯：从 {z_i^k} 估计 μ_k ∈ R^d 和 Σ_k ∈ R^{d×d}
3. 采样：z ~ N(μ_k, Σ_k)，然后 clamp 到 (0.01, 0.99)^d（因为 BF 的 sigmoid 只输出这个范围）
4. 逆映射：x = f_k^{-1}(z)

### 改进 2：按概率密度加权采样（可选增强）
如果 N(μ_k, Σ_k) 的某些样本超出 (0.01, 0.99)^d 过多，可以：
- 直接 clamp（简单，可能引入边界效应）
- 用 truncated multivariate Gaussian（严格数学正确，但更复杂）
- 拒绝超出范围的采样（等价于 rejection sampling）

**推荐实现**：clamp + 统计检查（若 >5% 的样本被 clamp 超过 0.05，增大 Σ_k 的缩放因子直到满足）

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**直觉**：cluster k 的数据在 latent 空间中形成一个**椭球形**分布（因为 BF 是一个连续、保形状的变换，不会把椭球变成任意形状）。从 N(μ_k, Σ_k) 采样 z：
- 采样集中在椭球中心（高概率区域）
- 快速衰减到椭球边缘
- 几乎不会采样到其他 cluster 的 latent 区域（除非两个 cluster 的 latent 表示严重重叠，那是专一化没做好的问题）

**与矩形 zone 的精确对比**：

| 特性 | LZR（矩形） | CALZS（多元高斯） |
|------|------------|----------------|
| 覆盖形状 | 超矩形（axis-aligned） | 椭球（捕获协方差） |
| 采样分布 | Uniform 在矩形内 | Gaussian 从中心向外衰减 |
| 无效角落 | 均匀采样到矩形角落 | 几何上正确，角落概率低 |
| 参数敏感性 | 对 percentile_low/high 敏感 | 仅有自由度：是否 scale Σ_k |
| 与 cluster 形状匹配 | 差（除非 cluster 轴对齐） | 好（对任意旋转椭球适用） |

**外部文献支撑**：
- 2024 年研究（arxiv openreview on GMM base distributions）：GMM base distribution 允许从特定 target 分布模式采样，同时维持可比的 in-distribution likelihood，优于标准 Gaussian base。本 Idea 是其在 MultiBF CDF-valued latent space 上的应用。
- Stimper et al. (2022)：resampled base distribution（LZR 的文献基础）；CALZS 是其在 per-component 场景的精确化版本。

---

## 与历史 idea 的关系

| 历史 Idea | 关系 | 说明 |
|-----------|------|------|
| **Idea 2（LZR, 1235）** | **明确升级（非替代）** | LZR 的矩形是 CALZS 多元高斯的特例（Σ_k 对角矩阵 + 硬截断）。CALZS 是 LZR 的严格推广，精度更高，参数更少（无 percentile 超参数），可替代 LZR |
| Idea 1 升级版（K-Means EM, 本轮） | **依赖关系（需要先运行）** | CALZS 的质量强烈依赖组件专一化程度：非专一化组件的 z^k 包含多个 cluster 的 latent codes，拟合 Gaussian 会失败。需先运行 K-Means Epoch EM 确保组件专一化 |
| Idea 3（ICDR, 1240） | 互补 | ICDR 从训练侧强化组件分离；CALZS 从采样侧精化生成区域；两者叠加最强 |

**旧 Idea 2（LZR）的价值**：LZR 的文档很详细，实现路径清晰。CALZS 保留其核心思路（post-training calibration），只将采样分布从 Uniform(矩形) 升级为 Gaussian(μ, Σ)。已有 LZR 实现代码的人可以直接迁移。

---

## 具体实现建议

### 核心方法：calibrate_latent_gaussians()

```python
def calibrate_latent_gaussians(mbf, x_train, responsibility_threshold=None):
    """
    Fit per-component multivariate Gaussian in latent space.
    
    Requires components to be approximately specialized (run K-Means Epoch EM first).
    
    :param mbf: MultiBF instance (trained with component specialization)
    :param x_train: full normalized training data (N, dim)
    :param responsibility_threshold: minimum responsibility for sample inclusion
                                     (default: 1/K, uniform threshold)
    :return: saves latent_gaussians = [(μ_k, Σ_k), ...] as attribute
    """
    if responsibility_threshold is None:
        responsibility_threshold = 1.0 / mbf.n_components
    
    mbf.latent_gaussians = []
    
    with torch.no_grad():
        # Compute soft responsibilities
        log_pi = mbf.get_mixture_log_weights()
        log_probs = []
        for k, bf in enumerate(mbf.components):
            ld = mbf._per_sample_log_det(bf, x_train)  # (N,)
            log_probs.append(log_pi[k] + ld)
        
        stacked = torch.stack(log_probs, dim=0)   # (K, N)
        log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
        resp = torch.exp(log_resp)   # (K, N)
        
        for k, bf in enumerate(mbf.components):
            resp_k = resp[k]  # (N,)
            
            # Use high-responsibility samples for Gaussian fitting
            mask = resp_k > responsibility_threshold
            
            # Fallback: use top-30% if threshold gives too few samples
            if mask.sum() < max(20, 0.1 * len(resp_k)):
                topk = max(20, int(0.3 * len(resp_k)))
                _, idx = torch.topk(resp_k, topk)
                mask = torch.zeros_like(resp_k, dtype=torch.bool)
                mask[idx] = True
            
            x_k = x_train[mask]   # (n_k, dim)
            
            # Forward-map through component k to get latent codes
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list)   # (n_k, dim), values in (0, 1)
            
            # Fit multivariate Gaussian: μ and Σ
            mu_k = z_k.mean(dim=0)   # (dim,)
            
            n_k = z_k.shape[0]
            z_centered = z_k - mu_k.unsqueeze(0)   # (n_k, dim)
            
            if n_k >= 2:
                # Sample covariance with small regularization for stability
                sigma_k = (z_centered.T @ z_centered) / (n_k - 1)
                # Regularize: add small diagonal to avoid singular matrix
                sigma_k = sigma_k + 1e-4 * torch.eye(mbf.dim)
            else:
                # Fallback to identity (shouldn't happen with good specialization)
                sigma_k = 0.01 * torch.eye(mbf.dim)
            
            mbf.latent_gaussians.append((mu_k, sigma_k))
            
            # Verify: check what fraction of samples would be clamped
            with torch.no_grad():
                dist = torch.distributions.MultivariateNormal(mu_k, sigma_k)
                test_samples = dist.sample((500,))
                clamped = test_samples.clamp(0.01, 0.99)
                clamp_fraction = (test_samples - clamped).abs().sum() / (500 * mbf.dim)
            
            print(f"Component {k}: n_k={n_k}, μ={mu_k.numpy().round(3)}, "
                  f"diag(Σ)={sigma_k.diag().numpy().round(4)}, "
                  f"clamp fraction={clamp_fraction.item():.3f}")
    
    return mbf.latent_gaussians
```

### 生成函数：inverse_map_with_latent_gaussian()

```python
def inverse_map_with_latent_gaussian(mbf, n_samples, max_gap=1e-3, decay_ratio=1.0):
    """
    Generate samples using per-component Gaussian latent space sampling.
    
    Requires calibrate_latent_gaussians() to be called first.
    
    :param mbf: MultiBF instance with latent_gaussians attribute
    :param n_samples: number of samples to generate
    :return: generated samples (n_samples, dim)
    """
    assert hasattr(mbf, 'latent_gaussians'), "Call calibrate_latent_gaussians() first"
    
    weights = mbf.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, mbf.dim)
    
    for k in range(mbf.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        mu_k, sigma_k = mbf.latent_gaussians[k]
        dist_k = torch.distributions.MultivariateNormal(mu_k, sigma_k)
        
        # Oversample to account for clamping rejection
        oversample_factor = 2
        z_raw = dist_k.sample((n_k * oversample_factor,))
        
        # Clamp to valid BF output range
        z_valid = z_raw.clamp(0.01, 0.99)
        
        # Take first n_k samples (or all if not enough after valid check)
        z = z_valid[:n_k]
        
        x_k = mbf.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
        results[mask] = x_k
    
    return results
```

### 完整工作流（结合 K-Means Epoch EM）

```python
# 1. 训练：使用 K-Means Epoch EM（idea_kmeans_epoch_em）
# ... training loop ...

# 2. 校准：拟合 per-component latent Gaussian
all_batch = (all_batch_raw - mean) / std
with torch.no_grad():
    calibrate_latent_gaussians(mbf, all_batch, responsibility_threshold=1.0/n_components)

# 3. 生成：从 latent Gaussian 采样
mbf.eval()
with torch.no_grad():
    samples = inverse_map_with_latent_gaussian(mbf, n_samples=3000)
    samples = samples * std + mean
```

### 超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `responsibility_threshold` | 1/K | 选高 responsibility 样本用于 Gaussian 拟合 |
| 正则化 | 1e-4 * I | 防止奇异协方差矩阵 |
| oversample_factor | 2 | 用于抵消 clamp 拒绝的样本损失 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **协方差矩阵奇异** | 样本数少于维度数时 Σ_k 不满秩 | 添加正则化 1e-4 * I；使用对角协方差（更稳定） |
| **Clamp 边界效应** | Gaussian 尾部超出 [0.01, 0.99]，被强制 clamp 到边界 → 密度失真 | 使用 truncated Gaussian；或检查 clamp fraction < 5% |
| **依赖专一化** | 若组件未专一化，z_k 包含多个 cluster → Gaussian 拟合为大方差混合 | 先运行 K-Means Epoch EM；使用更严格的 responsibility 阈值 |
| **高维下 Σ 不稳定** | d 大时全协方差矩阵不可靠 | 使用对角协方差（忽略维度间相关），或 PCA 降维后再拟合 |
| **Clamp 引入的 z 分布偏差** | 被 clamp 到 0.01 或 0.99 的 z 值在 inverse_map 中可能产生极端样本 | 用 rejection sampling 替代 clamp：拒绝超出范围的样本并重采样 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（与 K-Means Epoch EM 配合使用）**

理由：
1. **精度提升**：多元高斯捕获协方差，比 LZR 矩形更准确描述 latent cluster 形状
2. **参数更少**：只需正则化强度（固定值），不需要 percentile_low/high 两个超参数
3. **实现简单**：在 LZR 基础上只需改变采样方式（用 MultivariateNormal 替代 Uniform），约 30 行代码差异
4. **理论更严谨**：对各向异性 cluster（椭球形、斜向）的建模正确
5. **最优组合**：K-Means Epoch EM（专一化）+ CALZS（精确 latent 采样）= 从根本上同时修复训练和推断

---

## 参考文献

- Bevins, H.T.J. & Handley, W.J. (2023). "Piecewise Normalizing Flows." *arXiv:2305.02930*.  
  *(K-Means 聚类后每个流独立建模 — 本 Idea 的 latent 高斯是其 CDF 空间的对应)*
- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.  
  https://proceedings.mlr.press/v151/stimper22a.html  
  *(LZR 的原始依据，CALZS 是其 per-component 精确化版本)*
- OpenReview (2024). GMM base distributions for continuous normalizing flows.  
  *(从 cluster-specific GMM 基础分布采样优于标准 Gaussian 基础分布)*
- Kviman, O. et al. (2023). "Cooperation in the Latent Space." *ICML 2023*.  
  https://proceedings.mlr.press/v202/kviman23a.html  
  *(Latent 空间 mixture component 的相互作用分析)*
- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 2024*.  
  *(Latent 空间采样策略对多模态问题的重要性)*
