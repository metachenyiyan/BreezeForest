# Idea: Per-Component Gaussian Latent Sampling (GLS) — LZR 的原理升级版

**创建时间**: 2026-03-11 16:54 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（可立即在已训练模型上验证，无需重训练）

---

## 问题定义

MultiBF 的 **inference-time 生成问题**：

当前 `MultiBF.inverse_map()` 的生成策略：
```python
z = torch.rand(n_k, self.dim) * 0.98 + 0.01  # Uniform([0.01, 0.99]^d)
x_k = self.components[k].inverse_map(z, ...)
```

**根本缺陷**：每个组件 f_k 是一个从 R^d 到 (0,1)^d 的全局双射。Uniform([0.01, 0.99]^d) 均匀覆盖整个 (0,1)^d，而 f_k^{-1} 的**整个 (0,1)^d 都有合法映射目标**。

对于多 cluster 数据：
- f_k 在训练后（即使是 soft-EM）将不同 cluster 的数据映射到 (0,1)^d 的不同子区域
- cluster k 的数据在 (0,1)^d 中占据某子区域 Z_k ⊂ (0,1)^d
- cluster j≠k 的数据和 cluster 之间的区域占据 Z_k 的补集
- **Uniform([0.01, 0.99]^d) 采样时必然包含 Z_k 的补集部分 → 生成 inter-cluster 和其他 cluster 的样本**

**LZR（Idea 2）的问题**：LZR 使用矩形包围框 [a_k^d, b_k^d] 近似 Z_k。但：
1. **矩形假设不准确**：真实 Z_k 可能是非轴对齐的、斜的，甚至是椭圆形的
2. **对角区域误包含**：矩形会包含角落区域（没有数据但在矩形范围内），这些区域映射到 inter-cluster 位置
3. **维度独立假设**：矩形忽略了不同维度 CDF 值之间的相关性（实际上有相关，因为 breeze 连接）

**本 Idea（GLS）的改进**：用一个**多元高斯分布 N(μ_k, Σ_k)** 来替代矩形或均匀分布作为组件 k 的生成基分布。高斯分布自然地：
- 集中于 Z_k 的中心，随距离中心增大而概率指数下降
- 可以捕捉维度间的相关性（通过协方差矩阵 Σ_k）
- 通过温度参数 τ 控制采样集中程度

---

## 从当前项目代码与已有 idea 中得到的背景判断

### 代码层面分析

**BreezeForest.inverse_map() 的核心逻辑**：
```python
# tools.py: bisection function
def bisection(target, inc_func, distribution=None, ...):
    # Stage 1: coarse search using reference distribution (Normal)
    lo, hi = _bisect(0, 1, lambda m: inc_func(distribution.icdf(m)), gap_dis)
    lo = distribution.icdf(lo.clamp(min=1-anomaly_dis))
    hi = distribution.icdf(hi.clamp(max=anomaly_dis))
    # Stage 2: fine search in real space
    lo, hi = _bisect(lo, hi, inc_func, gap_real)
    return (lo + hi) / 2
```

bisection 的 `target` 就是采样的 z 值。z 决定了 x 的最终位置。

**关键洞察**：改变采样策略（z 的采样分布）不需要修改 bisection 算法本身，只需要修改传入 `inverse_map` 的 z 张量。

**BreezeForest.forward() 可以直接用于校准**：
- `bf.forward(x_train, breeze_list)` 给出每个训练样本的 z = f_k(x)
- 这些 z 值的统计量直接描述了 Z_k 的形状

### 与 LZR（Idea 2）的对比

| 方面 | LZR（Idea 2） | GLS（本 Idea） |
|------|--------------|--------------|
| Z_k 估计方式 | 按维度独立计算百分位数，得到矩形 | 计算多元高斯参数 (μ_k, Σ_k)，捕捉协方差 |
| 采样方式 | Uniform([a_k, b_k]^d)，矩形内均匀 | Normal(μ_k, τ²Σ_k)，以中心为核心指数衰减 |
| 处理非轴对齐 cluster | 不支持（矩形） | 支持（协方差矩阵自然表达旋转/倾斜） |
| 对 inter-cluster 防护 | 切断矩形以外的区域 | 指数衰减，越远离 Z_k 中心概率越低 |
| 温度控制 | 无直接温度参数（通过百分位数调整） | 通过 τ ∈ (0,1] 直接控制 |
| 优势 | 简单，无分布假设 | 更准确，更平滑，支持协方差 |
| 局限 | 矩形误包含角落区域 | 高斯假设可能不完全准确（可用 GMM 升级） |

**结论**：GLS 是 LZR 的**原理级升级**，在大多数情况下更准确，且实现复杂度相当。

---

## 核心思路

**训练后校准（Post-Training Calibration）**：

1. 对训练数据中分配给组件 k 的样本，计算其 latent 表示：z_i^k = f_k(x_i)（直接用 forward）
2. 计算这些 z_i^k 的**均值 μ_k 和协方差矩阵 Σ_k**（或简化为对角协方差 σ_k²）
3. 将 N(μ_k, τ²Σ_k) 作为组件 k 的生成基分布（温度 τ 控制采样集中程度）

**生成时约束**：
- 不再从 Uniform([0.01, 0.99]^d) 采样
- 改为从 N(μ_k, τ²Σ_k) 采样，截断（clamp）到 [0.01, 0.99]^d
- 然后用 f_k^{-1}（bisection）做 inverse_map

**为什么高斯是合理假设**：
- f_k 的输出是 CDF 值（Sigmoid 激活），每个维度输出在 (0,1) 之间
- 对于一个 unimodal cluster，f_k 将其 CDF 值集中在某个区域
- 由中心极限定理，多层 sigmoid 变换后的 CDF 值分布近似正态
- 即使不严格正态，高斯采样也比均匀采样更集中于实际数据的 latent 表示区域

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**严格论证**：

设 Z_k = {f_k(x_i) : x_i ∈ cluster_k} 是 cluster k 的数据在 latent 空间的表示集合。

- f_k 是双射，所以 f_k^{-1} 在 Z_k 上的像 = cluster k 的数据区域（近似）
- 高斯 N(μ_k, τ²Σ_k) 以 Z_k 的中心 μ_k 为核心，概率密度随距离指数衰减
- 当 τ 较小时，绝大多数采样 z 落在 Z_k 内 → f_k^{-1}(z) 落在 cluster k 附近
- 与 Uniform 采样相比：Uniform([0.01, 0.99]^d) 中来自 Z_k^c 的 z 会产生 inter-cluster 样本，而 Gaussian 采样中 Z_k^c 的 z 概率极低

**对比 Stimper et al. (2022) Resampled Base Distribution**：
- Stimper 的方法通过学习一个 rejection sampling 过程来构造 base distribution
- GLS 更简单：直接从数据中估计 Z_k 的高斯参数，无需训练额外模型
- GLS 是 Resampled Base Distribution 在 mixture flow 上的轻量级数据驱动实现

---

## 与历史 idea 的关系

**直接升级 LZR（历史 Idea 2）**，替代关系明确：

- LZR（Idea 2）：矩形包围框，维度独立
- GLS（本 Idea）：多元高斯，支持协方差，更准确
- GLS 包含了 LZR 的全部功能，并在几乎所有情况下更优
- **建议：用 GLS 替代 LZR，不再单独实施 LZR**

**与 Hard-EM（Idea 1）和 K-Means Warm-Start（本轮 Idea 1）的关系**：**互补**
- Hard-EM 训练后，各组件的 Z_k 更纯净（只包含一个 cluster 的 latent 表示）
- GLS 在 Hard-EM 训练后效果更好（Z_k 接近单 Gaussian，μ_k 和 Σ_k 估计更准确）
- **即使不用 Hard-EM，GLS 也能单独改善生成质量**

**对 Stimper (2022) Resampled Base Distribution 的关系**：同一思路的轻量级实现版本

---

## 具体实现建议

### 步骤 1：添加 calibrate_gaussian_latent() 方法到 MultiBF

```python
def calibrate_gaussian_latent(
    self, 
    x_train, 
    use_diagonal_cov=True,
    responsibility_threshold=None,
    temperature=0.7
):
    """
    Calibrate per-component Gaussian latent distributions.
    After training, fit N(mu_k, tau^2 * Sigma_k) to each component's latent representations.
    
    :param x_train: training data tensor (N, dim)
    :param use_diagonal_cov: if True, use diagonal covariance (faster, less accurate)
    :param responsibility_threshold: only use samples with responsibility > threshold
    :param temperature: tau parameter, controls sampling concentration (default 0.7)
    """
    self.gaussian_latent_params = []
    self.gaussian_latent_temperature = temperature
    
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
            resp_k = responsibilities[k]  # (N,)
            
            if responsibility_threshold is not None:
                mask = resp_k > responsibility_threshold
            else:
                # Use samples with responsibility above uniform threshold
                threshold = 1.0 / self.n_components
                mask = resp_k > threshold
            
            if mask.sum() < 5:
                # Fallback: top 20% by responsibility
                topk = max(int(0.2 * len(resp_k)), 5)
                _, idx = torch.topk(resp_k, topk)
                mask = torch.zeros_like(resp_k, dtype=torch.bool)
                mask[idx] = True
            
            # Forward pass: get latent representations z_k = f_k(x)
            x_k = x_train[mask]
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list)  # (n_k, dim), values in (0, 1)
            
            # Fit Gaussian
            mu_k = z_k.mean(dim=0)  # (dim,)
            
            if use_diagonal_cov:
                # Diagonal covariance: just variances
                var_k = z_k.var(dim=0).clamp(min=1e-6)  # (dim,)
                self.gaussian_latent_params.append({
                    'mu': mu_k,
                    'var': var_k,
                    'diagonal': True
                })
            else:
                # Full covariance matrix
                z_centered = z_k - mu_k.unsqueeze(0)  # (n_k, dim)
                cov_k = (z_centered.T @ z_centered) / (len(z_k) - 1)  # (dim, dim)
                # Add small regularization for numerical stability
                cov_k = cov_k + 1e-5 * torch.eye(self.dim)
                self.gaussian_latent_params.append({
                    'mu': mu_k,
                    'cov': cov_k,
                    'diagonal': False
                })
        
        print(f"Calibrated Gaussian latent params (temp={temperature}):")
        for k, params in enumerate(self.gaussian_latent_params):
            if params['diagonal']:
                print(f"  Component {k}: mu={params['mu'].numpy().round(3)}, "
                      f"std={params['var'].sqrt().numpy().round(3)}")
            else:
                print(f"  Component {k}: mu={params['mu'].numpy().round(3)}")
```

### 步骤 2：修改 inverse_map 使用 Gaussian latent sampling

```python
def inverse_map_gaussian(self, n_samples, max_gap=1e-3, decay_ratio=1.0, clamp_range=(0.01, 0.99)):
    """
    Generate samples using per-component Gaussian latent base distribution.
    Requires calibrate_gaussian_latent() to be called first.
    
    :param n_samples: number of samples to generate
    :param max_gap: bisection precision
    :param decay_ratio: bisection decay ratio  
    :param clamp_range: clamp z values to this range (to stay in valid bisection range)
    :return: generated samples (n_samples, dim)
    """
    assert hasattr(self, 'gaussian_latent_params'), "Call calibrate_gaussian_latent() first"
    
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)
    tau = self.gaussian_latent_temperature

    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        params = self.gaussian_latent_params[k]
        mu_k = params['mu']  # (dim,)
        
        if params['diagonal']:
            var_k = params['var']  # (dim,)
            # Sample from N(mu_k, tau^2 * diag(var_k))
            std_k = (tau * var_k.sqrt()).clamp(min=1e-4)
            z = mu_k.unsqueeze(0) + torch.randn(n_k, self.dim) * std_k.unsqueeze(0)
        else:
            cov_k = params['cov']  # (dim, dim)
            # Sample from N(mu_k, tau^2 * cov_k)
            try:
                L = torch.linalg.cholesky(tau**2 * cov_k)
                z = mu_k.unsqueeze(0) + torch.randn(n_k, self.dim) @ L.T
            except Exception:
                # Fallback to diagonal if Cholesky fails
                std_k = (tau * torch.diagonal(cov_k).clamp(min=1e-6).sqrt())
                z = mu_k.unsqueeze(0) + torch.randn(n_k, self.dim) * std_k.unsqueeze(0)
        
        # Clamp to valid range for bisection
        z = z.clamp(min=clamp_range[0], max=clamp_range[1])
        
        x_k = self.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
        results[mask] = x_k

    return results
```

### 步骤 3：在 demo_multi_bf.py 中添加校准步骤

```python
# 训练完成后：
# 1. 校准 Gaussian latent params
all_data_loader = DataLoader(distribution, batch_size=data_size, shuffle=True)
all_batch, _ = next(iter(all_data_loader))
all_batch = (all_batch - mean) / std

with torch.no_grad():
    mbf.calibrate_gaussian_latent(
        all_batch,
        use_diagonal_cov=True,   # 先用对角协方差，快速
        temperature=0.7           # tau=0.7 相当于将采样半径缩小到 cluster 70%
    )

# 2. 使用 Gaussian latent 生成
with torch.no_grad():
    samples = mbf.inverse_map_gaussian(n_samples=data_size)
    samples = samples * std + mean
```

### 温度参数 τ 调优建议

| τ 值 | 效果 | 适用场景 |
|------|------|---------|
| 1.0 | 与高斯 latent 完全匹配，接近原始数据分布（但排除 Z_k 以外的区域） | 验证 GLS 基本效果 |
| 0.7 | 稍微集中于 cluster 中心，平衡覆盖度和精确度 | **推荐默认值** |
| 0.5 | 较集中，生成样本更紧凑（接近 cluster 核心） | 需要高精度生成时 |
| 0.3 | 非常集中，可能遗漏 cluster 边缘样本 | 极端精确度需求 |

从 τ=1.0 开始验证，再调低到 0.7 观察效果。

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **高斯假设不准确** | 若 cluster 的 latent 表示是非高斯的（如双月、螺旋形），单高斯估计不准 | 可升级为 GMM（sklearn.mixture.GaussianMixture）；通常 2-3 个 GM 成分足够 |
| **Soft-EM 导致 Z_k 不纯** | 若用 soft-EM 训练，Z_k 可能混入其他 cluster 的 latent 点 | 用更严格的 responsibility 阈值（如 > 0.8）选取"纯"样本 |
| **低 τ 截断边缘样本** | τ 太小会使生成样本不覆盖 cluster 的边缘区域，多样性降低 | 从 τ=1.0 开始，可视化调整 |
| **clamp 引入边界效应** | 将 z 截断到 [0.01, 0.99] 可能引入概率堆积在边界 | 通常 cluster 的 latent 表示远离 [0, 1] 边界，不是实际问题 |
| **数值稳定性** | 满协方差矩阵可能是病态的（近奇异）| 用对角协方差（use_diagonal_cov=True）或添加正则化 1e-5 * I |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（本轮最优推荐之一，且无需重训练）**

理由：
1. **即时可验证**：不需要重训练，只需在已有模型上运行校准步骤
2. **原理升级**：比 LZR 更准确，处理非轴对齐 cluster 和协方差结构
3. **有理论支撑**：与 Stimper et al. (2022) Resampled Base Distribution 同源，是其数据驱动轻量版
4. **温度控制灵活**：τ 参数提供细粒度的生成多样性与精确度权衡
5. **与其他 idea 叠加**：Hard-EM + K-Means Init 后效果最佳，但单独也有效

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a.html
  - 直接启发：通过学习非均匀 base distribution 来解决 topology mismatch 问题
- Bevins, H.T.J. et al. (2023). "Piecewise Normalizing Flows." *arxiv 2305.02930*.
  - 支持：pre-clustering 后每个 flow 的 base distribution 近似单峰 Gaussian
- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 2024*. https://arxiv.org/abs/2305.12149
  - 相关：在 latent 空间施加采样约束，与 GLS 思路一致
- MacKay, D.J.C. (2003). "Information Theory, Inference, and Learning Algorithms." Cambridge.
  - 理论背景：CDF 变换后的分布近似性质
