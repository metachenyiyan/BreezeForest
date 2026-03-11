# Idea: Cluster-Conditioned Empirical Latent Sampling (CELS)

**创建时间**: 2026-03-11 20:09 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（Inference-time 修复，升级 LZR）

---

## 问题定义

BreezeForest（包括单独使用和作为 MultiBF 的组件）在生成阶段的问题，在于从 `Uniform([0.01, 0.99]^d)` 均匀采样 z，再通过 `inverse_map` 映射回数据空间。

**已有的 LZR idea（2026-03-11 12:35）的核心正确**，但存在一个根本性的近似问题：

LZR 通过百分位数估计每个组件的 latent zone `Z_k = [lo_k^d, hi_k^d]` per dimension，然后从 `Uniform(Z_k)` 采样。这是一个**轴对齐的矩形**近似。

对于 BreezeForest 的自回归结构，latent 空间中的 cluster 表示（cluster k 的训练数据经 f_k forward pass 后的 z 值分布）**不一定是轴对齐的矩形**：

- BreezeForest 的 breeze weights 在 z 空间中引入维度间的相关性
- 两个 cluster 在原始数据空间的分离，在 z 空间中可能表现为对角方向的分离
- LZR 的矩形 Z_k 可能包含实际上属于其他 cluster（或 inter-cluster 区域）的 z 值

**核心改进目标**：用 z-space 中的真实经验分布替代矩形 bounding box。

---

## 从当前项目代码与已有 idea 中得到的背景判断

**代码观察：**

1. `BreezeForest.forward()` 是确定性的：给定 x，总产生唯一 z = f(x)。这意味着训练数据 {x_i} 在 z-space 有确定的对应集合 {z_i = f(x_i)}。
2. `demo_functions.py::generate_sample()` 从 `Uniform(0.01, 0.99)` 采样 seeds（z），然后 `inverse_map(seeds)` → 这是问题的根源
3. `MultiBF.inverse_map()` 对每个组件 k 独立采样 `z = torch.rand(n_k, self.dim) * 0.98 + 0.01` → 与 LZR 思路一致，但仍用均匀采样
4. 单 BF 的 `generate_sample()` 也是均匀采样 → **CELS 对单 BF 也同样适用**，这是 LZR 未强调的
5. `BreezeForest.forward()` 的输出范围理论上是 [0, 1]^d（由 Sigmoid 激活保证），但实际训练数据的 z 值分布在 [0.01, 0.99] 的某个**子区域**

**现有 idea 分析：**

- LZR (1235)：矩形 zone，需要 percentile 超参数，对非轴对齐分布效果有限
- PIPT (new)：PIPT 训练后各组件的 z 分布更集中，CELS 效果更好（两者协同）
- Hard-EM (1230)：CELS 同样可以配合，但 CELS 对 PIPT 的依赖更小（即使单 BF 也可用）

**外部背景：**

Stimper et al. (2022) "Resampling Base Distributions of Normalizing Flows" 的核心思想是：学习一个 base distribution q(z)，使得从 q(z) 采样再做 inverse_map 的分布更接近目标。CELS 是这个思想的**非参数、数据驱动的实现**：用训练数据在 z-space 的经验分布直接替代均匀分布，无需额外学习 q。

---

## 核心思路

**训练后校准（Post-Training Calibration）+ 改进采样：**

**Step 1：收集各组件的 z 经验分布**

对每个组件 k，forward-pass 其分配的训练数据（或对单 BF，forward-pass 所有训练数据），收集 z_i^k = f_k(x_i)。

**Step 2：拟合 per-component 的 z 分布**

对收集到的 {z_i^k}，用**轻量的参数估计**：
- 选项 A（最简）：存储所有 z_i^k（N_k × dim 矩阵），生成时随机采样 + 加小扰动 → **Data-Resampling**
- 选项 B（推荐）：对每个维度 d 拟合一维 KDE（使用 bandwidth = Silverman 规则），生成时从 KDE 采样 → **KDE-based sampling**
- 选项 C（更精准）：拟合**多元 Gaussian（Cholesky factored）** 到 {z_i^k}，捕捉维度间相关性 → **Gaussian in z-space**

**Step 3：生成时从 per-component z 分布采样**

```python
z_k ~ Gaussian(mean_z_k, Sigma_z_k)   # instead of Uniform([0.01, 0.99]^d)
z_k = z_k.clamp(0.01, 0.99)           # ensure valid range
x_k = f_k^{-1}(z_k)                   # inverse_map
```

**核心洞察**：

`{z_i^k}` 是训练数据 cluster k 在 latent space 中的精确映像。从这个分布采样，等价于在 cluster k 的 latent 表示空间中"内插"，而不是从覆盖了所有 cluster（甚至 inter-cluster 区域）的均匀分布中采样。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**数学推理：**

设 f_k：X → [0,1]^d 是组件 k 的正向映射（bijection）。

对 cluster k 的训练数据 D_k = {x_i : i ∈ cluster_k}：
- 其 z 表示为 Z_k_train = {z_i = f_k(x_i) : x_i ∈ D_k}
- 由双射性：f_k^{-1}(Z_k_train) = D_k（精确！）

对 inter-cluster 区域的点 x_inter（位于 cluster A 和 cluster B 之间的空旷区域）：
- 其 z 表示为 z_inter = f_k(x_inter)
- 由于 f_k 是 monotone bijection，z_inter **必然**不在 Z_k_train 的覆盖范围内（或处于 Z_k_train 的稀疏区域）

因此，从 Z_k_train 采样（而不是从全 [0.01, 0.99]^d 均匀采样），**结构性地**排除了 inter-cluster 点对应的 z 值。

**与 LZR 的关键差异（为什么 CELS 更好）：**

| 方面 | LZR | CELS |
|------|-----|------|
| z-space cluster 形状 | 轴对齐矩形（bounding box） | 实际数据分布（任意形状） |
| 超参数 | percentile_low, percentile_high | Gaussian: 无；KDE: bandwidth（自动 Silverman） |
| 实现对维度相关性 | 每维度独立估计，忽略相关 | Gaussian 捕捉完整协方差结构 |
| 对非轴对齐分布 | 可能包含大量 non-cluster z 值 | 精确跟踪 cluster 的实际形状 |
| 与 PIPT 的协同 | 好 | 更好（PIPT 使各组件 z 分布更紧凑） |

**对单 BF 的适用性（LZR 未强调，CELS 扩展）：**

对于单 `BreezeForest`（非 MultiBF），同样存在 inter-cluster 误生成问题：

- 对所有训练数据做 forward pass，得到 {z_i = f(x_i)}
- 拟合 GMM（K 个 Gaussian，对应 K 个 cluster）到 {z_i}
- 生成时从 GMM 采样 z，然后做 `inverse_map`

这使单 BF 也能在一定程度上避免 inter-cluster 生成，而无需切换到 MultiBF 架构。

---

## 与历史 idea 的关系

| 历史 Idea | 关系 | 说明 |
|-----------|------|------|
| LZR (1235) | **升级（非替代，CELS 扩展了 LZR）** | CELS 保留了 LZR 的核心思想（训练后校准 z-space 区域），但用更精确的非参数/参数估计替代了矩形 bounding box。LZR 的 percentile 方法可以看作 CELS 的一个特殊简化情形 |
| Hard-EM (1230) / PIPT (new) | **正交，协同** | PIPT 训练使各组件的 z 分布更紧凑、更单峰；CELS 利用这个性质，可以用更精确（甚至单 Gaussian）来描述每个组件的 z 分布 |
| ICDR (1240) / EMRS (new) | **正交，互补** | EMRS 是 training-time 方案，CELS 是 inference-time 方案，两者叠加效果最佳 |

**与 Stimper et al. 2022 的关系：**

Stimper et al. 提出**学习**一个 resampled base distribution q(z)（通过最大化 ELBO 或 KL 散度）。CELS 是该方法的**非参数近似**：用训练数据的经验 z 分布直接作为 q，不需要额外的学习过程。这在 BreezeForest 的场景下更简单，且理论上更直接。

---

## 具体实现建议

### 选项 B（推荐）：基于 Gaussian 的 per-component z 分布

```python
class CELSCalibrator:
    """
    Post-training calibrator for Cluster-Conditioned Empirical Latent Sampling.
    Fits a Gaussian to per-component z distributions for use in inverse_map.
    """
    
    def __init__(self, mbf: MultiBF, x_train: torch.Tensor, min_variance: float = 1e-4):
        """
        :param mbf: trained MultiBF model
        :param x_train: normalized training data (N, dim)
        :param min_variance: minimum variance to prevent degenerate Gaussians
        """
        self.n_components = mbf.n_components
        self.dim = mbf.dim
        self.component_z_params = []  # list of (mean_z, L_cholesky) per component
        
        with torch.no_grad():
            # Compute soft responsibilities
            log_pi = mbf.get_mixture_log_weights()
            component_log_probs = []
            for k, bf in enumerate(mbf.components):
                ld = mbf._per_sample_log_det(bf, x_train)
                component_log_probs.append(log_pi[k] + ld)
            
            stacked = torch.stack(component_log_probs, dim=0)  # (K, N)
            log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
            assignments = torch.argmax(log_resp, dim=0)  # hard assignment (N,)
            
            for k, bf in enumerate(mbf.components):
                # Get samples strongly associated with component k
                mask = (assignments == k)
                if mask.sum() < 5:
                    # Fallback: use all data with uniform Gaussian
                    print(f"Warning: Component {k} has few samples; using global stats")
                    z_k = bf.forward(x_train, [])
                else:
                    x_k = x_train[mask]
                    z_k = bf.forward(x_k, [])  # (n_k, dim) z values for cluster k
                
                # Fit Gaussian: mean and covariance of z_k
                z_mean = z_k.mean(dim=0)  # (dim,)
                z_centered = z_k - z_mean  # (n_k, dim)
                
                # Compute covariance (regularized)
                if z_k.shape[0] > 1:
                    z_cov = (z_centered.T @ z_centered) / (z_k.shape[0] - 1)  # (dim, dim)
                    # Regularize: add min_variance to diagonal
                    z_cov = z_cov + torch.eye(self.dim) * min_variance
                else:
                    z_cov = torch.eye(self.dim) * min_variance
                
                # Cholesky decomposition for sampling
                try:
                    L = torch.linalg.cholesky(z_cov)
                except RuntimeError:
                    # Fallback if Cholesky fails (non-PD matrix)
                    z_cov = torch.eye(self.dim) * (z_k.var(dim=0).mean().item() + min_variance)
                    L = torch.linalg.cholesky(z_cov)
                
                self.component_z_params.append((z_mean, L))
                print(f"Component {k}: z_mean={z_mean.numpy().round(3)}, "
                      f"z_std_diag={torch.diagonal(L).numpy().round(3)}")
    
    def sample_z(self, k: int, n_samples: int, 
                  temperature: float = 1.0) -> torch.Tensor:
        """
        Sample z values from component k's fitted Gaussian distribution.
        
        :param k: component index
        :param n_samples: number of samples
        :param temperature: scale factor for variance (1.0 = fitted, <1.0 = tighter)
        :return: z samples (n_samples, dim), clamped to [0.01, 0.99]
        """
        mean_z, L = self.component_z_params[k]
        eps = torch.randn(n_samples, self.dim)  # (n_samples, dim)
        z = mean_z + temperature * (eps @ L.T)  # (n_samples, dim)
        return z.clamp(0.01, 0.99)
    
    def inverse_map_cels(self, mbf: MultiBF, n_samples: int, 
                          temperature: float = 1.0,
                          max_gap: float = 1e-3, 
                          decay_ratio: float = 1.0) -> torch.Tensor:
        """
        Generate samples using CELS: sample z from per-component Gaussian,
        then apply inverse_map.
        
        :param mbf: trained MultiBF
        :param n_samples: number of samples
        :param temperature: controls sampling tightness around cluster center
        :return: generated samples (n_samples, dim)
        """
        weights = mbf.get_mixture_weights().detach()
        component_indices = torch.multinomial(weights, n_samples, replacement=True)
        results = torch.zeros(n_samples, mbf.dim)
        
        for k in range(self.n_components):
            mask = (component_indices == k)
            n_k = mask.sum().item()
            if n_k == 0:
                continue
            
            # Sample from fitted z Gaussian (instead of Uniform)
            z_k = self.sample_z(k, n_k, temperature=temperature)
            
            # Inverse map: z -> x
            x_k = mbf.components[k].inverse_map(
                z_k, max_gap=max_gap, decay_ratio=decay_ratio
            )
            results[mask] = x_k
        
        return results
```

### 在 `demo_multi_bf.py` 中的使用

```python
# 训练完成后：
# 1. 校准 CELS calibrator
all_loader = DataLoader(distribution, batch_size=data_size, shuffle=True)
all_batch, _ = next(iter(all_loader))
all_batch_norm = (all_batch - mean) / std

cels = CELSCalibrator(mbf, all_batch_norm)

# 2. 使用 CELS 生成
mbf.eval()
with torch.no_grad():
    samples = cels.inverse_map_cels(mbf, n_samples=data_size, temperature=0.95)
    samples = samples * std + mean
    samples = samples.numpy()

# 可视化
pyplot.plot(samples[:, 0], samples[:, 1], ".", markersize=0.5)
pyplot.title("MultiBF + CELS Generated")
pyplot.show()
```

### 对单 BreezeForest 的 CELS（扩展 LZR 未覆盖的单 BF 场景）

```python
def single_bf_cels(bf: BreezeForest, x_train: torch.Tensor, 
                    n_components: int = None, n_samples: int = 3000,
                    temperature: float = 0.95) -> torch.Tensor:
    """
    Apply CELS to a single BreezeForest by clustering z-space.
    Uses GMM in z-space if n_components is given, otherwise uses full empirical distribution.
    """
    from sklearn.mixture import GaussianMixture
    
    with torch.no_grad():
        z_train = bf.forward(x_train, []).numpy()  # (N, dim)
    
    if n_components is not None and n_components > 1:
        # Fit GMM in z-space
        gmm = GaussianMixture(n_components=n_components, n_init=5)
        gmm.fit(z_train)
        z_samples, _ = gmm.sample(n_samples)  # (n_samples, dim)
        z_samples = torch.tensor(z_samples, dtype=torch.float)
    else:
        # Direct resampling from empirical z distribution
        idx = torch.randint(0, len(x_train), (n_samples,))
        z_noise = torch.randn(n_samples, x_train.shape[1]) * 0.02
        z_samples = torch.tensor(z_train[idx]) + z_noise
    
    z_samples = z_samples.clamp(0.01, 0.99)
    
    with torch.no_grad():
        x_gen = bf.inverse_map(z_samples)
    return x_gen
```

### Temperature 参数指南

| temperature | 效果 |
|-------------|------|
| 1.0 | 与 z 分布完全匹配，最紧凑但可能过于保守 |
| 0.9 – 0.95 | 轻微压缩，推荐默认值 |
| 0.7 – 0.8 | 较强压缩，样本集中在 cluster 核心，多样性略降 |
| 1.1 – 1.2 | 扩展，探索 cluster 边缘，可能产生少量 inter-cluster 样本 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **Gaussian 近似误差** | 如果 z-space 中的 cluster 表示是非 Gaussian 的（多峰、弯曲），单 Gaussian 近似会引入采样偏差 | 升级到 GMM（每个组件 fit 一个 GMM 而不是单 Gaussian）；或使用 KDE（Parzen window） |
| **z 分布估计需要训练数据** | Calibration 步骤需要 forward-pass 所有训练数据 | 使用 mini-batch 分批 forward-pass；或在训练中维护 running z 统计（mean/cov） |
| **Cholesky 分解不稳定** | 协方差矩阵接近奇异（某个维度 z 值接近常数）时 Cholesky 失败 | 添加正则化项 min_variance；检测后 fallback 到对角 Gaussian |
| **对 soft-EM 训练效果有限** | 如果模型用 soft-EM 训练（没有 PIPT/Hard-EM），各组件的 z 分布可能重叠，calibration 质量下降 | 与 PIPT 或 EMRS 配合使用；提高 responsibility 阈值 |
| **generation diversity 下降** | 从紧凑 Gaussian 采样可能限制生成多样性，使样本过于集中 | 调高 temperature 参数（1.0–1.1）增加多样性 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（与 PIPT 并列，且无需重训练，可立即验证）**

理由：
1. **即时可验证**：在任何已训练的 MultiBF（或单 BF）上，只需加 calibration 步骤即可，无需重训练
2. **比 LZR 更准确**：捕捉 z-space 中 cluster 的真实协方差结构（包括维度间相关性），而 LZR 只用矩形 bounding box
3. **对单 BF 也有效**：LZR 的 idea 主要针对 MultiBF，而 CELS 的 z-GMM 变体可以让单 BF 在生成时也避开 inter-cluster z 区域
4. **Temperature 参数**：提供了对 diversity 和 quality 的精细控制（LZR 没有这个机制）
5. **有理论支撑**：与 Stimper et al. 2022 的 resampled base distributions 思路完全一致，是其非参数化实现
6. **与 PIPT + EMRS 的组合效果**：PIPT 使 z 分布更集中，EMRS 使组件分工更明确，CELS 利用这两个效果做精确采样 → 三者形成完整的 pipeline

**推荐使用顺序：**
1. **CELS alone** → 立即验证现有模型是否可改善（无需重训练）
2. **PIPT + CELS** → 重训练为最佳效果，inference 阶段加 CELS
3. **PIPT + EMRS fine-tuning + CELS** → 完整 pipeline，最高生成质量

---

## 参考文献

- Stimper, V., Schölkopf, B., & Hernandez-Lobato, J.M. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.  
  https://proceedings.mlr.press/v151/stimper22a.html  
  (Primary theoretical backing: learned q(z) to improve flow sampling from multi-modal targets)
- Bevins, H.T.J. & Handley, W.J. (2023). "Piecewise Normalizing Flows." *arXiv:2305.02930*.  
  https://arxiv.org/abs/2305.02930  
  (Shows that data-partitioning approaches outperform resampled base distributions; supports CELS + PIPT combination)
- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 2024*.  
  https://arxiv.org/abs/2305.12149  
  (Latent-space manipulation of normalizing flow sampling for improved multi-modal coverage)
- Scott, D.W. (1992). "Multivariate Density Estimation: Theory, Practice, and Visualization."  
  (Silverman bandwidth rule for KDE; basis for automatic KDE calibration in CELS)
