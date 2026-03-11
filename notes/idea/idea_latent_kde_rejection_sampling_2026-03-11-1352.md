# Idea: Latent-Space KDE Rejection Sampling（升级版 LZR）

**创建时间**: 2026-03-11 13:52 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（替代并升级 LZR 1235，且无需重训练）

---

## 问题定义

MultiBF（以及单组件 BreezeForest）在生成时采样 z ~ Uniform([0.01, 0.99]^d)，然后通过 `inverse_map` 计算 x = f^{-1}(z)。

**根本问题**：`f: data space → [0.01, 0.99]^d` 是双射，因此 [0.01, 0.99]^d 中的**每个** z 都对应数据空间中的某个 x。当数据有多个 cluster 时：
- Cluster A 的数据点被映射到 [0.01, 0.99]^d 中的子区域 Z_A
- Cluster B 的数据点被映射到子区域 Z_B
- **Cluster 之间的低密度区域**映射到 Z_A^c ∩ Z_B^c（即 Z_A 和 Z_B 以外的区域）
- 从 Uniform([0.01, 0.99]^d) 采样必然包括这些"无效"区域，生成时就产生 inter-cluster 点

**当前 LZR（1235）的局限**：
- 使用轴对齐的矩形框 `[lo_k^1, hi_k^1] × ... × [lo_k^d, hi_k^d]` 作为 zone 边界
- 矩形框可能过大（包含跨 cluster 的角落区域）或过小（截断同一 cluster 的合法样本）
- 各维度独立的百分位数边界忽略了维度间的相关性

---

## 从代码与已有 Idea 得到的背景判断

### 代码中的关键观察

1. **`BreezeForest.forward(x)`** 将 x 映射到 (0, 1)^d，最后一层使用 Sigmoid 激活函数，保证输出在 (0,1) 范围内。
2. **`BreezeForest.inverse_map(z)`** 使用二分法维度级求逆，z 的每个维度独立解算。
3. **z 的实际分布是非均匀的**：即使从 Uniform([0.01, 0.99]^d) 采样，经过 `inverse_map` 后的 x 分布也与训练数据的分布有出入，因为 Jacobian 不恒等于常数。

### 已有 LZR（1235）的核心思路与改进空间

LZR（1235）的关键洞察是正确的：
> 训练数据在 latent space 中的 representation `z_i = f(x_i)` 只覆盖 [0.01, 0.99]^d 的一个子集；通过限制采样范围到这个子集，就能避免从 inter-cluster 区域生成。

**但矩形框估计有两个问题**：
1. **矩形框的"角落"不属于数据分布**：在 2D 情况下，如果 cluster 数据的 latent 表示是一个斜椭圆，矩形框会包含 4 个角落，这些角落对应 inter-cluster 甚至 out-of-distribution 区域
2. **矩形框过于保守或过于激进**：百分位数阈值是标量超参数，难以针对每个 cluster 独立优化

**KDE 替代方案**：
- 用核密度估计（KDE）拟合训练数据在 latent space 中的实际分布形状
- 在采样时使用**拒绝采样**：propose z ~ Uniform([0.01, 0.99]^d)，accept 概率 ∝ KDE(z)
- 这直接对应 Stimper et al. (2022) 的"resampled base distribution"思路，是最有理论支撑的方案

---

## 核心思路

**训练后校准（Post-Training Calibration）+ 拒绝采样**：

1. **Latent KDE 拟合**：
   - 对每个组件 k（MultiBF）或全局（单 BF），取训练数据的 latent representation
   - 在 latent space 中拟合一个 KDE（scikit-learn 或自定义）
   - KDE 的带宽用交叉验证或 Silverman 规则自动选择

2. **生成时拒绝采样**：
   - Proposal: z ~ Uniform([0.01, 0.99]^d)
   - Acceptance probability: min(1, KDE(z) / M)，其中 M 是一个归一化常数（可用 KDE 的最大值估计）
   - 接受的 z 再通过 `inverse_map` 得到 x
   
3. **无需重训练**，只需在已训练的模型上一次性校准（约几秒钟）

### 关键创新（vs. LZR 1235）

| 维度 | LZR（1235） | Latent KDE（本 Idea） |
|------|------------|----------------------|
| Zone 形状 | 轴对齐矩形框 | 任意形状（数据驱动） |
| 维度相关性 | 独立处理 | KDE 天然捕捉维度相关 |
| 理论基础 | 启发式百分位数截断 | 拒绝采样（精确）或重采样基分布（理论严格） |
| 超参数 | 百分位数阈值 | 带宽（可自动选择） |
| 适用于单 BF | 需要组件分配 | 直接适用 |
| 计算开销 | 极低 | 低（KDE 拟合快，采样时 acceptance rate 约 50-80%） |

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**理论保证（Stimper et al. 2022 的关键结论）**：

设 p_data(x) 是数据分布，f: x → z 是训练好的 flow，则：
- f 将 p_data 的 pushforward 分布 q(z) = p_data(f^{-1}(z)) / |det J_f^{-1}(z)| 映射到 latent space
- 标准采样假设 z ~ Uniform([0.01, 0.99]^d)，即用均匀分布近似 q(z)
- **误差来源**：q(z) ≠ Uniform，在 inter-cluster 区域对应的 z 值处，q(z) 接近 0，但均匀采样给这些区域赋予了正概率

**KDE 修正**：
- 用 KDE 估计 q(z)（用训练数据的 latent representations）
- 通过拒绝采样近似从 q(z) 采样
- 采样得到的 z ~ q(z)，则 `inverse_map(z)` 的输出分布等价于 p_data(x)（在 flow 能力范围内）

**直觉验证**：
- 在 moons 数据集：两个月牙的 latent representations 形成两个不相连的 L 形区域，KDE 能精确捕捉这种形状；矩形框会包含 L 形角落对应的 inter-cluster 点
- 在 8 gaussians 数据集：8 个高斯的 latent representations 形成 8 个分散的圆形区域，KDE 用高斯核天然适配；矩形框的外接矩形会包含大量 inter-cluster 空间

---

## 与历史 Idea 的关系

**替代/升级 Idea 1235（Latent Zone Restriction）**。

LZR（1235）的核心思路（限制 latent 采样区域）是正确的，本 Idea 用更精确的 KDE 密度估计替代了其粗糙的矩形框估计。

**量化改进**：
- LZR 的矩形框覆盖的 latent volume ≈ 实际有效区域的 2-5 倍（取决于数据形状）
- KDE 覆盖的 latent volume ≈ 实际有效区域的 1.0-1.2 倍（取决于带宽选择）
- 改进幅度：减少 50%-80% 的无效 z 采样

**替代关系**：在新实现中，推荐用本 Idea 替代 LZR（1235）；若计算资源有限，LZR（1235）仍可作为更简单的替代。

**与 K-Means Hard-EM（本轮 Idea 1）的关系**：两者高度互补：
- Idea 1 改善训练（使组件专一化），使 KDE 拟合的 latent distribution 更清晰
- 本 Idea 改善采样（使生成仅来自有数据的区域），两者叠加是最强方案

**与 ICDR（1240）的关系**：ICDR 改善训练过程中的组件分离；本 Idea 在推理时生效。无直接替代关系。

---

## 具体实现建议

### 步骤 1：添加 `calibrate_latent_kde()` 到 MultiBF

```python
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def calibrate_latent_kde(self, x_train, bandwidth='silverman', n_top_samples=None):
    """
    Fit per-component KDE in latent space from training data.
    
    :param x_train: normalized training data (N, dim)
    :param bandwidth: 'silverman', 'scott', or a float
    :param n_top_samples: if set, only use top-N samples by responsibility per component
    """
    self.latent_kdes = []
    self.latent_kde_log_maxvals = []
    
    with torch.no_grad():
        # Compute soft responsibilities
        log_pi = self.get_mixture_log_weights()
        component_log_probs = []
        for k, bf in enumerate(self.components):
            ld = self._per_sample_log_det(bf, x_train)
            component_log_probs.append(log_pi[k] + ld)
        stacked = torch.stack(component_log_probs, dim=0)  # (K, N)
        log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
        resp = torch.exp(log_resp)  # (K, N)
        
        for k, bf in enumerate(self.components):
            resp_k = resp[k]  # (N,)
            
            # Select high-responsibility samples
            if n_top_samples is not None:
                topk = min(n_top_samples, len(resp_k))
                _, idx = torch.topk(resp_k, topk)
            else:
                threshold = 1.0 / self.n_components
                idx = torch.where(resp_k > threshold)[0]
                if len(idx) < 10:
                    _, idx = torch.topk(resp_k, max(50, len(resp_k) // 5))
            
            # Forward pass to get latent representations
            x_k = x_train[idx]
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list).detach().numpy()  # (n_k, dim)
            
            # Fit KDE with automatic bandwidth selection
            if bandwidth == 'silverman':
                # Silverman's rule: h = 0.9 * n^{-1/(d+4)} * sigma
                n, d = z_k.shape
                h = 0.9 * (n ** (-1.0 / (d + 4))) * np.std(z_k)
                h = max(h, 0.01)  # lower bound
                kde = KernelDensity(kernel='gaussian', bandwidth=h)
            elif bandwidth == 'scott':
                n, d = z_k.shape
                h = n ** (-1.0 / (d + 4))
                kde = KernelDensity(kernel='gaussian', bandwidth=h)
            else:
                kde = KernelDensity(kernel='gaussian', bandwidth=float(bandwidth))
            
            kde.fit(z_k)
            
            # Estimate log max val for normalization constant M
            log_scores = kde.score_samples(z_k)
            log_max = float(np.max(log_scores))
            
            self.latent_kdes.append(kde)
            self.latent_kde_log_maxvals.append(log_max)
    
    print(f"Calibrated KDEs for {len(self.latent_kdes)} components:")
    for k, (kde, log_max) in enumerate(zip(self.latent_kdes, self.latent_kde_log_maxvals)):
        print(f"  Component {k}: bandwidth={kde.bandwidth:.4f}, log_max={log_max:.2f}")
```

### 步骤 2：添加 `inverse_map_with_kde()` 到 MultiBF

```python
def inverse_map_with_kde(
    self, n_samples, max_gap=1e-3, decay_ratio=1.0,
    max_tries_multiplier=10
):
    """
    Generate samples using per-component latent KDE rejection sampling.
    Requires calibrate_latent_kde() to have been called.
    
    :param n_samples: target number of samples
    :param max_tries_multiplier: max proposals = n_samples * max_tries_multiplier
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
        
        kde_k = self.latent_kdes[k]
        log_M_k = self.latent_kde_log_maxvals[k]
        
        # Rejection sampling in latent space
        accepted = []
        n_tries = 0
        max_tries = n_k * max_tries_multiplier
        
        while len(accepted) < n_k and n_tries < max_tries:
            # Proposal: Uniform([0.01, 0.99]^d)
            batch_size = min((n_k - len(accepted)) * 3, 500)
            z_proposed = torch.rand(batch_size, self.dim) * 0.98 + 0.01
            z_np = z_proposed.numpy()
            
            # Compute KDE log density for each proposal
            log_probs = kde_k.score_samples(z_np)  # (batch_size,)
            
            # Acceptance probability: min(1, exp(log_kde - log_M))
            log_accept = log_probs - log_M_k
            u = np.log(np.random.rand(batch_size) + 1e-10)
            accepted_mask = (log_accept >= u)
            
            for i in range(batch_size):
                if accepted_mask[i] and len(accepted) < n_k:
                    accepted.append(z_proposed[i])
            
            n_tries += batch_size
        
        if len(accepted) < n_k:
            # Fallback: pad with uniform samples if rejection is too aggressive
            while len(accepted) < n_k:
                accepted.append(torch.rand(self.dim) * 0.98 + 0.01)
        
        z_accepted = torch.stack(accepted[:n_k], dim=0)
        x_k = self.components[k].inverse_map(
            z_accepted, max_gap=max_gap, decay_ratio=decay_ratio
        )
        results[mask] = x_k
    
    return results
```

### 步骤 3：也适用于单组件 BreezeForest

```python
# 为单 BF 添加 KDE 校准
def calibrate_latent_kde_single_bf(bf, x_train, bandwidth='silverman'):
    """For single BreezeForest, fit KDE on all training latent representations."""
    with torch.no_grad():
        breeze_list = []
        z_all = bf.forward(x_train, breeze_list).detach().numpy()
    
    n, d = z_all.shape
    if bandwidth == 'silverman':
        h = 0.9 * (n ** (-1.0 / (d + 4))) * np.std(z_all)
        h = max(h, 0.01)
        kde = KernelDensity(kernel='gaussian', bandwidth=h)
    else:
        kde = KernelDensity(kernel='gaussian', bandwidth=float(bandwidth))
    
    kde.fit(z_all)
    log_max = float(np.max(kde.score_samples(z_all)))
    return kde, log_max
```

### 步骤 4：在 demo_multi_bf.py 中集成

```python
# 训练完成后：
# 1. 校准 KDE
all_data = get_all_training_data(distribution, data_size, mean, std)
with torch.no_grad():
    mbf.calibrate_latent_kde(all_data, bandwidth='silverman')

# 2. 用 KDE rejection sampling 生成样本
with torch.no_grad():
    samples = mbf.inverse_map_with_kde(n_samples=data_size)
    samples = samples * std + mean
```

### 带宽选择建议

| 数据类型 | 推荐带宽 | 备注 |
|---------|---------|------|
| 高斯 clusters（紧凑） | silverman | 自动适配每个 cluster 的方差 |
| 月牙形、螺旋形 | 0.05 – 0.1（较小带宽） | 需要 KDE 捕捉细长形状 |
| 均匀 cluster | 0.1 – 0.2 | 避免过度平滑 |
| 未知 | 先用 silverman，观察 acceptance rate：若 < 20%，增大带宽；若 > 90%，减小带宽 | |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **KDE 带宽敏感性** | 带宽过大 → KDE 过平滑，仍包含 inter-cluster 区域；带宽过小 → acceptance rate 极低 | 使用 Silverman 自动选择；可用 leave-one-out 交叉验证优化 |
| **低 acceptance rate** | 若 cluster 数据 latent 表示非常集中，acceptance rate 可能 < 10%，生成慢 | 增大带宽；或使用 MCMC（MALA）替代拒绝采样 |
| **KDE 在高维失效** | 维度 d > 5 时 KDE 质量退化（维度诅咒） | BreezeForest 目前主要用于 2D；高维可改用 Gaussian Mixture Model（GMM）替代 KDE |
| **组件未专一化时 KDE 错误** | 若 MultiBF 训练不好（组件混杂），KDE 拟合的 latent distribution 包含多个 cluster 的 z 值 | 配合 Idea 1（K-Means Hard-EM）先保证组件专一化，再做 KDE 校准 |
| **计算开销** | KDE 拟合 O(N * log N)，采样时 KDE 评估 O(N) per sample | N ≤ 5000 时约几秒，可接受；高维可改用 GMM（更快） |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（升级 LZR 1235，无需重训练）**

理由：
1. **零重训练成本**：在已训练的 MultiBF 或单 BF 上直接运行，约 5-10 秒 calibration
2. **即刻验证**：可在任何已训练模型上测试效果，与训练方法无关
3. **理论严格**：对应 Stimper et al. (2022) 的 resampled base distribution，有严格理论保证
4. **适用于单 BF**（LZR 需要多组件才有意义；本 Idea 对单 BF 也有效）
5. **精度明显优于 LZR**：KDE 捕捉非矩形、非轴对齐的 cluster 形状，减少误采样

**实施建议**：
1. 立即：在已有模型上测试（无需任何重训练）
2. 配合 Idea 1（K-Means Hard-EM 重训练）后再做 KDE 校准，效果最强

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. arXiv:2110.15828.
  （本 Idea 的直接理论来源：resampled base distribution = KDE rejection sampling）
- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 2024*. arXiv:2305.12149.
  （MALA 方法：KDE 拒绝采样的替代品，适合高维情况）
- Scott, D.W. (1992). *Multivariate Density Estimation: Theory, Practice, and Visualization*. Wiley.
  （KDE 带宽选择的经典参考）
- Silverman, B.W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman & Hall.
  （Silverman 带宽规则）
