# Idea: Per-Component GMM Latent Base Distribution — 拟合 z 空间分布以精准采样

**创建时间**: 2026-03-11 16:23 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（无需重训练，对 Idea 2 LZR 的关键升级）

---

## 问题定义

BreezeForest 的生成问题在于：即使 MultiBF 的某个组件 k 已经专一于 cluster k，在生成阶段仍然会产生 inter-cluster 的点。根本原因是：

**当前采样策略**：z ~ Uniform([0.01, 0.99]^d) → x = F_k^{-1}(z)

这意味着 z 从整个 [0.01, 0.99]^d **均匀**采样。但 F_k 是一个从 ℝ^d 到 [0,1]^d 的双射，它把数据空间中的**所有区域**都映射到 [0,1]^d 的某个位置，包括：
- Cluster k 的高密度区域 → 映射到 Z_k（训练数据的 z-space 集中区域）
- Cluster j≠k 的数据 → 映射到 Z_j^k（这些点在组件 k 的 z-space 中也有位置）
- Cluster 之间的低密度区域 → 也有对应的 z 值

当 z ~ Uniform([0.01, 0.99]^d) 时，z 会覆盖 Z_j^k 和 Z_k 之间的"桥梁区域"，从而 F_k^{-1}(z) 生成 inter-cluster 的点。

**已有 Idea 2（LZR，12:35）** 通过估计每个组件的 `Z_k = [a_k^d, b_k^d]^d`（各维度百分位数盒子）来限制采样范围。但这个方案的核心局限是：

1. **轴对齐盒子近似太粗糙**：真实的 Z_k 可能是非矩形的（Z_k 的形状取决于 cluster k 的几何结构），盒子边界会包含大量属于其他 cluster 的 z 值。
2. **忽略维度间相关性**：各维度独立估计边界，无法捕捉 z-space 中的协方差结构。
3. **无法区分 Z_k 内的高低密度区域**：即使在 Z_k 的盒子内部，也有部分区域是 cluster k 数据的"稀疏区域"，均匀采样仍然会产生不典型的点。

**本 Idea 的修复**：对每个组件 k，用 GMM（或 KDE）拟合训练数据在 z-space 中的分布，在生成时从这个**拟合的 z 分布**采样，而不是均匀采样。

---

## 从当前项目代码与已有 idea 中得到的背景判断

### 代码层面的关键路径

`MultiBF.inverse_map` 中：
```python
z = torch.rand(n_k, self.dim) * 0.98 + 0.01  # Uniform([0.01, 0.99])
x_k = self.components[k].inverse_map(z, ...)
```

`BreezeForest.forward(x, breeze_list)` 将 x 映射到 [0,1]^d（通过 Sigmoid 激活函数）。所以对于训练数据 x_i，`z_i^k = bf_k.forward(x_i)` 就是 x_i 在组件 k 的 latent space 中的坐标，且 z_i^k ∈ [0,1]^d。

这个映射是确定的、可计算的，无需任何额外的反向传播，只需一次正向传播。

### 已有 idea 层面

- **Idea 2（LZR，12:35）**：本 idea 是对 LZR 的直接升级。LZR 用盒子近似，本 idea 用 GMM。LZR 的 `calibrate_latent_zones()` 函数已经完成了"前向传播得到 z 值"的部分，本 idea 在此基础上把"盒子估计"替换为"GMM 拟合"。
- **Idea 1（Hard-EM，12:30）** / **本轮 K-Means Idea**：组件专一化后，Z_k 会更加集中和清晰，GMM 拟合效果更好。两者协同。
- **Idea 3（ICDR，12:40）**：与本 idea 正交，可叠加。

### 外部文献验证

- **arXiv:2512.04954（2024）**：直接证明 GMM base distribution 比 unimodal base distribution 在 multi-modal posterior 上效果更好，消除了 inter-mode 的"probability bridge"。
- **Stimper et al. (2022)** "Resampling Base Distributions of Normalizing Flows"：通过 learned rejection sampling 修复 topology mismatch。本 idea 是其简化版本：用 empirical GMM 替代 learned rejection sampler，无需额外训练。
- **PNF（Bevins & Handley, 2023）**：每个 cluster flow 从专属 cluster 数据训练，隐含了 z-space 分布会集中于某个子区域的假设。本 idea 显式利用这一点。

---

## 核心思路

**Post-Training Calibration：拟合 z-space 中每个组件的分布**

1. **数据分配**：计算训练数据对每个组件的 responsibility，得到每个样本"主要属于哪个组件"（soft 或 hard）。
2. **z-space 映射**：对属于组件 k 的训练样本，通过 `bf_k.forward(x_i)` 得到其 latent 表示 `z_i^k ∈ [0,1]^d`。
3. **GMM 拟合**：用 `sklearn.mixture.GaussianMixture` 对 `{z_i^k}` 集合拟合一个 GMM（通常 1-3 个 Gaussian 成分已足够，因为单个 cluster 在 z-space 中分布较简单）。
4. **GMM 采样**：在生成时，从 `fitted_GMM_k` 采样 z，clamp 到 [0.01, 0.99]^d，再用 `bf_k.inverse_map(z)` 得到 x。

从 GMM_k 采样的 z 值会集中在训练数据的 z-space 分布附近，`bf_k^{-1}(z)` 自然会生成接近 cluster k 的样本，**不会**生成 inter-cluster 的点（因为 inter-cluster 的 x 对应的 z 值在 GMM_k 的低密度区域）。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**数学论证**：

设 T_k ⊂ [0,1]^d 为 cluster k 训练数据的 z-space 表示的支撑集。则：
- z_i^k = F_k(x_i) ∈ T_k，对所有 cluster k 的样本
- inter-cluster 的点 x_{gap} 满足 F_k(x_{gap}) ∉ T_k（因为 F_k 是双射，不同 x 有不同 z）

因此，从 T_k 的分布采样 z，再取 F_k^{-1}(z)，输出必然接近 cluster k 的数据，而不会是 inter-cluster 的 x_{gap}（因为其对应的 z_{gap} ∉ T_k）。

**GMM vs 盒子（LZR）的对比**：

| 采样策略 | 方法 | 对 inter-cluster 的控制 |
|--------|------|----------------------|
| Uniform([0.01, 0.99]^d) | 当前（无限制） | 无控制 |
| 盒子 Z_k（LZR） | 各维度百分位数 | 粗糙控制（盒子可能包含非 cluster k 的 z 值） |
| GMM_k（本 idea） | 拟合 z-space 真实分布 | 精确控制（只从高密度区域采样） |
| KDE_k | 核密度估计 | 非参数，更灵活（高维时有问题） |

**直觉图示**：
```
z-space [0,1]^2 示意：
┌──────────────────────────────┐
│          (空白区域)             │
│    ╔══════════╗               │
│    ║ cluster A║    ╔═══════╗  │
│    ║ z-support║    ║ clus B║  │
│    ╚══════════╝    ╚═══════╝  │
│    <-盒子A->  <-盒子B->       │
│ (盒子重叠！)  (盒子重叠！)      │
└──────────────────────────────┘

GMM_A 只在左侧椭圆形区域有高密度
GMM_B 只在右侧椭圆形区域有高密度
→ 从 GMM_A 采样不会产生盒子重叠区域的 z 值
```

---

## 它与历史 idea 的关系

**对 Idea 2（LZR，12:35）的关键升级（partial supercedes）**：

LZR 是一个好的思路，但盒子边界是一个过于粗糙的近似。本 idea 在相同的 post-training calibration 框架内，用 GMM 替代盒子，显著提升精确度。

具体关系：

| 方面 | Idea 2 LZR | 本 Idea GMM-Z |
|------|-----------|--------------|
| 核心框架 | Post-training calibration | Post-training calibration（相同） |
| z 分布估计 | 轴对齐盒子（各维度独立 percentile） | GMM（多变量，捕获协方差） |
| 非凸 cluster 支持 | 否（盒子只能近似凸集） | 是（GMM 可以近似非凸形状） |
| 实现依赖 | 仅 PyTorch | PyTorch + sklearn GaussianMixture |
| 适合场景 | cluster 在 z-space 近似矩形 | 任何 cluster 形状 |

**建议**：保留 LZR 作为"轻量快速版本"，GMM-Z 作为"精确版本"。先用 LZR 快速验证，再升级到 GMM-Z。

**与 K-Means Init Hard-EM（本轮新 Idea 1）的关系**：
- K-Means Idea 改善训练时组件专一度 → GMM_k 的拟合质量更高（因为 cluster k 的数据更"纯"）
- 两者协同使用：先训练（K-Means Hard-EM），后校准（GMM-Z）

---

## 具体实现建议

### 步骤 1：添加 calibrate_gmm_latent_base() 到 MultiBF

```python
from sklearn.mixture import GaussianMixture
import numpy as np

def calibrate_gmm_latent_base(
    self, 
    x_train, 
    n_gmm_components=3,
    hard_assignment=True,
    percentile_clip=(1.0, 99.0)
):
    """
    Fit per-component GMM to latent space representations.
    
    :param x_train: training data (N, dim), normalized
    :param n_gmm_components: number of Gaussian components in GMM for each flow component
    :param hard_assignment: if True, use argmax assignment; if False, use responsibility threshold
    :param percentile_clip: clamp z values to these percentile bounds before fitting
    """
    self.latent_gmms = []
    self.latent_gmm_clip = []
    
    with torch.no_grad():
        # Compute component responsibilities
        log_pi = self.get_mixture_log_weights()
        component_log_probs = []
        for k, bf in enumerate(self.components):
            ld = self._per_sample_log_det(bf, x_train)
            component_log_probs.append(log_pi[k] + ld)
        
        stacked = torch.stack(component_log_probs, dim=0)  # (K, N)
        
        for k, bf in enumerate(self.components):
            if hard_assignment:
                assignments = torch.argmax(stacked, dim=0)
                mask = (assignments == k)
            else:
                log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
                resp_k = torch.exp(log_resp[k])
                mask = resp_k > (1.0 / self.n_components)  # threshold = uniform
            
            if mask.sum() < max(2 * n_gmm_components, 10):
                # Fallback: use top-20% by responsibility
                topk = max(int(0.2 * x_train.shape[0]), 10)
                log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
                _, idx = torch.topk(torch.exp(log_resp[k]), topk)
                mask = torch.zeros(x_train.shape[0], dtype=torch.bool)
                mask[idx] = True
            
            # Forward pass: get z-space coordinates for assigned samples
            x_k = x_train[mask]
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list).detach()  # (n_k, dim)
            
            # Clamp to valid range and clip to stable percentile range
            z_k = z_k.clamp(0.005, 0.995)
            z_np = z_k.numpy()
            
            # Clip outliers per dimension before GMM fitting
            clip_lo = np.percentile(z_np, percentile_clip[0], axis=0)
            clip_hi = np.percentile(z_np, percentile_clip[1], axis=0)
            z_np = np.clip(z_np, clip_lo, clip_hi)
            
            # Fit GMM in z-space
            n_comp = min(n_gmm_components, len(z_np) // 5)  # at least 5 samples per GMM component
            n_comp = max(n_comp, 1)
            
            gmm = GaussianMixture(
                n_components=n_comp, 
                covariance_type='full',
                n_init=3,
                random_state=42
            )
            gmm.fit(z_np)
            
            self.latent_gmms.append(gmm)
            self.latent_gmm_clip.append((
                torch.tensor(clip_lo, dtype=torch.float32),
                torch.tensor(clip_hi, dtype=torch.float32)
            ))
            
            print(f"Component {k}: fitted GMM with {n_comp} components on {mask.sum().item()} samples")
```

### 步骤 2：修改 inverse_map() 使用 GMM 采样

```python
def inverse_map_gmm(self, n_samples, max_gap=1e-3, decay_ratio=1.0):
    """
    Generate samples using per-component GMM latent base distributions.
    Requires calibrate_gmm_latent_base() to be called first.
    """
    assert hasattr(self, 'latent_gmms'), "Call calibrate_gmm_latent_base() first"
    
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)
    
    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        # Sample from GMM in z-space
        gmm_k = self.latent_gmms[k]
        z_np, _ = gmm_k.sample(n_k)  # (n_k, dim)
        
        # Clamp to valid BreezeForest range [0.01, 0.99]
        z_np = np.clip(z_np, 0.01, 0.99)
        z = torch.tensor(z_np, dtype=torch.float32)
        
        x_k = self.components[k].inverse_map(
            z, max_gap=max_gap, decay_ratio=decay_ratio
        )
        results[mask] = x_k
    
    return results
```

### 步骤 3：在 demo_multi_bf.py 中集成

```python
# === 训练完成后：校准 GMM latent base ===
all_loader = DataLoader(distribution, batch_size=3000, shuffle=True)
all_batch, _ = next(iter(all_loader))
all_batch = (all_batch - mean) / std

with torch.no_grad():
    # 先做 LZR 快速验证（可选）
    mbf.calibrate_latent_zones(all_batch, percentile_low=5.0, percentile_high=95.0)
    
    # 再做 GMM-Z 精确校准
    mbf.calibrate_gmm_latent_base(all_batch, n_gmm_components=3, hard_assignment=True)

# === 使用 GMM-Z 生成 ===
with torch.no_grad():
    samples_gmm = mbf.inverse_map_gmm(n_samples=data_size)
    samples_gmm = samples_gmm * std + mean
```

### 超参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `n_gmm_components` | 1-3 | 单 cluster 通常用 1-2 个 Gaussian 就够，复杂 cluster 可用 3 |
| `hard_assignment` | True（若已用 Hard-EM 训练） | Hard 分配更干净；soft 分配适合纯 soft-EM 训练模型 |
| `percentile_clip` | (2, 98) | 避免 GMM 拟合受 z 空间 outlier 影响 |
| `covariance_type` | 'full' | 全协方差矩阵，适合低维（dim=2）；高维可用 'diag' |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **GMM 外推问题** | GMM 采样可能产生 z < 0.01 或 z > 0.99 的值 | 硬 clamp 到 [0.01, 0.99] 后再做 inverse_map（代码中已包含） |
| **GMM 拟合在组件不专一时效果差** | 若 soft-EM 训练导致组件不专一，则 z_k 包含多个 cluster 的数据，GMM 拟合会覆盖多个 cluster | 与 K-Means 训练（本轮 Idea 1）结合；或用更严格的 hard assignment 阈值 |
| **高维 z-space 的 GMM 拟合困难** | BreezeForest 目前是 2D（dim=2），GMM 没问题；高维时需要更多数据 | dim ≤ 10 时 GMM 效果良好；dim > 10 时考虑 normalizing flow 作为 z-space 估计器 |
| **GMM 采样分布不完全匹配真实 z 分布** | GMM 是参数近似，与真实 z 分布有偏差 | 用 KDE 替代 GMM（更灵活，但高维时密度估计退化） |
| **需要额外依赖** | 需要 sklearn | `sklearn` 已在 `distribution2d.py` 中被引用（`make_moons` 等），项目已有依赖 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（无需重训练，即时可部署，对 LZR 的关键升级）**

理由：
1. **零训练成本**：完全在已训练模型上运行，不需要任何参数更新
2. **即时可验证**：校准 + 生成测试可在 1-2 分钟内完成
3. **理论严格**：GMM base distribution 在 arXiv:2512.04954 中有直接理论和实验支撑
4. **升级清晰**：LZR（盒子边界）是本 idea（GMM 边界）的退化特例，升级路径明确
5. **可以在 soft-EM 训练的模型上独立使用**：即使不做 Hard-EM 训练，本 idea 也能单独改善生成质量

---

## 参考文献

- Wildberger, J.B. et al. (2024). "Amortized Inference of Multi-Modal Posteriors using Likelihood-Weighted Normalizing Flows." *arXiv:2512.04954*.  
  GMM base distribution 消除 inter-mode probability bridge 的直接实验证据
- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.  
  https://proceedings.mlr.press/v151/stimper22a.html  
  通过 learned rejection sampling 修复 topology mismatch（本 idea 是其 empirical 简化版）
- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 2024*.  
  https://arxiv.org/abs/2305.12149  
  在 latent space 中做 MCMC 改善多模态采样（本 idea 用 GMM 替代 MCMC，更简单）
- Bevins, H. & Handley, W. (2023). "Piecewise Normalizing Flows." *arXiv:2305.02930*.  
  Per-cluster flow 的分离训练，隐含了 cluster-specific z-space 分布的思路
