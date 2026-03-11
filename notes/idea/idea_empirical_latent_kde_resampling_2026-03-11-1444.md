# Idea: Empirical Latent KDE Resampling (ELKS)

**创建时间**: 2026-03-11 14:44 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（无需重训练，对单 BF 和 MultiBF 均有效）

---

## 问题定义

BreezeForest（单模型）和 MultiBF（混合模型）在**生成阶段**都存在同一结构性缺陷：

1. 训练数据有多个 cluster（如 A 和 B），它们在 latent 空间（[0,1]^dim）中对应不同的 latent 区域（Z_A 和 Z_B）。
2. 在 cluster 之间的真实空间低密度"间隙"，在 latent 空间中对应一个**极窄的过渡带**（Z_gap）——因为 CDF 在间隙处几乎不增加，所以 Z_gap 体积很小。
3. 但是，当前生成策略从 **Uniform([0.01, 0.99])^dim** 均匀采样，给 Z_gap 分配了与 Z_A、Z_B 同等的采样概率。
4. 因此，一定比例的生成样本会落在 inter-cluster 区域。

**现有 Idea 2（LZR, 1235）的局限**：

- LZR 用轴对齐的矩形边界框来限制 latent 采样范围（per-component 百分位数）。
- 问题：对于单 BF（非 MultiBF），无法应用 per-component 的方法。
- 问题：如果 Z_k 的形状非矩形，或在 latent 空间内存在多个子结构，矩形框会包含大量无效区域。
- 问题：矩形框给框内所有 z 均匀采样，无法区分高密度和低密度 latent 区域。

**本 Idea 的修复方向**：用**核密度估计（KDE）**估计训练数据在 latent 空间的**经验分布**，在生成时从这个经验分布（而非 Uniform）采样。这自然地把采样集中在训练数据真正覆盖的 latent 区域，而非整个 [0.01, 0.99]^dim。

---

## 从当前项目代码与已有 idea 中得到的背景判断

**代码层面关键观察**：

- `BreezeForest.forward(x)` 将 data x ∈ ℝ^dim 映射到 u ∈ (0,1)^dim（通过 sigmoid 激活的 CDF 变换）。
- `BreezeForest.inverse_map(z)` 将 z ∈ (0,1)^dim 通过二分法映射回 x ∈ ℝ^dim。
- `demo_functions.py` 的 `generate_sample` 用 `Uniform(0.01, 0.99)` 采样 z，不考虑 latent 分布的实际结构。
- `MultiBF.inverse_map` 同样用 `torch.rand(n_k, self.dim) * 0.98 + 0.01`（uniform）。
- 训练数据在 latent 空间的经验分布从未被计算或使用。

**对 ELKS 关键性的认知**：

设 `z_train = f(X_train)` 为训练数据通过正向映射的 latent 表示。对 multi-cluster 数据：
- 如果 BreezeForest 学到了近似正确的 CDF 变换，cluster A 的 x 会映射到 Z_A ⊂ [0,1]^dim，cluster B 的 x 会映射到 Z_B ⊂ [0,1]^dim，且 Z_A 和 Z_B 在 latent 空间中相互分离（对应 CDF 中两段"增长区间"）。
- Z_gap（inter-cluster 区域在 latent 空间的像）体积极小（CDF 在数据空间间隙处几乎不变化）。
- z_train 的经验分布因此是**双峰的**（对应 Z_A 和 Z_B），且 Z_gap 中的密度极低。
- **KDE on z_train 自然具有这种双峰结构，并在 Z_gap 处密度最低**。

这正是我们想要的采样分布！

**已有 Idea 2（LZR, 1235）的关系**：ELKS 直接**升级并替代** LZR：
- LZR：矩形框限制，per-component only，适用 MultiBF
- ELKS：KDE 捕获真实 latent 密度，适用**单 BF 和 MultiBF**，不需要矩形框假设

**已有 Idea 3（ICDR, 1240）的关系**：ELKS 在 inference 侧完成了 ICDR 试图在 training 侧完成的事情（减少 inter-cluster 采样），但更简单，不需要修改训练代码。**ICDR 可以被 ELKS 代替**（inference-time 修复，零训练成本）。

---

## 核心思路

### 单 BreezeForest 版本

1. 训练后，对全部训练数据做一次 forward pass：`z_train = f(X_train)` ∈ [0,1]^dim
2. 在 z_train 上拟合 KDE（使用 Gaussian 核，带宽用 Scott's rule 估计）
3. 生成时：从 KDE 采样 z，再调用 `inverse_map(z)`

### MultiBF 版本

1. 训练后，计算各组件的 responsibility（soft-EM 风格）
2. 对组件 k，取 responsibility_k > threshold 的训练样本，通过 `f_k.forward(x)` 计算其 latent 表示
3. 在每个组件 k 的 latent 表示上拟合独立的 KDE_k
4. 生成时：k ~ Categorical(π)，z ~ KDE_k，x = f_k^{-1}(z)

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**直觉说明（以 1D 数据为例）**：

- 数据：cluster A ~ [1,2]，cluster B ~ [4,5]，间隙为 [2,4]。
- 理想 CDF：[1,2] → [0, 0.5]，[4,5] → [0.5, 1.0]，间隙 [2,4] → {0.5}（单点）。
- z_train 的经验分布：z ∈ [0, 0.5] 来自 cluster A，z ∈ [0.5, 1.0] 来自 cluster B，z = 0.5 附近几乎没有训练点。
- KDE on z_train：在 z ≈ 0.0~0.5 和 z ≈ 0.5~1.0 各有一个峰，z ≈ 0.5 附近密度极低。
- 从 KDE 采样：z ≈ 0.5 附近的 z 很少被采到 → `inverse_map(0.5)` ≈ 2-4（间隙区域）被生成的次数极少。

**对比 Uniform 采样**：Uniform 给 z ≈ 0.5 分配与其他值相同的概率，导致间隙区域被均匀采样。

**对比 LZR（矩形框）**：LZR 把 z 限制在 [lo_k, hi_k] 的矩形内，但如果 Z_A 的形状不是矩形（如对角线分布），矩形框会包含大量 Z_A 外的区域，并且不能减少矩形内的"低密度 z"。

**与 Stimper et al. (2022) 的关系**：  
Stimper 的方法是**学习一个参数化的 rejection sampling 基分布**，需要额外训练步骤。ELKS 是其**非参数版本**：用 KDE 直接估计 latent 经验分布，零训练成本，但功能等价（避免 low-density z 采样）。

---

## 它与历史 idea 的关系

| 方面 | 历史 Idea 2（LZR, 1235） | 本 Idea（ELKS） |
|------|--------------------------|----------------|
| 适用范围 | MultiBF only（per-component） | **单 BF + MultiBF** |
| Latent 区域估计 | 轴对齐矩形边界框（百分位数） | **KDE（捕获实际密度形状）** |
| 多峰结构处理 | 不支持（一个框只能表示一个区间） | **支持（KDE 天然多峰）** |
| Z_gap 概率 | 框内均匀（包含 Z_gap） | **低（KDE 在 Z_gap 处密度低）** |
| 实现复杂度 | 低（矩形框即可） | 中（KDE 拟合，1-3 行额外代码） |
| 理论基础 | Stimper 2022（启发） | Stimper 2022（等价非参数版本） |

**历史 Idea 3（ICDR, 1240）**：ELKS 在 inference 时完成了同样的目标（减少 inter-cluster 采样），不需要复杂的训练损失设计。ICDR 可以被 ELKS 替代（在 inference-only 场景下）。

---

## 具体实现建议

### 方案 A：单 BreezeForest（全局 KDE）

```python
from scipy.stats import gaussian_kde
import numpy as np

def calibrate_latent_kde_single(bf, x_train, n_samples_kde=None):
    """
    Fit a KDE on the empirical latent distribution of a single BreezeForest.
    
    :param bf: BreezeForest instance
    :param x_train: training data (N, dim) tensor
    :param n_samples_kde: limit for KDE fitting (None = use all)
    :return: fitted KDE object (scipy.stats.gaussian_kde)
    """
    with torch.no_grad():
        breeze_list = []
        z_train = bf.forward(x_train, breeze_list)  # (N, dim) in [0,1]
    
    z_np = z_train.detach().cpu().numpy()
    if n_samples_kde is not None and z_np.shape[0] > n_samples_kde:
        idx = np.random.choice(z_np.shape[0], n_samples_kde, replace=False)
        z_np = z_np[idx]
    
    # Fit multivariate KDE (Scott's bandwidth rule by default)
    kde = gaussian_kde(z_np.T)  # scipy expects (dim, N)
    return kde


def generate_sample_kde(bf, std, mean, sample_size, kde, batch_size=500):
    """
    Generate samples using KDE-based latent resampling.
    
    :param bf: BreezeForest instance
    :param kde: fitted KDE (scipy gaussian_kde)
    :param std, mean: normalization stats for denormalization
    """
    bf.eval()
    all_samples = []
    remaining = sample_size
    
    with torch.no_grad():
        while remaining > 0:
            n_batch = min(batch_size, remaining * 3)  # oversample for rejection
            
            # Sample from KDE (may produce values outside [0.01, 0.99])
            z_np = kde.resample(n_batch).T  # (n_batch, dim)
            
            # Clamp to valid range
            z_np = np.clip(z_np, 0.01, 0.99)
            z = torch.tensor(z_np, dtype=torch.float32)
            
            # Inverse map
            x = bf.inverse_map(z)
            all_samples.append(x)
            remaining -= x.shape[0]
    
    samples = torch.cat(all_samples, dim=0)[:sample_size]
    return samples * std + mean


# Usage in demo_functions.py
# 1. After training:
#    kde = calibrate_latent_kde_single(bf, x_all_normalized)
# 2. For generation:
#    samples = generate_sample_kde(bf, std, mean, data_size, kde)
```

### 方案 B：MultiBF（Per-Component KDE）

```python
def calibrate_latent_kde_multi(mbf, x_train, resp_threshold=None):
    """
    Fit per-component KDEs on the empirical latent distribution of MultiBF.
    
    :param mbf: MultiBF instance
    :param x_train: training data (N, dim) tensor
    :param resp_threshold: min responsibility to include a sample (None = 1/K)
    :return: list of K kde objects
    """
    if resp_threshold is None:
        resp_threshold = 1.0 / mbf.n_components
    
    with torch.no_grad():
        # Compute responsibilities
        log_pi = mbf.get_mixture_log_weights()
        component_lds = []
        for k, bf in enumerate(mbf.components):
            ld = mbf._per_sample_log_det(bf, x_train)
            component_lds.append(log_pi[k] + ld)
        
        stacked = torch.stack(component_lds, dim=0)   # (K, N)
        log_prob = torch.logsumexp(stacked, dim=0)    # (N,)
        log_resp = stacked - log_prob.unsqueeze(0)    # (K, N)
        resp = torch.exp(log_resp)                    # (K, N)
        
        kdes = []
        for k, bf in enumerate(mbf.components):
            # Select samples with high responsibility for component k
            resp_k = resp[k]  # (N,)
            mask = resp_k > resp_threshold
            
            if mask.sum() < 10:
                # Fallback: top 20%
                topk = max(10, int(0.2 * len(resp_k)))
                _, idx = torch.topk(resp_k, topk)
                mask = torch.zeros_like(resp_k, dtype=torch.bool)
                mask[idx] = True
            
            x_k = x_train[mask]
            
            # Forward pass through component k
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list)  # (n_k, dim) in [0,1]
            
            z_np = z_k.detach().cpu().numpy()
            kde_k = gaussian_kde(z_np.T)
            kdes.append(kde_k)
            
            print(f"Component {k}: {mask.sum().item()} samples for KDE")
    
    return kdes


def inverse_map_kde(mbf, n_samples, kdes, max_gap=1e-3, decay_ratio=1.0):
    """
    Generate samples using per-component KDE latent resampling.
    
    :param mbf: MultiBF instance
    :param kdes: list of K KDE objects (from calibrate_latent_kde_multi)
    :param n_samples: number of samples
    :return: generated samples (n_samples, dim)
    """
    weights = mbf.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, mbf.dim)
    
    for k in range(mbf.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        # Sample from KDE_k
        z_np = kdes[k].resample(n_k).T  # (n_k, dim)
        z_np = np.clip(z_np, 0.01, 0.99)
        z = torch.tensor(z_np, dtype=torch.float32)
        
        x_k = mbf.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
        results[mask] = x_k
    
    return results
```

### 带宽选择建议

| 带宽设置 | 效果 | 适用场景 |
|---------|------|---------|
| Scott's rule（默认） | 平滑，适中保守 | 数据量中等（N > 500） |
| Silverman's rule | 更平滑，bias 略大 | 数据量少（N < 500） |
| `bw_method=0.1`（小带宽） | 紧凑，不平滑 | 组件已经非常专一化时 |
| `bw_method=0.5`（大带宽） | 非常平滑，接近 LZR | soft-EM 训练、组件专一化差时 |

推荐从 Scott's rule 开始，根据可视化效果调整。

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **KDE 边界效应** | KDE 可能在 [0,1]^dim 边界附近有偏差（尾部溢出） | 采样后 `clip(0.01, 0.99)`；或用边界修正 KDE |
| **高维 KDE 失效** | `dim` 大时，高斯 KDE 受维度诅咒影响 | 对 d>4 的数据，用独立维度 KDE（边际 KDE）或 PCA 降维后 KDE |
| **模型不专一化时 KDE 无效** | 若组件完全不专一，KDE_k 包含多个 cluster 的 latent 点，KDE 仍然是多峰的 → 实际上这正好是我们想要的 | 即使如此，KDE 仍优于 Uniform（因为 Z_gap 真的少点） |
| **计算开销** | 高维 KDE 采样较慢 | 一次性生成大量 z 再批量 inverse_map；或用 sklearn.KernelDensity 替代 |
| **单 BF 上效果有限** | 对单 BF，z_train 若不分离（模型没学好），KDE 效果也有限 | 此时建议换用 MultiBF + TAHEM |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（无需重训练，即时可验证）**

理由：
1. **零训练成本**：只需在训练后做一次 forward pass，拟合 KDE，即可改变生成策略。
2. **适用范围最广**：单 BF 和 MultiBF 均适用，LZR 只适用于 MultiBF。
3. **比 LZR 更精确**：捕获实际 latent 密度形状（多峰、非矩形），而非轴对齐框。
4. **理论基础清晰**：是 Stimper et al. (2022) "Resampled Base Distributions" 的非参数近似，有严格理论依据。
5. **即时可验证**：可以在现有模型上立即运行，对比 Uniform 采样和 KDE 采样的生成样本，快速评估效果。

**建议使用顺序（与其他 idea 组合）**：
1. **单独使用 ELKS**：适合快速验证，无需重训练
2. **TAHEM 训练 + ELKS 采样**：最强组合，TAHEM 使组件专一化（KDE 的峰更分离），ELKS 在 inference 时利用这种分离
3. **ELKS + LPTRS**（Idea 3）：双重过滤，ELKS 减少 inter-cluster z，LPTRS 进一步过滤漏网的低密度样本

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a.html （ELKS 的参数化版本，理论依据）
- Scott, D.W. (1992). *Multivariate Density Estimation: Theory, Practice, and Visualization*. Wiley. （Scott's bandwidth rule）
- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 2024*. arXiv 2305.12149. （Latent space 采样修复的相关工作）
- Liu, F. et al. (2025). "StiCTAF: Stick-breaking mixture with component-wise tail adaptation for variational inference." *arXiv 2510.07965*. （非均匀基分布的参数化版本，ELKS 是其简化）
