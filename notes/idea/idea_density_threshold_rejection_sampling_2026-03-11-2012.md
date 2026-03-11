# Idea: Density-Threshold Rejection Sampling (DTRS) — 推断时密度过滤生成

**创建时间**: 2026-03-11 20:12 UTC  
**推荐优先级**: ⭐⭐⭐ 高优先级（可立即实施，无需重训练，比 LZR 更鲁棒）

---

## 问题定义

MultiBF 当前的生成流程（`inverse_map`）对每个组件 k 从 `Uniform(0.01, 0.99)^d` 采样 z，然后通过二分查找反演得到 x。问题在于：

1. 组件 k 的 `f_k` 是**全局连续双射**，将整个数据空间映射到 `(0,1)^d`
2. `Uniform(0.01, 0.99)^d` 涵盖的 z 值，有一部分对应低密度区域的 x（即 cluster 之间的间隙）
3. 当前没有任何机制阻止这些低密度 x 出现在最终生成结果中

**症状**：生成的样本中混入了落在各 cluster 之间区域的"幽灵点"，这些点在原始数据中完全不存在。

**已有方案 LZR（Idea 2, 12:35）的局限性**：
- LZR 通过估计每个组件的 latent cluster zone Z_k 来限制采样范围
- 但 Z_k 的估计依赖于 `soft-EM 训练后的组件质量`：如果组件未专一化，前向传播 `f_k(x)` 对 cluster k 和 cluster j 的数据混杂在一起，Z_k 的边界估计会包含多个 cluster 的 z 范围
- 因此，LZR 在组件质量差时可能失效（zone 估计不准，仍然产生 inter-cluster 样本）

**本方案的出发点**：
- 不依赖 per-component 的 zone 估计
- 直接利用 MultiBF 整体的**混合密度 p(x) = Σ_k π_k p_k(x)** 作为判断标准
- inter-cluster 区域的数据点在混合密度下概率极低（因为没有任何组件在那里有高 Jacobian）
- 通过**生成后拒绝（post-hoc rejection）**，丢弃低密度点，只保留高密度点

---

## 从项目代码与已有 idea 中得到的背景判断

### 代码分析结论

1. `MultiBF.train_forward(x)` 计算 `mean log p(x)`，可以用来对任意 x 评估混合密度 —— **这是 DTRS 的核心工具**
2. `MultiBF.inverse_map(n_samples)` 生成样本 —— DTRS 只需在此基础上增加过滤步骤
3. `_per_sample_log_det(bf, x)` 计算每个样本的 log|det J| —— DTRS 需要 per-sample 密度，而非 batch-mean
4. 生成时 `z = torch.rand(n_k, self.dim) * 0.98 + 0.01`（第 165 行）—— 这是均匀采样，没有密度感知

**关键观察**：`_per_sample_log_det` 已经实现了 per-sample 的 log|det J|，因此计算 per-sample 的 `log p(x)` 只需要再加上 log π_k 的 logsumexp，几乎不需要额外代码。

### 已有 idea 分析

- **LZR（Idea 2, 12:35）**：在 latent 空间限制采样范围 → 需要 per-component zone 估计 → 估计质量依赖组件专一化程度
- **Hard-EM/K-Means（Idea 1, 12:30 / Idea 本轮）**：Training 阶段修复，不是推断阶段修复
- **ICDR（Idea 3, 12:40）**：Training 阶段正则化，不是推断阶段修复

**本 Idea 与 LZR 的根本区别**：
| 维度 | LZR | DTRS |
|------|-----|------|
| 过滤依据 | per-component latent zone（Z_k 边界） | 全局混合密度 p(x) 阈值 |
| 依赖组件专一化 | 是（Z_k 估计依赖组件质量） | 否（p(x) 在任何训练质量下都有意义） |
| 计算位置 | latent 空间（z → x 之前） | data 空间（x 生成之后） |
| 适用条件 | 组件专一化之后效果最好 | 任何条件下都能工作 |
| 实现复杂度 | 中（需要 calibration 步骤） | 低（直接复用 train_forward） |

---

## 核心思路

**推断时密度过滤（DTRS: Density-Threshold Rejection Sampling）**：

1. 生成 `N * (1 + α)` 个候选样本（α 是预期拒绝率的上界，如 α = 0.5）
2. 对每个生成样本 x，计算其在混合分布下的对数密度 `log p(x)`
3. 用训练数据的密度分位数作为阈值 `τ`（如第 `q` 百分位）
4. 丢弃 `log p(x) < τ` 的样本（这些是 cluster 之间的低密度点）
5. 从剩余样本中取前 N 个

**为什么 inter-cluster 样本的 p(x) 低**：
- 训练数据的所有概率质量集中在各 cluster 内部
- 每个 BreezeForest 组件的 Jacobian 在 cluster 内部大（高密度），在 cluster 外部小（低密度）
- 混合密度 `p(x) = Σ_k π_k p_k(x)` 在 cluster 之间接近零，因为没有任何组件在那里有高密度

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**直接针对症状**：inter-cluster 样本的关键特征就是低 p(x)。DTRS 直接用 p(x) 作为过滤条件，最精准地定位和消除这些样本。

**模型无关性**：无论 MultiBF 是用 Soft-EM、Hard-EM 还是 K-Means 预训练，只要训练数据集中在 clusters 内，`p(x)` 就能正确区分高密度和低密度区域。

**理论基础**：
- 这是一种 importance rejection sampling 方法。在生成模型中，用模型自身的密度评估来过滤生成样本是完全合理的 —— Coeurdoux et al. (2024) 的 latent space Langevin sampling 论文展示了类似的思想：在 latent 空间用 Jacobian 引导采样来避免低密度区域。
- DTRS 是一个更简单的离散化版本：先生成再过滤，而非 MCMC 式地在 latent 空间引导采样。

**数学保证**：
设 `S_high = {x : p(x) > τ}` 为高密度区域，`S_low = {x : p(x) ≤ τ}` 为低密度区域（包括 cluster 之间）。
DTRS 将生成分布从 `p(x)` 近似修正为 `p̃(x) ∝ p(x) * 1[p(x) > τ]`，这是截断混合分布，
消除了低密度尾巴对应的 inter-cluster 区域。

**与 Stimper et al. (2022) resampled base distribution 的关系**：
Stimper 在 base distribution（latent 空间）做 rejection sampling，本方案在 output distribution（data 空间）做 rejection sampling —— 效果等价，但本方案不需要修改模型或训练。

---

## 与历史 idea 的关系

| 历史 Idea | 关系类型 | 说明 |
|----------|--------|------|
| LZR（Idea 2, 12:35） | **互补 / 可替代** | LZR 在 latent 空间过滤（更早），DTRS 在 data 空间过滤（更晚但更鲁棒）。两者可以叠加：LZR 先做粗过滤，DTRS 再做精过滤。当组件专一化差时，DTRS 是更可靠的替代。 |
| Hard-EM（Idea 1, 12:30） | **正交 / 互补** | Hard-EM 是 training 阶段修复，DTRS 是 inference 阶段修复。Hard-EM 训练后，DTRS 的过滤效率更高（因为 p(x) 在 inter-cluster 区域更接近零）。 |
| ICDR（Idea 3, 12:40） | **正交 / 互补** | ICDR 是 training 正则化，DTRS 是 inference 过滤。可以叠加。 |
| K-Means 预训练（本轮 Idea 1） | **协同增强** | K-Means 预训练后，组件高度专一化 → p(x) 在 inter-cluster 区域更低 → DTRS 的过滤效果更明显（阈值 τ 与正常样本密度的差距更大）。 |

**与 LZR 的优先级比较**：
- 如果模型已经用 K-Means 预训练：LZR 和 DTRS 都好，推荐 LZR（更直接、更省计算）
- 如果模型用标准 Soft-EM 训练：DTRS 更鲁棒（不依赖 zone 估计质量）
- **最优组合**：K-Means 预训练 → LZR 主过滤 → DTRS 后备过滤

---

## 具体实现建议

### 步骤 1：添加 per-sample 密度计算到 MultiBF

```python
def per_sample_log_prob(self, x, exact=False):
    """
    Compute per-sample log p(x) under the mixture distribution.
    
    log p(x_i) = logsumexp_k( log pi_k + log |det J_k(x_i)| )
    
    :param x: samples to evaluate (N, dim)
    :return: log p(x) per sample, tensor of shape (N,)
    """
    log_pi = self.get_mixture_log_weights()  # (K,)
    det_fn = self._per_sample_log_det_exact if exact else self._per_sample_log_det
    
    component_log_probs = []
    with torch.no_grad():
        for k, bf in enumerate(self.components):
            per_sample_ld = det_fn(bf, x)        # (N,)
            component_log_probs.append(log_pi[k] + per_sample_ld)
    
    stacked = torch.stack(component_log_probs, dim=0)  # (K, N)
    return torch.logsumexp(stacked, dim=0)              # (N,)
```

### 步骤 2：计算训练数据密度分位数（calibration）

```python
def calibrate_density_threshold(self, x_train, percentile=10.0):
    """
    Compute density threshold from training data.
    Samples with log p(x) below this threshold are classified as low-density.
    
    :param x_train: normalized training data (N, dim)
    :param percentile: percentage of training data to set as threshold
                       (e.g., 10.0 means bottom 10% of training density)
    :return: threshold value (scalar)
    """
    with torch.no_grad():
        train_log_probs = self.per_sample_log_prob(x_train)
    threshold = torch.quantile(train_log_probs, percentile / 100.0).item()
    print(f"Density threshold (p{percentile}): {threshold:.4f}")
    print(f"Training data log p(x) stats: "
          f"min={train_log_probs.min().item():.4f}, "
          f"p10={torch.quantile(train_log_probs, 0.1).item():.4f}, "
          f"median={train_log_probs.median().item():.4f}")
    return threshold
```

### 步骤 3：密度过滤生成（DTRS inverse_map）

```python
def inverse_map_with_dtrs(self, n_samples, density_threshold,
                           oversample_factor=2.0, max_attempts=5,
                           max_gap=1e-3, decay_ratio=1.0):
    """
    Generate samples using Density-Threshold Rejection Sampling.
    
    :param n_samples: target number of output samples
    :param density_threshold: log p(x) threshold (from calibrate_density_threshold)
    :param oversample_factor: generate this many extra candidates per iteration
    :param max_attempts: maximum number of regeneration attempts
    :return: filtered samples (≈ n_samples, dim)
    """
    collected = []
    remaining = n_samples
    
    for attempt in range(max_attempts):
        n_generate = int(remaining * oversample_factor) + 100
        
        # Standard generation
        candidates = self.inverse_map(n_generate, max_gap=max_gap, 
                                       decay_ratio=decay_ratio)
        
        # Density evaluation
        with torch.no_grad():
            log_probs = self.per_sample_log_prob(candidates)
        
        # Filter: keep high-density samples
        high_density_mask = log_probs >= density_threshold
        good_samples = candidates[high_density_mask]
        
        print(f"  Attempt {attempt+1}: generated={n_generate}, "
              f"kept={high_density_mask.sum().item()} "
              f"({100*high_density_mask.float().mean().item():.1f}%)")
        
        if len(good_samples) > 0:
            collected.append(good_samples)
            remaining -= len(good_samples)
        
        if remaining <= 0:
            break
    
    if len(collected) == 0:
        print("Warning: No samples passed threshold. Returning unfiltered samples.")
        return self.inverse_map(n_samples, max_gap=max_gap, decay_ratio=decay_ratio)
    
    all_good = torch.cat(collected, dim=0)
    return all_good[:n_samples]  # Trim to exact n_samples
```

### 步骤 4：集成到 demo_multi_bf.py

```python
# 训练完成后：
# 1. 标定密度阈值（使用训练数据）
threshold = mbf.calibrate_density_threshold(
    x_train_normalized, percentile=10.0  # 丢弃密度最低的 10%
)

# 2. 使用 DTRS 生成
with torch.no_grad():
    samples = mbf.inverse_map_with_dtrs(
        n_samples=data_size,
        density_threshold=threshold,
        oversample_factor=2.0  # 生成 2× 数量再过滤
    )
    samples = samples * std + mean
```

### 阈值调优建议

| 阈值百分位 | 效果 | 适用场景 |
|----------|------|---------|
| 5% | 轻度过滤：只去掉最极端的 outlier | 组件专一化较好时 |
| 10% | 标准过滤：去掉低密度尾巴 | **默认推荐** |
| 20% | 激进过滤：更严格，可能截断部分边缘样本 | 组件专一化差，inter-cluster 样本多 |
| 30%+ | 过于激进：可能使生成分布偏离训练分布 | 不推荐，除非 inter-cluster 问题极严重 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **计算开销** | 需要为每个候选样本评估 `per_sample_log_prob`：O(K * N * forward_pass)。当 N 很大时会慢。 | 批处理评估；限制 oversample_factor（如 1.5 而不是 2.0）；或用 LZR 先做粗过滤再用 DTRS 精过滤 |
| **高拒绝率** | 如果组件训练很差，大多数样本都会被拒绝 → 需要很多 attempts | 增大 max_attempts；或降低 percentile 阈值（如从 10% 改为 5%） |
| **密度估计不准** | `_per_sample_log_det` 用有限差分近似（delta=0.0005），有一定误差 | 可以用 `_per_sample_log_det_exact` 代替（精确但更慢）；或通过 calibration 时的稳健性测试调整 |
| **正常样本误拒** | 训练数据中密度偏低的边缘点（cluster 边缘）也可能被过滤 | 用低百分位阈值（5%）；或直接用绝对密度而非百分位（需要理解密度的绝对量级） |
| **与 LZR 的重叠** | 如果已经用 LZR 过滤过，再用 DTRS 可能没有额外收益 | 二者互补但有部分重叠：LZR 在 latent 空间，DTRS 在 data 空间 |

---

## 推荐优先级

**⭐⭐⭐ 高优先级（即时可用，无需重训练）**

理由：
1. **最简单的推断时修复**：约 30 行新代码，不需要修改训练流程
2. **比 LZR 更鲁棒**：LZR 需要好的组件专一化，DTRS 不需要；在任何训练质量下都能工作
3. **直接靶向症状**：inter-cluster 样本就是低密度样本，DTRS 精确识别并丢弃它们
4. **无需额外学习**：复用已训练的 MultiBF 密度估计，零额外训练开销
5. **理论支持**：Coeurdoux et al. (2024) 在 latent 空间的 Langevin 采样从理论上验证了密度引导采样的有效性；本方案是其 data 空间的简化版

**使用建议**：
1. **单独使用**：即使不做 K-Means 预训练，也能立即改善 Soft-EM 训练的 MultiBF 的生成质量
2. **与 K-Means 预训练（本轮 Idea 1）结合**：预训练后 p(x) 更清晰，DTRS 效果更好
3. **与 LZR（Idea 2, 12:35）叠加**：LZR 先限制 z 范围（粗过滤），DTRS 再过滤 p(x) 低的点（精过滤）

---

## 参考文献

- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 2024*. https://arxiv.org/abs/2305.12149  
  （latent 空间密度引导采样的直接相关工作；DTRS 是其 data 空间的简化版）
- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.  
  （在 base distribution 做 rejection sampling，与 DTRS 同一类思路但方向相反）
- Bevins, H. et al. (2023). "Piecewise Normalizing Flows." *arXiv:2305.02930*.  
  （确认了 multi-cluster flow 的生成质量问题的存在及其来源）
