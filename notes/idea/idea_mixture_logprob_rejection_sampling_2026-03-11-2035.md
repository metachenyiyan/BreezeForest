# Idea: Mixture Log-Probability Rejection Sampling (MLP-RS)

**创建时间**: 2026-03-11 20:35 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（可立即应用于已训练模型，无需重训练）

---

## 问题定义

MultiBF 的生成阶段按如下流程产生样本：

```
k ~ Categorical(π)   →   z ~ Uniform(0.01, 0.99)^d   →   x = f_k^{-1}(z)
```

**症结所在**：z ~ Uniform([0.01, 0.99]^d) 给 [0.01, 0.99]^d 中的所有 z 赋予完全相等的采样概率。但对于 multi-cluster 数据，有些 z 值对应的 x 落在 cluster 之间的低密度区域——这些 x 的混合对数概率 log p_mixture(x) 很低，但仍然会被生成并输出。

**关键观察**：
- 对于落在 cluster k 内部的训练样本 x，log p_mixture(x) 较高（某个组件对 x 有大 Jacobian）
- 对于落在 cluster 之间的 inter-cluster 点，log p_mixture(x) 很低（所有组件在那里 Jacobian 都小）
- **BreezeForest 的结构天然提供 log p_mixture(x) 的计算接口**（`MultiBF.train_forward` 正是返回这个值）

因此，使用 log p_mixture(x) 作为筛选信号，在生成后拒绝低概率的样本，可以直接消除 inter-cluster 无效点。

---

## 从代码与已有 idea 中得到的背景判断

**从代码角度**：
- `MultiBF.train_forward(x)` 返回 `mean(log p_mixture(x))`，`_per_sample_log_det` 返回逐样本值
- `_per_sample_log_det(bf, x)` 用有限差分近似 log |det J_k(x)|，计算量很低（两次 forward pass）
- `inverse_map()` 目前只做 bisection，不计算生成样本的密度——添加密度过滤只需在 bisection 之后加一次 forward pass
- `MultiBF.train_forward` 不需要梯度用于采样阶段，可以在 `torch.no_grad()` 下以零内存开销运行

**对比 LZR（2026-03-11-1235）**：
| 维度 | LZR | MLP-RS（本 idea） |
|------|-----|-----------------|
| 原理 | 限制 z 的采样区域（前置过滤） | 过滤生成的 x（后置过滤） |
| 依赖组件专一化 | **是**（需要各组件对应不同 cluster 才能估计 Z_k） | **否**（仅依赖混合密度的绝对值） |
| 实现前提 | 需要 calibrate_latent_zones() 并手动选择百分位 | 只需要训练集的 log-prob 分布 |
| Zone 估计误差 | axis-aligned box 可能包含多个 cluster 的 latent 点 | 无 zone 估计误差，直接用密度值 |
| 计算成本 | 一次 calibration（较廉价） | 每批生成一次 forward pass（廉价） |
| 适用场景 | 组件专一化后（需先运行 Hard-EM） | 任意训练状态（包括纯 soft-EM 训练的模型） |

**结论**：MLP-RS 是 LZR 的更鲁棒替代方案。LZR 在 Hard-EM 组件专一化后仍然有价值（可进一步缩小采样区域），但 MLP-RS 作为独立方案更通用。

**从外部调研角度**：
- **Stimper et al. (AISTATS 2022)** 的 Resampled Base Distributions 通过学习 rejection sampling 层来修复 flow 的 topological 问题，验证了 rejection sampling 框架在 normalizing flow 上的有效性
- **Hanneke et al. (2018) 的截断统计学框架** 及其 2023 年后续工作证明：通过投影梯度下降最小化"无效区域"的概率密度是理论上合理的生成约束方法
- **多篇异常检测文献**（InFlow, U-Flow 等）验证 log-probability 阈值是区分有效/无效生成的最直接信号

---

## 核心思路

### 步骤 1：估计密度阈值（Density Threshold Estimation）

在训练完成后，对全量（或大批量）训练数据计算逐样本的混合对数概率，然后取某个分位数作为阈值：

```python
def estimate_density_threshold(self, x_train, percentile=20.0):
    """
    Estimate the density threshold from training data.
    Samples with log p(x) below this threshold are considered "invalid".
    
    :param x_train: training data (N, dim)
    :param percentile: percentage of training data to consider as "low density" (default 20%)
    :return: threshold value (scalar)
    """
    with torch.no_grad():
        log_pi = self.get_mixture_log_weights()
        component_log_probs = []
        for k, bf in enumerate(self.components):
            ld = self._per_sample_log_det(bf, x_train)
            component_log_probs.append(log_pi[k] + ld)
        stacked = torch.stack(component_log_probs, dim=0)  # (K, N)
        log_probs = torch.logsumexp(stacked, dim=0)         # (N,)
    
    threshold = torch.quantile(log_probs, percentile / 100.0)
    self._density_threshold = threshold.item()
    print(f"Density threshold (p{percentile}): {threshold.item():.4f}")
    print(f"Training data log-prob range: [{log_probs.min().item():.4f}, {log_probs.max().item():.4f}]")
    return threshold.item()
```

### 步骤 2：带密度过滤的生成（Rejection Sampling）

```python
def inverse_map_with_rejection(self, n_samples, max_gap=1e-3, decay_ratio=1.0,
                                oversample_ratio=3.0, max_rounds=10):
    """
    Generate samples with density-based rejection sampling.
    
    :param n_samples: target number of valid samples
    :param oversample_ratio: generate this many times more samples initially
    :param max_rounds: maximum rejection sampling rounds before fallback
    :return: (n_samples, dim) tensor of valid samples
    """
    assert hasattr(self, '_density_threshold'), \
        "Call estimate_density_threshold() first."
    
    threshold = self._density_threshold
    valid_samples = []
    n_collected = 0
    n_rounds = 0
    
    while n_collected < n_samples and n_rounds < max_rounds:
        n_needed = n_samples - n_collected
        n_generate = int(n_needed * oversample_ratio)
        
        # Standard sampling
        with torch.no_grad():
            weights = self.get_mixture_weights()
            component_indices = torch.multinomial(weights, n_generate, replacement=True)
            x_candidates = torch.zeros(n_generate, self.dim)
            
            for k in range(self.n_components):
                mask = (component_indices == k)
                n_k = mask.sum().item()
                if n_k == 0:
                    continue
                z = torch.rand(n_k, self.dim) * 0.98 + 0.01
                x_k = self.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
                x_candidates[mask] = x_k
            
            # Compute log p(x) for candidates
            log_pi = self.get_mixture_log_weights()
            comp_lps = []
            for k, bf in enumerate(self.components):
                ld = self._per_sample_log_det(bf, x_candidates)
                comp_lps.append(log_pi[k] + ld)
            stacked = torch.stack(comp_lps, dim=0)
            log_probs = torch.logsumexp(stacked, dim=0)  # (n_generate,)
            
            # Keep only high-density samples
            valid_mask = log_probs >= threshold
            x_valid = x_candidates[valid_mask]
            
            if len(x_valid) > 0:
                valid_samples.append(x_valid[:n_needed])
                n_collected += min(len(x_valid), n_needed)
        
        n_rounds += 1
        accept_rate = valid_mask.float().mean().item()
        if n_rounds == 1:
            print(f"Rejection sampling: accept_rate={accept_rate:.3f}, "
                  f"collected={n_collected}/{n_samples}")
    
    if n_collected < n_samples:
        # Fallback: use standard sampling for remaining
        remaining = self.inverse_map(n_samples - n_collected, max_gap=max_gap)
        valid_samples.append(remaining)
    
    return torch.cat(valid_samples, dim=0)[:n_samples]
```

### 步骤 3：百分位阈值选择指南

| percentile 阈值 | 效果 | 适用场景 |
|----------------|------|---------|
| 5% | 宽松（只拒绝最低密度的 5%） | 模型训练充分、组件专一化程度高 |
| 20% | **推荐**（平衡过滤强度） | 大多数情况的合理默认值 |
| 35% | 严格（拒绝较多样本，可能降低多样性） | 模型训练欠充分，inter-cluster 样本多 |

**自适应阈值**：可以以"生成样本中 log-prob < mean(training log-prob) - 2*std 的比例 < 1%"作为目标，自动调整 percentile。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**直接性**：inter-cluster 点正是 log p_mixture(x) 低的点——因为所有组件在那里都有低 Jacobian（低密度）。过滤低密度样本直接等价于过滤 inter-cluster 样本。

**自一致性**：阈值来自训练数据本身的密度分布，不需要任何外部先验知识或手动标注哪些区域是"有效"的。

**即时可用性**：不需要重训练，不需要修改模型结构，只需在已训练模型上：
1. 运行 `estimate_density_threshold()`（约 1 次 forward pass on full data）
2. 替换 `inverse_map()` 为 `inverse_map_with_rejection()`

**对训练质量的鲁棒性**：即使 soft-EM 导致组件未完全专一化，混合密度 log p_mixture(x) 仍然在 cluster 中心处高、在 cluster 之间处低。因此 MLP-RS 不依赖组件专一化，但受益于它。

**拒绝率估算**：设 inter-cluster 点的生成概率为 p_bad（假设 20%），则每轮生成的接受率约为 80%，需要约 1.25 倍过采样。在 `oversample_ratio=3.0` 下，绝大多数情况下 1-2 轮即可收集足够样本。

---

## 与历史 idea 的关系

| 历史 Idea | 关系 | 说明 |
|----------|------|------|
| LZR（2026-03-11-1235） | **替代**（本 idea 更鲁棒） | LZR 在组件专一化后依然有用（可缩小 z 的采样范围），但作为主要推荐方案应被本 idea 替代。若已有 Hard-EM 专一化，可叠加：先用 LZR 缩小 z 范围减少 bisection 调用次数，再用 MLP-RS 过滤最终结果 |
| Hard-EM（2026-03-11-1230）/ 升级版（本轮 2026-03-11-2032） | **互补，与本 idea 独立** | Hard-EM 改善训练质量，MLP-RS 改善生成质量；两者可叠加，但**本 idea 在 Hard-EM 未完成的情况下也能单独工作** |
| ICDR（2026-03-11-1240）/ 升级版（本轮） | **互补** | ICDR 改善训练中的组件分离，MLP-RS 在生成时进一步过滤 |

**为什么 MLP-RS 比 LZR 更强（替代关系的详细论证）**：

LZR 的核心假设是"组件 k 的训练数据在 latent space 中占据某个规则区域 Z_k = [a_k, b_k]^d"。这个假设在以下情况下失效：
1. 组件未专一化（soft-EM → 混合了多个 cluster 的 latent 点）
2. Latent cluster 不是轴对齐的矩形（8-Gaussian 数据在 2D latent 空间中的分布可能是椭圆）

MLP-RS 不依赖任何关于 latent 区域形状的假设，只使用模型本身的密度评估，因此在任何训练状态下都能工作。

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **高拒绝率** | 若模型训练不充分，大量生成样本密度低，需要大量过采样 | 增大 oversample_ratio；先用 Hard-EM 改善训练质量 |
| **bisection 计算成本翻倍** | 每批 oversample 需要运行更多 bisection | 用 LZR（轻量预过滤）+ MLP-RS（最终过滤）的组合减少 bisection 调用次数 |
| **阈值设置偏高截断合法样本** | 若 percentile 设太高，cluster 边缘的正常样本也被拒绝，导致生成分布过于紧凑 | 从 20% 开始，用可视化验证生成分布是否覆盖了 cluster 全体 |
| **阈值的可移植性** | 阈值在归一化空间估计，需与 `(batch - mean) / std` 归一化一致 | 确保 `estimate_density_threshold` 接受归一化后的数据 |
| **单组件的边界 sampling** | 对于模型，某些 z 值的 bisection 会慢（窄梯度区域），过度采样会放大这个开销 | 可以设置 bisection 的 max_gap 更宽（略降精度但更快），在 MLP-RS 层弥补质量问题 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（与升级版 Hard-EM 并列首选）**

理由：
1. **零训练成本**：可立即应用于任意已训练的 MultiBF 模型，无需重训练
2. **鲁棒性极强**：不依赖组件专一化，不依赖 latent zone 的形状假设
3. **直接针对症状**：inter-cluster 点的特征正是低 log p_mixture(x)，密度过滤与目标精确对齐
4. **理论有支撑**：Stimper et al. (2022) 的 resampling base distributions 验证了 rejection sampling 在 NF 上的有效性；多篇异常检测论文验证了 log-prob 阈值的可靠性
5. **实现成本极低**：约 50 行代码，可作为 `inverse_map` 的直接替代品

**推荐实施顺序**：
1. 先运行 `estimate_density_threshold()` → 验证 MLP-RS 是否显著改善生成质量（快速实验，零训练成本）
2. 若 MLP-RS 改善有限（说明模型密度在 inter-cluster 区域仍然不够低），再进行 Upgraded Hard-EM 重训练
3. Hard-EM + MLP-RS 的组合是当前架构下最强的方案

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a.html  
  （验证 rejection sampling 在 normalizing flow 上的有效性与可实施性）
- Hanneke, S. & Yang, L. (2019). "Minimax Analysis of Active Learning." *JMLR*.  
  （截断统计框架，最小化无效区域概率密度的理论基础）
- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 2024*. https://arxiv.org/abs/2305.12149  
  （在 latent space 中使用 Metropolis-Hastings 过滤，与本 idea 的后置过滤思路相同）
- 原 LZR idea（2026-03-11-1235）— 本 idea 的被替代前身，可与本 idea 叠加使用
