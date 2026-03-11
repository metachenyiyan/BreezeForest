# Idea: Responsibility-Based Rejection Sampling for MultiBF Generation

**创建时间**: 2026-03-11 16:10 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（可立即实施，无需重训练，比 LZR 更鲁棒）

---

## 问题定义

MultiBF 的 `inverse_map` 生成过程：
1. 采样组件 k ~ Categorical(π_k)
2. 采样 z ~ Uniform([0.01, 0.99]^d)
3. x = f_k^{-1}(z)（通过 bisection 求逆）

生成 inter-cluster 点的根本原因：z 的每个值都映射到某个 x，但并非所有 x 都属于组件 k 对应的 cluster。当 z 落在组件 k 的 latent 空间中属于"其他 cluster 的区域"或"inter-cluster 区域"时，f_k^{-1}(z) 就会生成 cluster 之间的无效点。

**当前历史 Idea 2（LZR）的限制**：
- LZR 需要一个 `calibrate_latent_zones()` 步骤，基于训练数据的百分位数估计 zone 边界
- 边界估计是**硬截断**（hard cutoff），不自然，可能截断合法样本边缘
- 当组件未完全专一化时（soft-EM 训练的模型），zone 估计不准确
- **本质问题**：LZR 是在 latent 空间中过滤，而 inter-cluster 问题本质上是一个 data 空间问题

**一个更直接的问法**：生成样本 x = f_k^{-1}(z) 之后，我们如何判断这个 x 真的属于 cluster k，而不是其他 cluster 或 cluster 之间？

**答案**：使用 MultiBF 的混合模型后验 P(k | x) = r_k(x)！

---

## 从当前项目代码与已有 idea 中得到的背景判断

**代码分析**：

1. `MultiBF.inverse_map()`：当前生成过程是"射后不管"——生成 x 后不检查其质量
2. `MultiBF._per_sample_log_det()`：已有高效的单样本 log-density 计算
3. `MultiBF.get_mixture_weights()`：混合权重 π_k 已有
4. 从以上三个 API，可以直接计算 r_k(x) = π_k * p_k(x) / Σ_j π_j * p_j(x)，完全无需任何新的训练

**历史 idea 分析**：
- Idea 2（LZR）：latent 空间中的 pre-filter（在 z 层面）→ 估计不准时效果差
- 本 Idea：data 空间中的 post-filter（在 x 层面）→ 不依赖 z 空间的 zone 估计，直接用模型自身判断
- 两者互补，可以同时使用

**数学严密性**：

设已训练好的 MultiBF 混合模型 p(x) = Σ_k π_k p_k(x)。生成时，组件 k 产生的样本 x 应该满足：

```
P(k | x) = π_k p_k(x) / p(x) > 1/K  （高于均匀责任）
```

如果一个从组件 k 生成的样本 x 满足 P(k | x) < 1/K，说明其他组件比组件 k 更认为这个 x 属于自己。这类样本正是 inter-cluster 样本（或被其他 cluster"认领"的样本）。

---

## 核心思路

**训练后、无需修改任何参数、基于混合模型 posterior 的 rejection sampling**：

```
Algorithm Responsibility-Based Rejection Sampling:
  Input: MultiBF model, n_samples (目标样本数), threshold τ
  Output: n_samples 个生成样本
  
  collected = []
  while len(collected) < n_samples:
    k ~ Categorical(π)              # 采样组件
    z ~ Uniform([0.01, 0.99]^d)    # 采样 latent
    x = f_k^{-1}(z)                # 生成样本
    
    compute r_k(x) = P(k | x)      # 计算 k 的 posterior
    if r_k(x) >= τ:                # 接受条件
      collected.append(x)
  
  return collected
```

**接受条件的直觉**：
- 如果 x = f_k^{-1}(z) 落在 cluster k 附近：r_k(x) ≈ 1 → 接受
- 如果 x 落在 cluster j ≠ k 附近：r_k(x) ≈ 0, r_j(x) ≈ 1 → 拒绝（同时说明 k 应该从 z 更靠近 cluster k 的区域采样）
- 如果 x 落在 inter-cluster 区域：所有 r_k(x) 都很小 → 拒绝

---

## 为什么它适合解决 multi-cluster 中间点生成问题

### 直接阻断 inter-cluster 生成路径

当前问题：f_k^{-1}(z) 对某些 z 值产生 inter-cluster 的 x。这些 x 的特征是：**它们在混合模型下的后验 r_k(x) 很低**，因为混合模型本身（如果训练得好）对 cluster 之间的区域赋予低概率。

Responsibility Rejection Sampling 精确地利用了这个信息：
- 接受 r_k(x) ≥ τ 的样本 → 只保留"模型认为属于组件 k"的样本
- 自动拒绝 inter-cluster 样本

### 与 LZR（Idea 2）的关键差异

| 维度 | LZR（Idea 2） | 本 Idea |
|------|--------------|---------|
| 操作空间 | Latent 空间（z ∈ [0,1]^d）| Data 空间（x ∈ R^d）|
| 过滤基准 | 经验百分位数边界 | 模型 posterior P(k|x) |
| 需要校准 | 是（需要 calibrate_latent_zones()）| 否（直接使用已训练模型）|
| 当组件未专一化时 | Zone 估计不准 → 效果差 | 模型 posterior 仍有效 |
| 适用于高维 | Zone 在高维中估计困难 | Posterior 在任意维度有效 |
| 计算开销 | 低（calibration 一次性）| 中（每个候选样本需计算 K 次 log-density）|
| 能否过滤错误 cluster 分配 | 否（LZR 不检查生成的 x 是否属于 k）| 是（责任度直接反映 x 属于哪个 cluster）|

### 对比 Importance Corrected Neural JKO Sampling（ICLR 2025 submission）

文献中的同类工作（Importance Corrected Neural JKO Sampling）在 CNF 中使用 importance resampling 来处理多模态采样，证明了在 flow 生成中加入拒绝/重采样步骤可以显著改善多模态分布的采样质量。本 Idea 是这类思路在 MultiBF 中的针对性实现。

---

## 它与历史 idea 的关系

**替代 LZR（Idea 2）作为主要推理修复方案，或与之并用**：

- LZR 是 pre-filter（在 z 采样前限制范围）
- 本 Idea 是 post-filter（在 x 生成后检验质量）
- **两者可以叠加**：先用 LZR 缩小 z 范围（减少 bisection 次数），再用 responsibility 过滤剩余的 inter-cluster 样本（消除 LZR 遗漏的误生成）
- **单独使用时**：本 Idea 更鲁棒，不依赖 zone 估计的准确性

**与 Hard-EM（Idea 1）和 K-Means Pre-Training（本轮新 Idea 1）**：
- 这两个 Idea 通过训练改善组件专一化 → r_k(x) 对 cluster k 的峰度更高 → 本 Idea 的过滤效果更强
- 即使没有好的训练（纯 soft-EM），本 Idea 仍然有部分效果（因为 soft-EM 训练的模型 posterior 虽然不清晰，但仍然能识别明显的 inter-cluster 样本）

**与 ICDR（Idea 3）**：
- ICDR 在训练时推开组件
- 本 Idea 在推理时过滤结果
- 互补，不替代

---

## 具体实现建议

### 步骤 1：添加 compute_responsibilities() 方法

```python
def compute_responsibilities(self, x):
    """
    Compute per-component posterior P(k | x) for each sample.
    
    :param x: tensor (batch_size, dim)
    :return: responsibilities (K, batch_size), each column sums to 1
    """
    log_pi = self.get_mixture_log_weights()  # (K,)
    component_log_probs = []
    for k, bf in enumerate(self.components):
        per_sample_ld = self._per_sample_log_det(bf, x)  # (batch_size,)
        component_log_probs.append(log_pi[k] + per_sample_ld)
    
    stacked = torch.stack(component_log_probs, dim=0)  # (K, batch_size)
    log_prob = torch.logsumexp(stacked, dim=0)          # (batch_size,)
    log_resp = stacked - log_prob.unsqueeze(0)          # (K, batch_size)
    return torch.exp(log_resp)                           # (K, batch_size)
```

### 步骤 2：添加 inverse_map_with_rejection() 方法

```python
def inverse_map_with_rejection(
    self, 
    n_samples, 
    responsibility_threshold=None,   # τ，None = 使用 1/K
    max_oversample_ratio=5.0,        # 最多生成 n_samples * ratio 个候选
    max_gap=1e-3,
    decay_ratio=1.0
):
    """
    Generate samples using responsibility-based rejection sampling.
    
    Only accepts generated samples where P(k | x) >= threshold,
    filtering out inter-cluster and wrong-cluster samples.
    
    :param n_samples: number of valid samples to generate
    :param responsibility_threshold: acceptance threshold (default: 1/n_components)
    :param max_oversample_ratio: cap on total candidates to prevent infinite loop
    :return: (valid_samples, acceptance_rate)
    """
    if responsibility_threshold is None:
        responsibility_threshold = 1.0 / self.n_components  # uniform threshold
    
    weights = self.get_mixture_weights().detach()
    
    collected = []
    total_generated = 0
    max_total = int(n_samples * max_oversample_ratio)
    
    # Generate in batches for efficiency
    batch_size = min(n_samples, 500)
    
    while len(collected) < n_samples and total_generated < max_total:
        # Sample component indices
        k_batch = torch.multinomial(weights, batch_size, replacement=True)  # (batch_size,)
        
        results = torch.zeros(batch_size, self.dim)
        for k in range(self.n_components):
            mask = (k_batch == k)
            n_k = mask.sum().item()
            if n_k == 0:
                continue
            z = torch.rand(n_k, self.dim) * 0.98 + 0.01
            x_k = self.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
            results[mask] = x_k
        
        # Compute responsibilities for generated batch
        with torch.no_grad():
            resp = self.compute_responsibilities(results)  # (K, batch_size)
        
        # Accept samples where generating component has high responsibility
        accepted_mask = torch.zeros(batch_size, dtype=torch.bool)
        for i in range(batch_size):
            k_i = k_batch[i].item()
            if resp[k_i, i].item() >= responsibility_threshold:
                accepted_mask[i] = True
        
        accepted = results[accepted_mask]
        collected.append(accepted)
        total_generated += batch_size
    
    all_collected = torch.cat(collected, dim=0)
    if len(all_collected) >= n_samples:
        all_collected = all_collected[:n_samples]
    
    acceptance_rate = len(all_collected) / total_generated
    return all_collected, acceptance_rate
```

### 步骤 3：集成到 demo_multi_bf.py

```python
# 训练完成后，使用 rejection sampling 生成
mbf.eval()
with torch.no_grad():
    samples, accept_rate = mbf.inverse_map_with_rejection(
        n_samples=data_size,
        responsibility_threshold=0.5,  # 要求 dominant component
        max_oversample_ratio=10.0      # 最多生成 10x 样本数来找够合法样本
    )
    print(f"Acceptance rate: {accept_rate:.2%}")  # 诊断信息
    samples = samples * std + mean
```

### 步骤 4：超参数调优策略

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| `responsibility_threshold` | 0.4 - 0.7 | 越高 → 越纯净但拒绝率越高。从 0.5 开始（过半概率属于该组件）|
| `max_oversample_ratio` | 3 - 20 | 拒绝率高时需要更大。可通过 acceptance_rate 监控调整 |
| `batch_size` for rejection | 200 - 1000 | 较大 batch 的 responsibility 计算更稳定 |

**诊断建议**：
```python
# 监控 acceptance_rate 来判断模型质量
# 好的训练（组件高度专一）→ acceptance_rate > 80%
# 差的训练（soft-EM，组件重叠）→ acceptance_rate < 50%
# acceptance_rate 本身是 multi-cluster 分离质量的定量指标！
```

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **接受率过低** | 若模型训练得差（组件混淆严重），大量样本被拒绝，生成很慢 | 调低 threshold（如 1/K）；同时使用 Hard-EM 或 K-Means Pre-Train 改善训练 |
| **生成分布偏移** | 拒绝后的样本不再是 p(x) 的无偏采样，而是 "条件于 cluster 清晰的样本"的分布 | 这实际上是期望的行为；但如需无偏，可改用 importance sampling 而非 rejection |
| **计算开销** | 每个候选样本需要计算 K 次 log|det J|，是原来的 K 倍 | 使用批处理优化；K 通常较小（3-5），开销可接受 |
| **极端情况：所有样本都被拒绝** | 若模型完全没有分离出 cluster，所有 r_k < threshold | 设置 max_oversample_ratio 上限，回退到无过滤生成；同时给出警告 |
| **数值精度** | log-density 计算中的 Jacobian clamp (min=0.001) 可能引入偏差 | 使用 exact Jacobian (jacrev) 获得更准确的 responsibility 估计 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（与 LZR 并列，且更鲁棒）**

理由：
1. **零重训练成本**：不需要修改任何训练代码或重新训练模型，立即可用
2. **数学上更严格**：直接使用混合模型 posterior 作为质量判断，比 LZR 的经验百分位数更可靠
3. **自适应**：threshold 是统一的（1/K），无需针对每个模型调参
4. **诊断价值**：acceptance_rate 是模型多 cluster 分离质量的定量度量
5. **高维泛化**：Posterior-based filtering 在高维空间仍然有效，LZR 在高维中 zone 估计困难
6. **可与 LZR 叠加**：先 LZR 缩小 z 范围，再 rejection sampling 过滤，双重保障

**建议优先尝试顺序**（无需重训练的快速实验）：
1. 先应用本 Idea（责任度拒绝采样），观察 acceptance_rate 和生成质量
2. 若 acceptance_rate < 60%，说明模型训练本身有问题 → 再考虑 K-Means Pre-Training 或 Hard-EM
3. 若 acceptance_rate ≥ 70% 且生成质量仍差 → LZR 和 rejection sampling 双重使用

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. (同类思路前置工作)
- Importance Corrected Neural JKO Sampling. (2025). *OpenReview (ICLR 2025)*. https://openreview.net/forum?id=yQBZZeWPdQ (Importance resampling in flow generation for multimodal distributions - 理论基础)
- DiverseFlow: Morshed & Boddeti (2025). "DiverseFlow: Sample-Efficient Diverse Mode Coverage in Flows." *CVPR 2025*. (DPP-based 推理时多样性增强，同类推理层面改进)
- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 2024*. (Latent 空间采样改进的同类工作)
