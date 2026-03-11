# Idea: Density-Gated Generation (DGG) — Post-Hoc Rejection Sampling at Output

**创建时间**: 2026-03-11 19:34 UTC  
**推荐优先级**: ⭐⭐ 高优先级（全新方向，零训练成本，直接可验证）

---

## 问题定义

MultiBF 在生成时产生 cluster 间无效样本，**根本特征**是：这些无效样本在训练数据分布 p(x) 下具有**极低的密度**。

间无效样本（inter-cluster samples）之所以低密度：
- 它们落在所有 cluster 之间的区域
- 没有任何组件 k 对这个区域有高 Jacobian（没有 cluster 把它映射到高密度区域）
- 因此 p(x) = Σ_k π_k * p_k(x) 在该点极小

**关键洞察**：若我们能在生成后立即评估 p(x)，并拒绝 p(x) 过低的样本，就可以直接过滤掉 inter-cluster 无效样本，无需任何训练修改。

---

## 从项目代码与已有 idea 得到的背景判断

**代码实现能力分析**：
- `MultiBF.train_forward(x)` 可以计算任意输入 x 的 log p(x)（mixture log-likelihood）
- 生成代码（`inverse_map`）：对每个 batch k 生成 z ~ Uniform，然后 bisection 求 x
- **当前完全没有生成后的密度检查**：生成的所有样本不管好坏都被接受

**计算开销评估**：
- 对 3000 个生成样本，评估 log p(x) = 1 次 `train_forward` 调用（batch_size=3000）
- `_per_sample_log_det` 需要 K 个组件的 forward pass，总开销 = K × 2 × forward pass（finite difference）
- 对 K=3, dim=2, batch=3000：约 18000 次神经网络前向传播，在 CPU 上约 1-5 秒
- 生成本身（bisection）通常已是主要开销，密度评估只增加约 10-30%

**与已有 idea 的区别**：
- Idea 1/升级版（Hard-EM/K-Means EM）：修复**训练**，减少组件非专一化问题
- Idea 2/升级版（LZR/CALZS）：修复**latent 采样**，在 z 空间限制输入范围
- 本 Idea（DGG）：修复**生成后**，在 x 空间过滤低密度输出
- **DGG 是唯一完全不需要修改训练或模型参数的 idea**，可直接用于任意已训练模型

---

## 核心思路

**两步 DGG 算法**：

### 步骤 1：确定密度阈值 τ（在训练数据上一次性计算）
对一批训练数据 x_train，计算它们的 log p(x)，取第 p 百分位数（如 p=10）作为阈值 τ：
```
τ = percentile(log p(x_train), 10%)
```
这意味着：至少 90% 的训练数据满足 log p(x) ≥ τ；inter-cluster 样本的 log p(x) << τ。

### 步骤 2：生成时过滤
```
生成 M 个候选样本（M = oversample_factor × n_samples）
评估每个候选样本 x 的 log p(x)
接受 log p(x) ≥ τ 的样本
若接受数量 < n_samples，再生成一批补充
```

**关键自适应**：oversample_factor 根据模型质量自适应：
- 对专一化好的模型（训练后），有效率 > 80%，oversample_factor = 1.5 足够
- 对未专一化的模型，有效率可能 40-60%，oversample_factor = 2.5

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**直觉**：

对于 8 Gaussians 数据集（GAUSSIANS），8 个 cluster 之间有明显的低密度区域。若 p(x) 在 cluster 中心约为 2.0，在 inter-cluster 区域约为 0.001（差了 3 个数量级），则设 τ = percentile(log p(x_train), 10%) 就可以区分几乎所有 inter-cluster 样本。

**数学保证**：

假设训练数据来自真实分布 p*(x)，MultiBF 学习的 p(x) ≈ p*(x)。则：
- 对真实 cluster 内的点 x_in：p(x_in) ≈ p*(x_in) 较大
- 对 inter-cluster 的点 x_out：p(x_out) ≈ p*(x_out) ≈ 0（若 clusters 分离）

因此拒绝 p(x) < τ 的样本 ≈ 拒绝所有 inter-cluster 样本，保留所有 cluster 内样本。

**与其他 idea 的互补性**：

DGG 不要求组件专一化。即使模型完全未专一化（所有组件都覆盖所有 cluster），DGG 仍然有效，因为：
- 混合密度 p(x) = Σ_k π_k * p_k(x) 在任何组件分配方式下，对真实 cluster 内的点总是较高
- 只有 inter-cluster 区域 p(x) 是低的

**可以与 Idea 1 升级版（K-Means EM）、Idea 2 升级版（CALZS）叠加**：
- K-Means EM 减少 inter-cluster 生成的来源（减少候选中的坏样本比例）
- CALZS 进一步收紧 latent 采样范围（减少 inter-cluster 候选）
- DGG 作为最终质量过滤器（消除漏网之鱼）
- 三者叠加：训练 + 采样 + 后处理，三重保护

---

## 与历史 idea 的关系

| 历史 Idea | 关系 | 说明 |
|-----------|------|------|
| Idea 1（Hard-EM, 1230） | 互补 | Hard-EM 减少 inter-cluster 生成的根本原因；DGG 是防御性后处理 |
| Idea 2（LZR, 1235） | 互补（不同维度） | LZR 在 latent z 空间过滤；DGG 在生成的 x 空间过滤 |
| Idea 3（ICDR, 1240） | 互补（不同机制） | ICDR 在训练时推开组件；DGG 在推断时过滤坏样本 |

**本 Idea 不替代任何已有 idea**，而是增加了一个**完全独立的防御层**：

```
训练质量（K-Means EM）→ 采样质量（CALZS）→ 生成质量（DGG）
```

**为何没有被历史 idea 覆盖**：
- 历史 3 个 idea 全部聚焦于训练时或 latent 空间的干预
- 没有任何 idea 提出在 x 空间（生成后）做密度评估过滤
- DGG 是一个独立的正交视角

---

## 具体实现建议

### 步骤 1：计算阈值（训练后一次性）

```python
def compute_generation_threshold(mbf, x_train, percentile=10.0, batch_size=1000):
    """
    Compute log p(x) threshold from training data.
    
    Samples with log p(x) < threshold are classified as "invalid" (inter-cluster).
    
    :param mbf: trained MultiBF instance
    :param x_train: normalized training data (N, dim)
    :param percentile: samples below this percentile are rejected (default 10%)
    :param batch_size: batch size for density evaluation
    :return: (threshold scalar, log p values for training data)
    """
    log_probs = []
    
    with torch.no_grad():
        for i in range(0, len(x_train), batch_size):
            batch = x_train[i:i+batch_size]
            log_p = mbf.train_forward(batch)  # Actually need per-sample; use helper
            
            # Per-sample log p(x)
            log_pi = mbf.get_mixture_log_weights()
            component_log_probs = []
            for k, bf in enumerate(mbf.components):
                ld = mbf._per_sample_log_det(bf, batch)  # (batch_size,)
                component_log_probs.append(log_pi[k] + ld)
            stacked = torch.stack(component_log_probs, dim=0)  # (K, B)
            log_p_per_sample = torch.logsumexp(stacked, dim=0)  # (B,)
            log_probs.append(log_p_per_sample)
    
    log_probs_all = torch.cat(log_probs)  # (N,)
    threshold = torch.quantile(log_probs_all, percentile / 100.0).item()
    
    print(f"Log p(x) statistics on training data:")
    print(f"  Min: {log_probs_all.min().item():.2f}, Max: {log_probs_all.max().item():.2f}")
    print(f"  Mean: {log_probs_all.mean().item():.2f}, Std: {log_probs_all.std().item():.2f}")
    print(f"  Threshold (p={percentile}%): {threshold:.2f}")
    
    return threshold, log_probs_all
```

### 步骤 2：密度过滤的生成器

```python
def inverse_map_with_density_gate(mbf, n_samples, threshold, 
                                   max_gap=1e-3, decay_ratio=1.0,
                                   max_attempts=10, oversample_factor=2.0):
    """
    Generate samples with density-gated filtering.
    
    Rejects samples with log p(x) < threshold (inter-cluster).
    
    :param n_samples: number of ACCEPTED samples to generate
    :param threshold: log p(x) threshold (from compute_generation_threshold)
    :param max_attempts: maximum number of generation rounds
    :param oversample_factor: initial oversampling ratio
    :return: accepted samples (n_samples, dim)
    """
    all_accepted = []
    n_needed = n_samples
    attempts = 0
    
    # Adaptive oversampling
    current_factor = oversample_factor
    
    while n_needed > 0 and attempts < max_attempts:
        n_generate = int(n_needed * current_factor)
        
        # Generate candidate samples using standard inverse_map
        with torch.no_grad():
            candidates = mbf.inverse_map(n_generate, max_gap=max_gap, decay_ratio=decay_ratio)
        
        # Evaluate density for all candidates
        with torch.no_grad():
            log_pi = mbf.get_mixture_log_weights()
            component_log_probs = []
            for k, bf in enumerate(mbf.components):
                ld = mbf._per_sample_log_det(bf, candidates)
                component_log_probs.append(log_pi[k] + ld)
            stacked = torch.stack(component_log_probs, dim=0)
            log_p = torch.logsumexp(stacked, dim=0)  # (n_generate,)
        
        # Filter: accept samples above threshold
        accept_mask = log_p >= threshold
        accepted = candidates[accept_mask]
        
        accept_rate = accept_mask.float().mean().item()
        
        if len(accepted) > 0:
            all_accepted.append(accepted[:n_needed])
            n_needed -= len(accepted[:n_needed])
        
        # Adaptive: if acceptance rate is low, increase oversampling
        if accept_rate < 0.3:
            current_factor = min(current_factor * 2.0, 10.0)
        elif accept_rate > 0.7:
            current_factor = max(current_factor * 0.8, 1.5)
        
        attempts += 1
        print(f"  Round {attempts}: generated {n_generate}, accepted {accept_mask.sum().item()} "
              f"(rate={accept_rate:.1%}), still need {n_needed}")
    
    result = torch.cat(all_accepted)
    
    if len(result) < n_samples:
        print(f"Warning: only {len(result)}/{n_samples} samples accepted after {max_attempts} rounds")
    
    return result[:n_samples]
```

### 步骤 3：集成到 demo_multi_bf.py

```python
# 训练完成后：
# 1. 计算密度阈值（在训练数据上）
with torch.no_grad():
    threshold, log_probs_train = compute_generation_threshold(
        mbf, all_batch, percentile=10.0  # 拒绝 log p(x) 低于 10th 百分位数的样本
    )

# 2. 使用密度过滤的生成
with torch.no_grad():
    samples = inverse_map_with_density_gate(mbf, n_samples=3000, threshold=threshold)
    samples = samples * std + mean
```

### 阈值选择指南

| percentile | 效果 | 适用场景 |
|-----------|------|---------|
| 1% | 非常宽松（只拒绝极端异常值） | 轻微 inter-cluster 问题 |
| 5% | 宽松（保留 95% 训练数据质量） | 一般情况 |
| **10%** | **推荐**（保留 90% 训练数据质量） | 明显的 inter-cluster 问题 |
| 20% | 严格（可能截断部分 cluster 边缘样本） | 严重的 inter-cluster 问题 |
| 50% | 非常严格（仅保留高密度核心样本） | 极端情况，可能改变样本分布 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **生成变慢** | 需要对候选样本做额外的密度评估 + 潜在多轮生成 | 批量评估（batch_size=1000）；若接受率 >50% 实际开销 <2×；用 oversample_factor 调整 |
| **阈值过严格** | 高 percentile 阈值截断 cluster 边缘的合法样本 → 生成分布变窄 | 从 percentile=5% 开始，监控生成样本分布；不要超过 20% |
| **p(x) 估计不准** | BF 用有限差分近似 Jacobian（epsilon=0.0005），会引入误差 | 用 `exact=True` 模式做精确 Jacobian；接受少量假阳性/假阴性 |
| **接受率过低** | 模型训练很差时，大多数生成样本都是 inter-cluster → 多轮生成 | max_attempts 保护；同时配合 K-Means EM 改善训练质量 |
| **改变生成分布** | 过滤后，生成分布不再精确等于 p(x)；相当于用 p̃(x) = p(x) * I[p(x) ≥ τ] / Z | 对于我们的目标（避免无效样本）可以接受；若需要精确 p(x) 采样则用 importance reweighting |
| **Cluster 边缘欠代表** | 若 percentile 较高，cluster 边界的合法样本被过滤 → 生成的 cluster 过于"紧凑" | 使用较低 percentile（5%），或对过滤后的生成分布与训练数据分布做 KL 检验 |

---

## 推荐优先级

**⭐⭐ 高优先级（独立、零成本、可即时验证）**

理由：
1. **完全无需修改训练**：任何现有 MultiBF 模型（不管是否用了 Hard-EM 或 K-Means EM）都可以立即使用
2. **直接针对症状**：inter-cluster 样本的特征就是低 log p(x)，直接过滤最直接
3. **实现极简**：约 40 行代码，主要是 `_per_sample_log_det` 的复用
4. **文献验证**：Hertrich & Gruhlke (arxiv:2407.20444, ICML 2025) 证明了 flow + rejection-resampling 能解决多模态分布的局部极小和低收敛问题，原理完全相同
5. **与 K-Means EM + CALZS 互补**：形成完整的三层防护

**建议使用顺序（当前旧模型验证）**：
1. **立即**：用 DGG 在现有训练模型上验证效果（无需任何修改）
2. **其次**：用 K-Means Epoch EM 重训练，得到专一化模型
3. **之后**：在专一化模型上用 CALZS 精化 latent 采样
4. **最终**：叠加 DGG 作为最终质量过滤层

---

## 与 Stimper 2022 Resampling 的关系

Stimper et al. 2022 的核心是在**基础分布**上做 rejection sampling（修改 prior），目的是让 learned distribution 更好地匹配 target 的 topology。

DGG 的 rejection 是在**生成的 x 空间**上，基于已学习的 log p(x) 直接过滤。两者区别：
- Stimper：学习一个更好的 prior → 改变 q(z) → 改变 p(x)（训练时修改）
- DGG：不改变模型，仅在推断时过滤 x 空间的低密度输出（推断时后处理）

DGG 更简单，不需要任何额外学习，但代价是生成变慢（需要过采样+过滤）。

---

## 参考文献

- Hertrich, J. & Gruhlke, R. (2024). "Importance Corrected Neural JKO Sampling." *arXiv:2407.20444*. Accepted ICML 2025.  
  https://arxiv.org/abs/2407.20444  
  *(Flow + 重要性权重拒绝重采样解决多模态分布问题的理论与实验依据)*
- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.  
  https://proceedings.mlr.press/v151/stimper22a.html  
  *(同类思路：rejection sampling 改善 flow 的多模态覆盖)*
- Bevins, H.T.J. & Handley, W.J. (2023). "Piecewise Normalizing Flows." *arXiv:2305.02930*.  
  *(证明单一 flow 在多模态数据上产生 bridge samples 的具体实验依据)*
- He, K. et al. (2020). "Momentum Contrast for Unsupervised Visual Representation Learning." *CVPR 2020*.  
  *(Contrastive/rejection 机制在生成质量控制中的理论背景)*
