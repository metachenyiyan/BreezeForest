# Idea: Responsibility-Filtered Sampling (RFS) — X 空间主动过滤

**创建时间**: 2026-03-11 22:11 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（无需重训练，可立即在任意已训练 MultiBF 上部署）

---

## 问题定义

`MultiBF.inverse_map()` 的当前生成流程：

```python
for k in range(self.n_components):
    z = torch.rand(n_k, self.dim) * 0.98 + 0.01   # Uniform([0.01, 0.99]^d)
    x_k = self.components[k].inverse_map(z, ...)    # bisection: z → x
    results[mask] = x_k
```

**核心问题**：生成的 `x_k` 没有经过任何质量检验。即使组件 k 对应 cluster k，它的 CDF 映射 `f_k` 是全局双射，会将 [0.01, 0.99]^d 中的**每个** z 值都映射到某个 x。由于 inter-cluster 区域的数据点在 `f_k` 的映射下落在某个 z 值，对应地，从那个 z 值做反演会得到 inter-cluster 的 x。

**LZR（已有 idea_latent_zone_restriction_2026-03-11-1235.md）的局限**：
- LZR 在 z 空间用矩形 box（百分位数边界）限制采样范围
- 问题：矩形 box 是对 cluster 在 z 空间分布的粗糙近似，不能精确捕捉 cluster 的真实 z 形状（例如对角方向分布的 cluster 在 z 空间可能是斜椭圆形）
- LZR 的有效性依赖于 cluster 在 z 空间的分布接近矩形，否则矩形 box 要么过宽（漏进 inter-cluster 区域），要么过窄（截断合法样本）

**本 Idea 的出发点**：不在 z 空间施加先验几何约束，而是在 **x 空间直接验证生成样本的合法性**。

---

## 从当前项目代码与已有 idea 中得到的背景判断

**MultiBF 的 density 估计已有现成接口**：
- `MultiBF._per_sample_log_det(bf, x)` 可计算每个样本在某组件下的 log|det J|
- `MultiBF.get_mixture_log_weights()` 可获取 log π_k
- 通过这两个接口，可以无缝计算任意 x 在各组件下的 responsibility r_k(x)

**关键观察**：
- inter-cluster 的点 x 在**任何组件**下的 log|det J_k(x)| 都较低（因为任何组件都没有训练过 inter-cluster 区域的高密度）
- 同时，inter-cluster 点的 responsibility r_k(x) 对任意 k 都不会很高（因为没有组件对它负责）
- 这提供了一个**自然的筛选标准**：r_k(x) < threshold 的样本很可能是 inter-cluster 点

**LZR 的方向正确但实现间接**：LZR 在 z 空间施加约束，间接地减少 inter-cluster 样本。RFS 直接在 x 空间检查 responsibility，是更直接、更准确的方法。

**ICDR（已有 idea_inter_component_density_repulsion_2026-03-11-1240.md）的角色**：ICDR 是训练时的推动（push components apart）；RFS 是推理时的过滤（filter bad samples）。两者互补，RFS 可与任何训练策略搭配使用。

---

## 核心思路

**Responsibility-Filtered Sampling（RFS）**：

在 `MultiBF.inverse_map()` 中，生成 x = f_k^{-1}(z) 后，计算该样本对组件 k 的 responsibility r_k(x)，并根据 responsibility 进行接受/拒绝决策：

**方案 A（硬阈值过滤）**：
```
accept x_k iff r_k(x_k) ≥ threshold
```
即：如果生成的 x 对组件 k 的 responsibility 低于阈值，则丢弃并重新采样。

**方案 B（概率接受）**：
```
accept x_k with probability r_k(x_k)
```
这是 acceptance-rejection sampling 的标准形式：以 r_k(x) 为接受概率。

**关键性质**：
- r_k(x) ∈ [0, 1]，对 cluster k 内部的点接近 1（高 responsibility），对 inter-cluster 点和其他 cluster 的点接近 0
- 在 r_k(x) 上做 rejection 等价于：只生成 cluster k 的典型代表性样本，过滤掉 cluster k "不自信"的样本
- 这不需要任何额外训练，也不依赖 z 空间的几何假设

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**直接针对症状**：

inter-cluster 点 x 的特征是：对任何组件的 responsibility 都低（没有组件"认领"它）。RFS 直接利用这一特征作为过滤标准。

**与 LZR 的对比**：

| 维度 | LZR（已有） | RFS（本 Idea） |
|------|------------|--------------|
| 过滤空间 | z 空间（latent CDF 空间） | x 空间（数据空间） |
| 形状假设 | 矩形 box（独立维度边界） | 无形状假设（基于模型密度） |
| 需要预校准 | 是（需要 calibrate_latent_zones()） | 否（实时计算 responsibility） |
| 对未完全专一化组件的鲁棒性 | 低（zones 可能重叠） | 高（responsibility 在组件混淆时仍能识别低质量点） |
| 计算开销 | 低（校准一次，采样时查表） | 中（每个候选样本需计算 K 个组件的 log_det） |
| 效果保证 | 依赖 z 空间矩形近似质量 | 直接与模型密度估计对齐 |

**理论支撑**：
RFS 本质上是在混合密度 p(x) = Σ_k π_k p_k(x) 的 p_k(x) 作为 "validity score" 来过滤样本。这与 Stimper et al. (2022) 的 resampled base distributions 在精神上一致：通过密度评估来避免从低密度区域生成。

**对比 Amortized Multi-Modal Posterior（arxiv 2512.04954, 2024）**：
该文献发现 unimodal base distribution 会在 disconnected modes 之间制造"spurious probability bridges"。RFS 通过后处理过滤这些 bridge 上的点，是针对该问题的推理时 patch。

---

## 与历史 idea 的关系

**升级 LZR（idea_latent_zone_restriction_2026-03-11-1235.md）**：

- LZR 在 z 空间用矩形 box 近似 cluster 区域（间接方法）
- RFS 在 x 空间用 responsibility 直接验证（直接方法）
- 两者不互斥：可以先用 LZR 减少 z 空间的候选范围（提高接受率），再用 RFS 做最终过滤
- **推荐使用**：当 LZR 单独使用效果不够好（zone 估计不准或有重叠）时，RFS 可作为额外的安全保障

**与 Hard-EM / PLT 的关系**：
- PLT/Hard-EM 训练后，组件专一化好，r_k(x) 对 cluster k 内的点接近 1，RFS 接受率高（效率好）
- 即使不用 PLT/Hard-EM，RFS 在 soft-EM 训练的模型上也有效（接受率低但质量高）

**与 ICDR 的关系**：
- ICDR 在训练时推动组件分离
- RFS 在推理时过滤低质量样本
- 两者从不同阶段共同解决问题

---

## 具体实现建议

### 在 `MultiBF` 中添加 `inverse_map_filtered()` 方法

```python
def _compute_responsibility(self, x, exact=False):
    """
    Compute per-sample responsibility r_k(x) for each component.
    
    :param x: tensor (batch_size, dim)
    :return: responsibilities (K, batch_size)
    """
    log_pi = self.get_mixture_log_weights()  # (K,)
    det_fn = self._per_sample_log_det_exact if exact else self._per_sample_log_det
    
    component_log_probs = []
    for k, bf in enumerate(self.components):
        per_sample_ld = det_fn(bf, x)  # (batch_size,)
        component_log_probs.append(log_pi[k] + per_sample_ld)
    
    stacked = torch.stack(component_log_probs, dim=0)  # (K, batch_size)
    log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)  # (K, batch_size)
    return torch.exp(log_resp)  # (K, batch_size)

def inverse_map_filtered(
    self,
    n_samples,
    max_gap=1e-3,
    decay_ratio=1.0,
    threshold=0.5,
    max_attempts=5,
    exact=False
):
    """
    Generate samples with Responsibility-Filtered Sampling (RFS).
    
    For each generated x_k = f_k^{-1}(z):
      - Compute r_k(x_k) = p_k(x_k) / sum_j p_j(x_k)
      - Accept x_k if r_k(x_k) >= threshold, otherwise resample
    
    :param n_samples: number of samples to generate
    :param threshold: minimum responsibility for acceptance (default 0.5)
    :param max_attempts: maximum resampling attempts per slot
    :return: generated samples (n_samples, dim)
    """
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)
    accepted = torch.zeros(n_samples, dtype=torch.bool)

    for attempt in range(max_attempts):
        # Only resample slots not yet accepted
        unaccepted_mask = ~accepted
        if not unaccepted_mask.any():
            break
        
        for k in range(self.n_components):
            # Find unaccepted slots assigned to component k
            slot_mask = unaccepted_mask & (component_indices == k)
            n_k = slot_mask.sum().item()
            if n_k == 0:
                continue
            
            # Sample new z and generate candidates
            z = torch.rand(n_k, self.dim) * 0.98 + 0.01
            with torch.no_grad():
                x_k = self.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
                
                # Compute responsibility of component k for generated samples
                responsibilities = self._compute_responsibility(x_k, exact=exact)  # (K, n_k)
                r_k = responsibilities[k]  # (n_k,)
                
                # Accept samples where r_k >= threshold
                accept_mask = r_k >= threshold  # (n_k,)
            
            # Store accepted samples in their original slots
            slot_indices = slot_mask.nonzero(as_tuple=True)[0]  # global indices
            accepted_local = accept_mask
            
            for local_idx, global_idx in enumerate(slot_indices):
                if accepted_local[local_idx]:
                    results[global_idx] = x_k[local_idx]
                    accepted[global_idx] = True
        
        if attempt == max_attempts - 1:
            # Final fallback: use whatever was generated (no filter)
            for k in range(self.n_components):
                slot_mask = unaccepted_mask & (component_indices == k)
                n_k = slot_mask.sum().item()
                if n_k == 0:
                    continue
                z = torch.rand(n_k, self.dim) * 0.98 + 0.01
                with torch.no_grad():
                    x_k = self.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
                slot_indices = slot_mask.nonzero(as_tuple=True)[0]
                for local_idx, global_idx in enumerate(slot_indices):
                    results[global_idx] = x_k[local_idx]
    
    return results
```

### 在 `demo_multi_bf.py` 中使用

```python
# 替换原有的 mbf.inverse_map(n_samples=data_size)
with torch.no_grad():
    samples = mbf.inverse_map_filtered(
        n_samples=data_size,
        threshold=0.5,    # 接受阈值：r_k(x) ≥ 0.5
        max_attempts=5    # 最多重采样 5 次
    )
    samples = samples * std + mean
```

### 超参数调优指南

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `threshold` | 0.4 – 0.6 | 越高过滤越严格，接受率越低。从 0.5 开始，如果接受率 < 50% 则降低到 0.4 |
| `max_attempts` | 3 – 10 | 越大最终覆盖率越高，越慢。5 是默认值 |
| 效率监控 | `r_k.mean()` | 监控平均 responsibility，越接近 1 说明模型越专一化 |

### 性能优化版（批量向量化）

```python
def inverse_map_filtered_batched(self, n_samples, threshold=0.5, max_gap=1e-3, oversample_ratio=3.0):
    """
    Faster version: oversample by ratio, filter, keep top n_samples.
    """
    n_oversample = int(n_samples * oversample_ratio)
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_oversample, replacement=True)
    all_samples = torch.zeros(n_oversample, self.dim)
    all_resp = torch.zeros(n_oversample)

    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        z = torch.rand(n_k, self.dim) * 0.98 + 0.01
        with torch.no_grad():
            x_k = self.components[k].inverse_map(z, max_gap=max_gap)
            resp = self._compute_responsibility(x_k)  # (K, n_k)
            r_k = resp[k]  # (n_k,)
        all_samples[mask] = x_k
        all_resp[mask] = r_k

    # Filter: keep samples with high responsibility
    high_resp_mask = all_resp >= threshold
    filtered = all_samples[high_resp_mask]
    
    if len(filtered) >= n_samples:
        return filtered[:n_samples]
    else:
        # If not enough accepted, return all filtered + pad with unfiltered
        pad = all_samples[~high_resp_mask][:n_samples - len(filtered)]
        return torch.cat([filtered, pad], dim=0)
```

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **低接受率（效率问题）** | 若模型组件不专一，大量样本被拒绝，需要多次重采样 | 配合 PLT 训练使组件专一化；或降低 threshold（0.3-0.4）；或使用 oversample 批量版本 |
| **计算开销** | 每个候选样本需计算 K 个组件的 log_det（约 K 倍于无过滤生成） | 使用 oversample 批量版本，一次性生成多倍样本再过滤，避免循环重采样 |
| **极端情况下无法接受** | 若所有组件都混淆严重，r_k(x) 对所有 k 都低，max_attempts 用尽后仍无质量样本 | 这说明模型需要重训练（用 PLT）；fallback 到无过滤生成 |
| **Responsibility 计算不稳定** | 若某组件的 log_det 数值非常小（接近 log(0)），softmax 可能出现数值问题 | BreezeForest 的 `du_dx.clamp(min=0.001)` 已做了保护；必要时添加 log_prob 的 clamp |
| **阈值选择敏感** | 不同训练质量的模型需要不同的 threshold | 提供自动调整：基于训练数据计算期望接受率，将 threshold 设为 r_k(x_train) 的 25th 百分位数 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（推理时即用，无需重训练）**

理由：
1. **零训练成本**：不需要修改模型，不需要重训练，可在已有任何 MultiBF 模型上立即部署
2. **直接针对问题**：responsibility 是模型内置的"对生成点质量的评估"，直接用于过滤是最自然的方法
3. **比 LZR 更鲁棒**：不依赖 z 空间几何假设，适用于任何 cluster 形状
4. **与训练策略解耦**：无论使用 soft-EM、Hard-EM 还是 PLT 训练，RFS 都能在推理时提供保障
5. **效果可量化**：接受率（acceptance rate）直接反映模型质量，是一个有意义的诊断指标

**推荐使用顺序**：
1. 先部署 **RFS**（立即可用，验证 inter-cluster 问题的改善程度）
2. 如果效果不足，说明模型需要重训练 → 使用 **PLT** 重训练
3. PLT 训练后，继续使用 RFS 作为推理时的最后防线

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.  
  (Density-based rejection sampling to avoid low-density regions in normalizing flows)
- arxiv 2512.04954 (2024). "Amortized Inference of Multi-Modal Posteriors using Likelihood-Weighted Normalizing Flows."  
  (Identifies spurious probability bridges between modes; RFS directly filters these bridge samples)
- Devroye, L. (1986). "Non-Uniform Random Variate Generation." Springer.  
  (Acceptance-rejection sampling theory: accept with probability proportional to density ratio)
