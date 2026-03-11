# Idea: Self-Density Rejection Sampling（SDRS）—— 利用模型自身密度过滤 inter-cluster 生成样本

**创建时间**: 2026-03-11 15:21 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（替代旧 ICDR Idea，通用于单 BF 和 MultiBF，无需重训练）

---

## 问题定义

BreezeForest（包括单 BF 和 MultiBF）在生成阶段的 inter-cluster 问题本质上是：

**生成过程（z → x）不知道哪些 x 是"合理的"**。

- 单 BreezeForest：`z ~ Uniform([0.01, 0.99])` → `x = BF^{-1}(z)`，对所有 z 值一视同仁，包括 z-space 中对应 inter-cluster 区域的值
- MultiBF：即使 Idea 1（顺序专家训练）和 Idea 2（GMM latent 采样）都没有应用，生成的候选样本中也会混入 inter-cluster 点

关键洞察：**模型自身已经编码了密度信息**。BreezeForest 的 `train_forward()` 会返回 `log_det`，即 `log |det J_BF(x)|`，也就是 `log p(x)` 的近似（差一个基分布常数）。

**inter-cluster 区域的数学性质**：
- 在 inter-cluster void 中，训练数据密度 `p_data(x) ≈ 0`
- 如果 BF 训练良好，模型估计的密度 `p̂(x) = |det J_BF(x)|` 也会在 inter-cluster void 中偏低
  - 直觉：inter-cluster void 中的 CDF 值变化缓慢（数据稀疏区域），所以 `dF/dx` 小，所以 `log(dF/dx)` 是负数且绝对值大
- 因此，`log p̂(x)` 可以作为一个有效的过滤信号：低 `log p̂(x)` → 可能是 inter-cluster 样本 → 可过滤

旧 ICDR idea（`idea_inter_component_density_repulsion_2026-03-11-1240.md`）通过添加密度排斥正则项来训练时推开组件，但存在以下问题：
- 需要 bisection（在训练时做 inverse_map 来生成 x_k）= 极高计算开销
- 超参数 `icdr_lambda` 调优困难，过大会破坏 NLL 优化
- 只适用于 MultiBF，不适用于单 BF
- 本质上是间接机制（训练时影响组件，再期待生成时改善），而 SDRS 是直接过滤

---

## 从当前项目代码与已有 idea 中得到的背景判断

### 代码分析

`BreezeForest.train_forward()` 已经计算了每批次的 mean log|det J|：
```python
def train_forward(self, x, light=False):
    ...
    x_logDet = torch.sum(torch.mean(torch.log(du_dx), dim=0))
    return x * self.dim_mask, x_logDet  # x_logDet = mean log p(x) over batch
```

这个 `x_logDet` 是密度估计的代理。在生成后，对生成的样本调用 `train_forward` 可以得到其 log 密度估计。

`generate_sample()` 目前的逻辑：
```python
seeds = distribution.sample(torch.Size([sample_size, 2]))
generated = model.inverse_map(seeds)
```
完全没有密度过滤步骤。

`MultiBF.train_forward()` 返回 `mean log p(x)` = `mean logsumexp_k(log π_k + log|det J_k(x)|)`——这是 MultiBF 的整体密度估计，可直接用于 SDRS 过滤。

### 旧 ICDR 的根本问题

旧 ICDR 通过在训练时添加排斥损失来"事后弥补"训练目标的缺陷。但这是间接路径，且存在梯度冲突风险。SDRS 不改变训练，直接在生成后过滤，更简洁、更可控。

### 外部调研背景

- Stimper et al. (AISTATS 2022) "Resampling Base Distributions"：通过学习 rejection sampling（额外神经网络）来修复 base distribution。本 Idea 不需要额外的学习步骤——直接用模型自身的密度来过滤。
- 模型自身密度用于过滤的原理：与 **Importance Sampling** 密切相关：如果从某个 proposal 分布生成样本，再用模型密度加权，可以近似真实分布。SDRS 是 IS 的截断版本（硬过滤）。

---

## 核心思路

**生成时自密度过滤（Self-Density Rejection Sampling, SDRS）**：

1. **过生成**：生成 `N_oversample = α × N_target` 个候选样本（如 `α = 5`）
2. **密度评估**：对每个候选样本 x 计算 `log p̂(x)` 使用已有的 `train_forward` / `MultiBF.train_forward`
3. **设置阈值**：取第 `q%` 分位数作为密度阈值（如 `q = 60%`，意味着只保留密度前 40% 的样本）
4. **过滤**：保留密度高于阈值的样本，丢弃低密度样本

**关键参数**：
- `α`（过生成倍数）：越大，筛出的样本质量越好，但计算代价越高。推荐 `α = 3~10`
- `q`（过滤百分位）：越低的 `q` = 越严格的过滤（保留比例更少），需要更大的 `α` 补偿

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**为什么 inter-cluster 样本会有低密度**：

在 BreezeForest 的训练框架中，模型优化的目标是最大化 `mean log|det J_BF(x)|`。在 inter-cluster void 中：

- 训练数据密度 `p_data(x) ≈ 0`：没有训练样本落在 void 中
- BF 的 CDF 在 void 中缓慢增长：`dF/dx` 小，因为 CDF 在没有数据点的区域斜率趋近于零
- 因此 `log|det J(x)| = Σ_d log(dF_d/dx_d)` 在 void 中很小（负值大）

**对比 cluster 内部**：
- `p_data(x)` 高 → BF 的 CDF 斜率大 → `log(dF/dx)` 大 → 高 `log p̂(x)`

这保证了密度可以区分 cluster 内部（高密度）和 inter-cluster void（低密度），SDRS 的过滤是有效的。

**对比 Idea 1（顺序专家训练）和 Idea 2（GMM latent 采样）**：

| 方面 | Idea 1（训练策略） | Idea 2（latent 采样） | Idea 3（SDRS，本 Idea） |
|------|-----------------|---------------------|----------------------|
| 阶段 | 训练时 | 生成时（校准后） | 生成时（无需校准） |
| 适用范围 | MultiBF | MultiBF | **单 BF + MultiBF** |
| 需要重训练 | 是 | 否（需校准） | **否** |
| 数学原理 | 数据分配专一化 | latent 分布建模 | **模型自密度过滤** |
| 实现复杂度 | 较高 | 中等 | **最低** |

SDRS 是三个 Idea 中唯一**不需要任何额外训练或校准步骤，且同时适用于单 BF 和 MultiBF** 的方法。

---

## 与历史 idea 的关系

**替代旧 ICDR idea（`idea_inter_component_density_repulsion_2026-03-11-1240.md`）**：

旧 ICDR 试图通过训练时的密度排斥来改善生成，但路径间接且代价高。SDRS 直接在生成阶段用模型自身密度过滤，更高效、更可控。

| 方面 | 旧 ICDR | 本 SDRS |
|------|---------|---------|
| 作用阶段 | 训练时 | 生成时 |
| 计算开销 | 高（需要 bisection 在训练中运行） | 低（只需一次 train_forward） |
| 是否需要重训练 | 是 | **否** |
| 超参数调优难度 | 高（`icdr_lambda` 影响 NLL 优化） | 低（只有 `α` 和 `q`，可视化调优） |
| 适用范围 | MultiBF only | **单 BF + MultiBF** |
| 与 ICDR 的联系 | - | 直接替代，效果更可控 |

与 **Stimper 2022（Resampling Base Distributions）** 的关系：
- Stimper 2022 是 SDRS 的**学习版本**：额外训练一个 rejection 网络
- SDRS 是 Stimper 2022 的**零学习版本**：用模型自身密度作为 rejection 准则
- 当 BF 训练良好时，两者效果等价；SDRS 更简单但依赖 BF 密度估计的质量

---

## 具体实现建议

### 单 BreezeForest 的 SDRS

```python
def generate_sample_with_sdrs(model, std, mean, sample_size, 
                               oversample_ratio=5, filter_percentile=60):
    """
    Generate samples with Self-Density Rejection Sampling.
    
    :param model: trained BreezeForest
    :param oversample_ratio: generate this many times more candidates than needed
    :param filter_percentile: keep top (100 - filter_percentile)% by density
    """
    model.eval()
    n_candidates = sample_size * oversample_ratio
    
    with torch.no_grad():
        # Step 1: Generate candidate samples
        distribution = torch.distributions.uniform.Uniform(
            torch.tensor(0.01), torch.tensor(0.99)
        )
        seeds = distribution.sample(torch.Size([n_candidates, model.dim]))
        candidates = model.inverse_map(seeds)  # (n_candidates, dim)
        
        # Step 2: Compute density for each candidate
        # train_forward returns (z, mean_log_det) — we want per-sample log_det
        # Use the numerical derivative approach:
        epsilons = model.epsilon
        x_deltas = torch.cat([
            (candidates - epsilons).view(1, -1, candidates.size(1)),
            (candidates + epsilons).view(1, -1, candidates.size(1))
        ], dim=0)
        
        breeze_list = []
        y = model.forward(candidates, breeze_list)
        x_deltas_out = model.breeze_forward(x_deltas, breeze_list)
        
        du_dx = (x_deltas_out[1] - x_deltas_out[0]) / (2 * epsilons)
        du_dx = torch.abs(du_dx * model.dim_mask + 1 - model.dim_mask).clamp(min=0.001)
        log_density = torch.sum(torch.log(du_dx), dim=1)  # (n_candidates,)
        
        # Step 3: Filter by density threshold
        threshold = torch.quantile(log_density, filter_percentile / 100.0)
        keep_mask = log_density >= threshold
        filtered = candidates[keep_mask]
        
        # Step 4: Subsample to sample_size (if more than needed, random subsample)
        if filtered.shape[0] >= sample_size:
            idx = torch.randperm(filtered.shape[0])[:sample_size]
            result = filtered[idx]
        else:
            # If too few passed the filter, relax threshold
            print(f"Warning: only {filtered.shape[0]} samples passed filter, "
                  f"relaxing threshold to get {sample_size}")
            idx = torch.argsort(log_density, descending=True)[:sample_size]
            result = candidates[idx]
        
        # Rescale back to data space
        result = result * std + mean
        return result.cpu().numpy()
```

### MultiBF 的 SDRS

```python
def inverse_map_with_sdrs(self, n_samples, oversample_ratio=5, 
                           filter_percentile=60, max_gap=1e-3):
    """
    Generate samples from MultiBF with Self-Density Rejection Sampling.
    """
    n_candidates = n_samples * oversample_ratio
    
    # Generate candidates using existing inverse_map
    candidates = self.inverse_map(n_candidates, max_gap=max_gap)  # (n_candidates, dim)
    
    # Compute mixture log-density for each candidate
    with torch.no_grad():
        log_pi = self.get_mixture_log_weights()
        component_log_probs = []
        for k, bf in enumerate(self.components):
            per_sample_ld = self._per_sample_log_det(bf, candidates)
            component_log_probs.append(log_pi[k] + per_sample_ld)
        
        stacked = torch.stack(component_log_probs, dim=0)  # (K, n_candidates)
        log_prob = torch.logsumexp(stacked, dim=0)         # (n_candidates,)
    
    # Filter by density threshold
    threshold = torch.quantile(log_prob, filter_percentile / 100.0)
    keep_mask = log_prob >= threshold
    filtered = candidates[keep_mask]
    
    # Subsample to n_samples
    if filtered.shape[0] >= n_samples:
        idx = torch.randperm(filtered.shape[0])[:n_samples]
        return filtered[idx]
    else:
        idx = torch.argsort(log_prob, descending=True)[:n_samples]
        return candidates[idx]
```

### 超参数调优策略

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| `oversample_ratio` | 3 – 10 | 越大越好，计算代价也越大；建议先用 5 |
| `filter_percentile` | 40 – 70 | 从 50% 开始（保留密度前 50%），若仍有 inter-cluster 样本则增大 |
| 自适应阈值 | 从训练数据密度统计 | 将训练数据的密度第 10 百分位数作为阈值下限，更有原则性 |

**自适应阈值计算**：
```python
# 在训练结束时计算训练数据的密度分布
with torch.no_grad():
    train_log_probs = []
    for batch, _ in data_loader:
        batch_normalized = (batch - mean) / std
        # 单 BF：
        _, log_det = bf.train_forward(batch_normalized)
        # 用 per-sample log_det（见上面代码）替代 mean log_det
        train_log_probs.append(log_det)
    
    train_log_prob_tensor = torch.stack(train_log_probs)
    density_threshold_p10 = torch.quantile(train_log_prob_tensor, 0.10)
    # 生成时拒绝 log p̂(x) < density_threshold_p10 的样本
```

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **密度估计偏差** | 如果 BF 训练不够充分，inter-cluster void 的密度估计不够低，过滤失效 | 增加训练步数或与 Idea 1 组合使用（更好的训练 → 更准确的密度估计） |
| **生成样本偏置** | 过滤后的样本偏向于高密度 cluster 中心，边缘样本可能被过度丢弃 | 用更宽松的 `filter_percentile`（如 40-50%），或仅用自适应阈值（而非硬百分位） |
| **oversample 计算代价** | 需要运行 `α` 倍数的 bisection + 一次 train_forward | bisection 在 CPU 上慢；如果 `dim` 小（≤4），影响不大；建议 batch 化处理 |
| **min density for cluster 边缘** | cluster 边缘的合法样本（低密度但有效）可能被误过滤 | 使用自适应阈值（训练数据密度的 10% 分位）而非硬百分位 |
| **单 BF 上的密度计算方式** | 目前代码中 `train_forward` 返回 mean log_det，需要改为 per-sample log_det | 见上方实现代码中的 per-sample 版本，约 10 行修改 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（独立可用，与 Idea 1 和 2 互补）**

**调整说明**：与旧 ICDR（⭐⭐）相比，本 SDRS 优先级更高，原因：
1. **通用性**：同时适用于单 BF 和 MultiBF，无需修改训练代码
2. **零成本实施**：不需要重训练，只需要修改生成代码
3. **直接有效**：直接利用模型自身对 inter-cluster void 的低密度估计过滤，原理清晰
4. **旧 ICDR 替代**：SDRS 解决同一问题（生成阶段的 inter-cluster 样本）但路径更直接、代价更低
5. **与 Idea 1+2 形成三层保障**：
   - Idea 1（顺序训练）：训练时保证组件专一化
   - Idea 2（GMM 采样）：生成前约束 z-space 采样区域
   - Idea 3（SDRS）：生成后过滤低密度样本（最后防线）

**建议使用顺序**：
1. **立即**：在现有已训练模型上用 SDRS 验证 inter-cluster 问题是否可以直接过滤
2. **短期**：用 Idea 1（顺序专家训练）重训练模型
3. **短期**：加上 Idea 2（GMM latent 采样）
4. **任何时候**：SDRS 作为额外安全网叠加使用

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.  
  https://proceedings.mlr.press/v151/stimper22a.html  
  (Learned rejection sampling for normalizing flows; SDRS is the "no-learning" version)
- Gelfand, A.E. & Smith, A.F.M. (1990). "Sampling-Based Approaches to Calculating Marginal Densities." *JASA*.  
  (Rejection sampling foundation)
- Neal, R.M. (2001). "Annealed Importance Sampling." *Statistics and Computing*.  
  (Importance sampling for normalizing flows; SDRS is a hard-threshold approximation of IS)
- Annealing Flow Generative Models Towards Sampling High-Dimensional and Multi-Modal Distributions. *ICML 2025* (arXiv:2409.20547).  
  (Related work on density-guided sampling; validates that density-based strategies help in multimodal settings)
