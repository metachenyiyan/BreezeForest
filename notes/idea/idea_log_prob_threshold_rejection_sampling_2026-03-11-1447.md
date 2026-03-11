# Idea: Log-Probability Threshold Rejection Sampling (LPTRS)

**创建时间**: 2026-03-11 14:47 UTC  
**推荐优先级**: ⭐⭐ 高优先级（最简单的即时修复，可作为任何方案的最后一层防线）

---

## 问题定义

BreezeForest 和 MultiBF 的 inter-cluster 误生成问题，本质上是：**model 对 inter-cluster 区域赋予了非零但低的密度**，而生成策略（uniform latent sampling）没有利用这个密度信息来过滤生成结果。

关键洞察：

- 训练目标是**最大化 log|det J(x)|** 对训练数据 x。
- inter-cluster 区域的 x 从未出现在训练数据中，model 对这些 x 的 log|det J(x)| **没有被优化**，通常较低（接近 0 或负值）。
- 训练数据的 log|det J(x)| 通常较高（正值）。
- **因此，我们可以用 log p(x) = log|det J(x)| 作为区分"合法 cluster 内样本"与"inter-cluster 无效样本"的信号**。

**已有 Idea 3（ICDR, 1240）的问题**：

ICDR 试图在训练时通过增加 inter-component density repulsion 来解决这个问题，但：
- 需要在训练中运行 `inverse_map`（bisection，计算代价高）
- V2 版本（training batch 代理）是一种间接近似
- 超参数 λ 需要仔细调优
- 不能修复已训练好的模型，必须重训练

**本 Idea 的修复方向**：在**推理阶段（无需重训练）**，利用模型自身的 log p 信号，对生成样本进行后处理过滤。

---

## 从当前项目代码与已有 idea 中得到的背景判断

**代码层面关键观察**：

1. `BreezeForest.train_forward(x)` 返回 `(u, log_det)`，其中 `log_det = log|det J(x)|`。
   - 对训练数据：log_det 被最大化，因此高
   - 对 inter-cluster 区域：log_det 未被最大化，通常低

2. `MultiBF.train_forward(x)` 返回 `mean log p(x) = logsumexp_k(log π_k + log|det J_k|)`。
   - 同样，训练数据的 log p 高，inter-cluster 区域的 log p 低

3. 生成时（`inverse_map`）：从 Uniform 采样 z，通过 bisection 得到 x，**完全不计算 x 的 log p**。

4. 因此：只需在生成后计算 log p(x)，并与阈值比较，即可过滤低密度样本。

**为什么 inter-cluster 点有低 log p**：

- BreezeForest 的 CDF 变换在 cluster 内快速增长（高 Jacobian），在 cluster 外缓慢变化（低 Jacobian）。
- Inter-cluster 区域没有训练数据，CDF 在这里几乎是平的（近似常数），对应极低的 Jacobian（接近 0）。
- `clamp(min=0.001)` 防止了数值崩溃，但允许 log|det J| ≈ log(0.001) ≈ -6.9（很低）。
- 训练数据的 log|det J| 通常在较高的正值附近。

**已有 Idea 3（ICDR, 1240）关系**：LPTRS 在 inference 时直接实现了 ICDR 想在 training 时达到的效果（减少低密度生成），但更简单、无需重训练。**ICDR 在有 LPTRS 的情况下价值降低**，不再是必须的补充。

---

## 核心思路

**三步流程**：

1. **Calibration**（校准阶段，训练后一次性执行）：
   - 对训练数据计算 log p(x) 分布
   - 设定阈值 θ = percentile_p(log p(X_train))（如第 10 百分位数）

2. **Generate-then-Filter**（生成+过滤循环）：
   - 批量生成 x = f^{-1}(z)，计算每个样本的 log p(x)
   - 保留 log p(x) > θ 的样本，丢弃 log p(x) ≤ θ 的样本
   - 重复直到收集到足够数量

3. **Adaptive threshold**（自适应阈值，可选）：
   - 如果拒绝率 > 50%，适当降低 θ
   - 如果拒绝率 < 10%，适当提高 θ

**数学理由**：

设 p_data(x) 为真实数据分布，p_model(x) 为学得的模型密度，则：

- 接受条件：`p_model(x) > θ`（等价于 `log p_model(x) > log θ`）
- 这是一个**针对 p_model 的截断分布**：`q(x) ∝ p_model(x) * 1[p_model(x) > θ]`
- 对于 inter-cluster 点：`p_model(x) ≈ 0`（未被训练优化）→ 几乎全部被拒绝
- 对于 cluster 内点：`p_model(x)` 高 → 大部分被接受

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**直接机制**：

- 对 2-cluster 数据，inter-cluster 区域的 x 有 log p ≈ log(near 0)（极负）
- Cluster 内部的 x 有 log p ≈ (positive, well-trained)
- θ 设为训练数据 log p 的第 10 百分位数：90% 的训练数据都能通过，而 inter-cluster 点几乎全部被过滤

**不需要任何假设**：

- 不需要知道有多少个 cluster
- 不需要知道 cluster 的位置
- 不需要训练时有任何特殊策略
- 对单 BF 和 MultiBF 均适用
- 对任何已训练的模型均适用（zero-shot fix）

**与现有文献的对应**：

- **Stimper et al. (2022)** "Resampling Base Distributions"：通过 learned rejection sampling 在 latent 空间实现等效效果。LPTRS 在 data 空间直接实现，更简单。
- **Verine et al. (2024)** "Optimal Budgeted Rejection Sampling for Generative Models"（OBRS, AISTATS 2024）：在有限采样预算下最优化 rejection sampling 策略。LPTRS 是 OBRS 的特例（密度阈值版本）。
- **Na et al. (2024)** "Diffusion Rejection Sampling"（ICML 2024）：将 rejection sampling 应用于扩散模型的中间步骤。LPTRS 对 BreezeForest 应用相同的 post-generation 过滤思路。

---

## 它与历史 idea 的关系

**替代 Idea 3（ICDR, 1240）**：

| 方面 | 历史 Idea 3（ICDR, 1240） | 本 Idea（LPTRS） |
|------|--------------------------|----------------|
| 作用时机 | Training time | **Inference time（无需重训练）** |
| 机制 | 显式排斥 loss，推开组件密度 | **直接过滤低 log p 样本** |
| 计算成本 | 高（训练中每步运行 inverse_map 或 proxy） | **极低（仅需 train_forward 一次前向传播）** |
| 超参数 | λ（排斥强度），n_gen_samples（生成数） | **仅阈值 θ（可从数据自动校准）** |
| 适用范围 | MultiBF only | **单 BF + MultiBF** |
| 理论基础 | 对比学习 repulsive loss | **Rejection Sampling 理论（Verine 2024，Stimper 2022）** |
| 可与 TAHEM/ELKS 叠加 | 复杂（需三者同时训练） | **完全正交，可简单叠加** |

**结论**：LPTRS 在功能上**替代 ICDR**，且更简单、更通用、理论更清晰。ICDR 可以不再列为优先 idea。

---

## 具体实现建议

### 步骤 1：Calibration（校准阈值）

```python
def calibrate_log_prob_threshold(model, x_train, percentile=10.0):
    """
    Compute log p(x) for training data, return the percentile-th quantile as threshold.
    
    Works for both BreezeForest and MultiBF.
    
    :param model: BreezeForest or MultiBF instance
    :param x_train: training data (N, dim) tensor
    :param percentile: lower percentile for threshold (default: 10%)
    :return: threshold theta (scalar)
    """
    model.eval()
    log_probs = []
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size]
            
            if isinstance(model, MultiBF):
                lp = model.train_forward(x_batch)  # scalar (mean), but we need per-sample
                # Re-compute per-sample log prob
                from model.MultiBF import MultiBF
                log_pi = model.get_mixture_log_weights()
                comp_lds = []
                for k, bf in enumerate(model.components):
                    ld = model._per_sample_log_det(bf, x_batch)  # (batch,)
                    comp_lds.append(log_pi[k] + ld)
                stacked = torch.stack(comp_lds, dim=0)
                per_sample_lp = torch.logsumexp(stacked, dim=0)  # (batch,)
            else:
                # BreezeForest: log p(x) = log|det J(x)|
                _, log_det = model.train_forward(x_batch)
                # train_forward returns mean over batch, we need per-sample
                # Use the sum formulation:
                epsilons = model.epsilon
                x_deltas = torch.cat([
                    (x_batch - epsilons).view(1, -1, x_batch.size(1)),
                    (x_batch + epsilons).view(1, -1, x_batch.size(1))
                ], dim=0)
                breeze_list = []
                y = model.forward(x_batch, breeze_list)
                x_deltas = model.breeze_forward(x_deltas, breeze_list)
                du_dx = (x_deltas[1] - x_deltas[0]) / (2 * epsilons)
                du_dx = torch.abs(du_dx * model.dim_mask + 1 - model.dim_mask).clamp(min=0.001)
                per_sample_lp = torch.sum(torch.log(du_dx), dim=1)  # (batch,)
            
            log_probs.append(per_sample_lp)
    
    all_log_probs = torch.cat(log_probs, dim=0)
    threshold = torch.quantile(all_log_probs, percentile / 100.0).item()
    
    print(f"Log prob stats: min={all_log_probs.min():.2f}, "
          f"p10={threshold:.2f}, median={all_log_probs.median():.2f}, "
          f"max={all_log_probs.max():.2f}")
    
    return threshold


def compute_per_sample_log_prob(model, x):
    """Compute per-sample log p(x) for filtering. Returns (N,) tensor."""
    if hasattr(model, 'n_components'):  # MultiBF
        log_pi = model.get_mixture_log_weights()
        comp_lds = [log_pi[k] + model._per_sample_log_det(bf, x)
                    for k, bf in enumerate(model.components)]
        return torch.logsumexp(torch.stack(comp_lds, dim=0), dim=0)
    else:  # BreezeForest
        epsilons = model.epsilon
        x_deltas = torch.cat([
            (x - epsilons).view(1, -1, x.size(1)),
            (x + epsilons).view(1, -1, x.size(1))
        ], dim=0)
        breeze_list = []
        model.forward(x, breeze_list)
        x_deltas = model.breeze_forward(x_deltas, breeze_list)
        du_dx = (x_deltas[1] - x_deltas[0]) / (2 * epsilons)
        du_dx = torch.abs(du_dx * model.dim_mask + 1 - model.dim_mask).clamp(min=0.001)
        return torch.sum(torch.log(du_dx), dim=1)
```

### 步骤 2：Generate with Rejection Sampling

```python
def generate_with_rejection(model, n_samples, threshold, dim=2, 
                             oversample_factor=3, max_rounds=10):
    """
    Generate samples from model, filtering out low log-prob samples.
    
    Works for BreezeForest (direct inverse_map) and MultiBF.
    
    :param model: BreezeForest or MultiBF
    :param n_samples: target number of samples
    :param threshold: log prob threshold from calibrate_log_prob_threshold()
    :param oversample_factor: how many extra samples to generate per round
    :return: accepted samples (n_samples, dim)
    """
    model.eval()
    accepted = []
    total_generated = 0
    
    for round_i in range(max_rounds):
        n_generate = (n_samples - len(accepted)) * oversample_factor
        
        with torch.no_grad():
            if hasattr(model, 'n_components'):  # MultiBF
                z_candidates = model.inverse_map(n_generate)
            else:  # BreezeForest
                z = torch.rand(n_generate, dim) * 0.98 + 0.01
                z_candidates = model.inverse_map(z)
            
            total_generated += n_generate
            
            # Compute per-sample log prob and filter
            log_probs = compute_per_sample_log_prob(model, z_candidates)
            accept_mask = log_probs > threshold
            accepted_batch = z_candidates[accept_mask]
            accepted.append(accepted_batch)
        
        current_count = sum(s.shape[0] for s in accepted)
        acceptance_rate = current_count / total_generated
        print(f"Round {round_i+1}: generated {n_generate}, "
              f"accepted {accepted_batch.shape[0]}, "
              f"total accepted {current_count}/{n_samples}, "
              f"acceptance rate: {acceptance_rate:.1%}")
        
        if current_count >= n_samples:
            break
        
        # Adaptive threshold: if acceptance rate is very low, lower threshold
        if acceptance_rate < 0.1 and round_i > 1:
            threshold *= 1.05  # slightly lower (threshold is log scale, so +0.05 → less strict)
            print(f"  Low acceptance rate, adjusting threshold to {threshold:.2f}")
    
    all_accepted = torch.cat(accepted, dim=0)[:n_samples]
    return all_accepted
```

### 步骤 3：在 demo_multi_bf.py 中集成

```python
# 训练完成后：
# 1. 校准阈值（使用训练数据的第 10 百分位）
all_batch = (x_all - mean) / std
with torch.no_grad():
    threshold = calibrate_log_prob_threshold(mbf, all_batch, percentile=10.0)

# 2. 带拒绝采样的生成
mbf.eval()
with torch.no_grad():
    raw_samples = generate_with_rejection(
        mbf, n_samples=data_size, threshold=threshold, 
        dim=2, oversample_factor=5
    )
    samples = raw_samples * std + mean
    samples = samples.numpy()
```

### 阈值选择指南

| 百分位数 | 效果 | 适用场景 |
|---------|------|---------|
| 5% | 宽松，过滤少 | 模型质量较好时，轻微修复 |
| 10% | 推荐默认 | 一般情况 |
| 20% | 较严格 | 模型 inter-cluster 问题严重时 |
| 30% | 严格，拒绝率高 | 仅在与 TAHEM 配合且模型已专一化后使用 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **高拒绝率** | 如果模型质量差，大部分样本都被拒绝，生成效率低下 | 降低阈值百分位数（如 5%）；或先用 TAHEM 改善模型质量 |
| **cluster 边缘被过度过滤** | 高阈值可能过滤掉 cluster 边缘的合法稀疏点 | 使用 5-10% 百分位数（不要过高） |
| **不改变模型本身** | LPTRS 不修复模型内部问题，只是过滤输出 | 与 TAHEM 组合：TAHEM 修复训练，LPTRS 作为最后一层防线 |
| **计算开销** | 每个候选样本需要额外的 forward pass（log p 计算） | 可以批量化计算 log p，开销 ≈ 1 次额外 forward pass |
| **阈值不适配新分布** | 如果使用模型前需要重新归一化（不同 std/mean），阈值需要重新校准 | 始终在归一化空间中校准和使用 |

---

## 推荐优先级

**⭐⭐ 高优先级（最简单的修复，可作为任何方案的最后一层防线）**

理由：
1. **零重训练成本**：对任何已训练的 BreezeForest 或 MultiBF 都能即时应用。
2. **极简实现**：核心逻辑约 10 行代码（生成 + 过滤 + 循环）。
3. **有理论支撑**：Optimal Budgeted Rejection Sampling（OBRS, Verine 2024）提供了 rejection sampling 在生成模型上的最优化理论；Stimper et al. (2022) 证明 rejection sampling base distribution 对 topology mismatch 有效。
4. **与 TAHEM + ELKS 完全正交**：可以叠加：TAHEM 训练出好模型，ELKS 减少 inter-cluster latent 采样，LPTRS 过滤最终漏出的低密度样本。
5. **好的调试工具**：运行 LPTRS 并记录拒绝率，可以作为模型质量的量化指标——拒绝率越低，模型越好。

**使用场景定位**：
- **单独使用 LPTRS**：最快速验证效果，评估 inter-cluster 问题严重程度（拒绝率 > 50% 说明问题严重）
- **LPTRS + ELKS**：inference-time 双重修复（无需重训练）
- **TAHEM + ELKS + LPTRS**：完整 pipeline（训练修复 + 采样修复 + 最终过滤）

---

## 参考文献

- Verine, A. et al. (2024). "Optimal Budgeted Rejection Sampling for Generative Models." *AISTATS 2024*. https://proceedings.mlr.press/v238/verine24a.html （在有限预算下最优 rejection sampling for generative models）
- Na, D. et al. (2024). "Diffusion Rejection Sampling." *ICML 2024*. https://proceedings.mlr.press/v235/na24a.html （将 rejection sampling 应用于扩散模型的类似工作）
- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a.html （在 latent space 应用 rejection sampling 解决 topology mismatch）
- Grover, A. et al. (2019). "Bias Correction of Learned Generative Models using Likelihood-Free Importance Weighting." *NeurIPS 2019*. （用 importance weighting 修正生成模型的密度误差，与 LPTRS 同源）
