# Idea: Empirical Latent Resampling — 用训练数据的潜空间经验分布替代均匀采样

**创建时间**: 2026-03-11 12:42  
**优先级**: ★★★★☆（次高）  
**分类**: 采样策略改造（不需修改训练过程）

---

## 一、问题定义

BreezeForest 的 `MultiBF.inverse_map` 当前生成过程为：

```python
z = torch.rand(n_k, self.dim) * 0.98 + 0.01   # z ~ Uniform(0.01, 0.99)^d
x_k = self.components[k].inverse_map(z)
```

**根本原因（Root Cause 3：基础分布采样无区分性）**：

每个组件 k 对潜空间 z ∈ [0,1]^d 的整个范围一视同仁地采样。然而，并非所有 z 都对应真实数据所在区域。

具体而言：
- 训练数据 x 通过正向传播 BF_k(x) → z ∈ [0,1]^d，得到的 z 值并**非均匀分布**于 [0,1]^d
- 真实 cluster 对应的 z 值聚集在 [0,1]^d 的某些子区域（"活跃区域"）
- 训练数据 cluster 之间的空白区域对应的 z 值在训练时从未出现
- 但 `torch.rand` 会均匀采样整个 [0,1]^d，包括那些"从未见过训练样本"的 z 区域
- 这些"冷门" z 值逆映射回数据空间时，恰好落在 cluster 间的空白区域

**关键洞察**：流模型保证了 BF_k 是双射，因此 [0,1]^d 中的每个 z 必然对应某个 x。但训练只告诉流"z_active → x_cluster"是高密度的，并未告诉流"z_inactive → x_void"是低密度的——这种非对称性让均匀采样 z 产生幽灵点。

---

## 二、核心思路

**用训练数据的实际潜空间分布替代均匀分布进行采样。**

具体步骤：
1. **收集阶段**（训练结束后或训练中动态维护）：对每个训练样本 x_i 和分配到的组件 k，计算其潜空间表示 z_i = BF_k(x_i)，收集为经验集合 Z_k = {z_i : k*(x_i) = k}
2. **近似建模**：对 Z_k 拟合一个轻量级的核密度估计（KDE）或直方图，作为组件 k 的采样分布
3. **替换采样**：生成时，不再从 Uniform(0.01, 0.99)^d 采样，而是从 Z_k 中随机抽取一个 z（加小噪声 ε 以获得多样性），再通过 inverse_map 映射到数据空间

最简实现（"带扰动的经验回放"）：
```
z ~ Sample_from_Z_k + Normal(0, σ²I)，其中 σ ≈ 0.02
```

这是 Stimper et al. (2022) "Resampling Base Distributions of Normalizing Flows" 的轻量近似版本，适合在 BreezeForest 架构内低成本实现。

---

## 三、为什么适合解决 multi-cluster 中间点问题

| 问题根源 | 本方案的针对方式 |
|---|---|
| Uniform 采样覆盖"无训练数据"的 z 区域 | 仅从 Z_k（有训练样本的 z 区域）附近采样 |
| cluster 间空白在 z 空间的对应区域被均匀采到 | Z_k 中不包含这些区域，自然排除 |
| 不需要重新训练 | 只改变生成时的采样步骤 |
| 对任何 BreezeForest 已训练的模型可用 | post-hoc 方案，零训练代价 |

**直观图示**：
```
z 空间 [0,1]^d:
  实际分布 Z_k: ███         ███         ███   (三个 cluster 对应的 z 聚类)
  Uniform 采:   ████████████████████████████   (全覆盖，包括空白区)
  本方案采:     ███         ███         ███   (只覆盖活跃区)
```

---

## 四、与历史 Idea 的关系

- **补充 Idea 1（Hard-EM）**：Idea 1 解决训练专化问题；本方案解决**生成采样**问题。两者针对不同根因，可叠加使用：
  - 先用 Hard-EM 训练使组件专化
  - 再用本方案的经验潜空间采样生成高质量样本
- **与 Stimper et al. (2022) 的关系**：本方案是其简化版——Stimper 的完整方案训练一个显式的能量模型作为 rejection sampler，而本方案直接使用经验分布，无需额外训练。
- **与现有 `inverse_map` 的关系**：仅扩展 `MultiBF.inverse_map`，不改变核心架构和 `BreezeForest.inverse_map`。
- **现有 notes 中无对应内容**：`bf_vs_bnaf_2026_02_10.md` 未提及基础分布采样改进；`autoregressive_normalizing_flows_2026_02_10.md` 的 UMNN 和 SOS 讨论与此方向正交。

---

## 五、具体实现建议

### 5.1 收集潜空间编码

在 `MultiBF` 类中添加：

```python
def collect_latent_codes(self, data_loader, mean, std):
    """
    Pass all training data through each component and collect latent codes.
    Stores self.latent_codes: list of (n_k, dim) tensors, one per component.
    """
    self.latent_codes = [[] for _ in range(self.n_components)]

    self.eval()
    with torch.no_grad():
        for batch, _ in data_loader:
            batch = (batch - mean) / std

            # Hard assignment: find best component per sample
            log_pi = self.get_mixture_log_weights()
            comp_lp = []
            for k, bf in enumerate(self.components):
                ld = self._per_sample_log_det(bf, batch)   # (batch_size,)
                comp_lp.append(log_pi[k] + ld)
            assignments = torch.argmax(torch.stack(comp_lp, dim=0), dim=0)

            # Collect z for each component
            for k, bf in enumerate(self.components):
                mask = (assignments == k)
                if mask.sum() == 0:
                    continue
                x_k = batch[mask]
                breeze_list = []
                z_k = bf.forward(x_k, breeze_list)          # (n_k, dim), in [0,1]^d
                self.latent_codes[k].append(z_k.detach())

    # Concatenate
    for k in range(self.n_components):
        if self.latent_codes[k]:
            self.latent_codes[k] = torch.cat(self.latent_codes[k], dim=0)
        else:
            self.latent_codes[k] = None
    self.train()
```

### 5.2 替换采样方法

```python
def inverse_map_empirical(self, n_samples, noise_std=0.02, max_gap=1e-3, decay_ratio=1.0):
    """
    Generate samples using empirical latent resampling instead of Uniform.
    Requires self.latent_codes to be populated via collect_latent_codes().
    
    :param noise_std: Gaussian noise added to empirical z (diversity control)
                      0 = pure memorization, 0.05+ = more generalization
    """
    assert hasattr(self, 'latent_codes'), \
        "Call collect_latent_codes() before using empirical sampling."

    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)

    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue

        if self.latent_codes[k] is not None:
            # Sample from empirical distribution with perturbation
            n_codes = self.latent_codes[k].shape[0]
            idx = torch.randint(0, n_codes, (n_k,))
            z = self.latent_codes[k][idx] + torch.randn(n_k, self.dim) * noise_std
            z = z.clamp(0.01, 0.99)  # keep in valid range for sigmoid output
        else:
            # Fallback to uniform if no codes collected for this component
            z = torch.rand(n_k, self.dim) * 0.98 + 0.01

        x_k = self.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
        results[mask] = x_k

    return results
```

### 5.3 带拒绝采样的增强版（可选）

若需要更严格的保证，可在 5.2 基础上加轻量拒绝采样：

```python
def inverse_map_with_rejection(self, n_samples, density_threshold_quantile=0.05, ...):
    """
    Generate and reject samples below a density threshold.
    Threshold determined from training data density distribution.
    """
    # Step 1: compute density threshold from training data
    with torch.no_grad():
        train_log_probs = []
        for batch in data_loader:
            lp = self.train_forward(batch)
            train_log_probs.append(lp)
        threshold = torch.quantile(torch.stack(train_log_probs), density_threshold_quantile)

    # Step 2: generate and filter
    accepted = []
    while len(accepted) < n_samples:
        candidates = self.inverse_map(n_samples * 2)    # oversample
        lps = self._log_prob(candidates)                # eval density
        accepted.extend(candidates[lps > threshold])
    return torch.stack(accepted[:n_samples])
```

### 5.4 超参数建议

| 超参数 | 推荐范围 | 说明 |
|---|---|---|
| `noise_std` | 0.01–0.05 | 太小=记忆训练集；太大≈均匀采样 |
| `max_gap` | 1e-3（现有值） | 不变 |
| 数据量要求 | ≥100 样本/cluster | KDE 需要足够样本密度 |
| 可选：KDE bandwidth | Scott's rule（sklearn） | 若使用完整 KDE 而非扰动采样 |

---

## 六、潜在风险 / 副作用

| 风险 | 严重性 | 缓解措施 |
|---|---|---|
| 经验 z 分布过拟合训练集（生成新奇样本能力下降） | 中 | 调大 `noise_std`；视任务需求平衡 |
| 初始化后 latent_codes 需重新收集（若模型更新） | 低 | 在训练完成后统一收集一次；或在训练中定期更新（每 K 步更新一次） |
| 某组件无训练样本分配（dead component） | 低 | fallback 到 Uniform 采样，不影响其他组件 |
| 对 non-smooth 分布（如 2spirals）效果可能受限 | 中 | 可增大 `noise_std` 或切换为 KDE 方案 |
| 内存占用（存储全部训练样本的 z 值） | 低 | 对于 2D/小维度数据可忽略；高维可用 coreset |

---

## 七、与 Idea 1 的协同使用

最佳实践（推荐顺序）：
1. 用 **Idea 1（Hard-EM）** 训练模型，使组件专化
2. 训练完成后，调用 `collect_latent_codes()` 收集各组件的 z 分布
3. 用 `inverse_map_empirical()` 生成样本，进一步过滤空白区点

两步叠加后，空白区幽灵点的数量预计可降至接近零。

---

## 八、推荐优先级

**★★★★☆（高优先级，次于 Idea 1）**

理由：
1. **零训练代价**：不改变任何训练过程，可直接对已训练模型应用
2. 从第一原理出发：训练数据的 z 分布就是"正确"的采样分布
3. 实现极简：核心代码约 30 行
4. 有成熟理论支撑：Stimper et al. (2022) 在标准 benchmark 上验证
5. 风险可控：`noise_std` 连续调节，可平滑过渡

---

## 九、参考文献

1. Stimper, V., Schölkopf, B., & Hernández-Lobato, J.M. (2022). "Resampling Base Distributions of Normalizing Flows." AISTATS 2022. https://proceedings.mlr.press/v151/stimper22a.html
2. GitHub: https://github.com/VincentStimper/resampled-base-flows
3. Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." Machine Learning. arXiv:2305.12149
4. Optimal Budgeted Rejection Sampling (OBRS) (2024). AISTATS 2024. https://proceedings.mlr.press/v238/verine24a.html
