# Idea: Self-Density Rejection Sampling (SDRS) — Post-Generation Density Filtering

**创建时间**: 2026-03-12 03:10 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（零成本、立即可用，填补单 BF 推理修复的空白）

---

## 问题定义

MultiBF 的所有现有采样改进方案（LZR、Latent GMM Resampling）都依赖于**组件专一化**——即需要训练后各组件清晰对应各自的 cluster。这有两个局限：

1. **单 BreezeForest 无解**：现有所有推理阶段 idea（LZR、Latent GMM）仅针对 MultiBF，对**单 BreezeForest** 在多 cluster 数据上的 inter-cluster 生成问题没有任何修复方案。
2. **组件专一化程度依赖**：若训练后组件仍然混淆（例如 DAEM 退火不充分），Latent GMM 拟合的是一个混乱的多峰分布，修复效果有限。

**关键洞察**：任何经过充分训练的 flow-based 模型（无论是单 BF 还是 MultiBF），在训练数据的高密度区域（cluster 内部）会分配高 log|det J|，在低密度区域（cluster 之间）会分配低 log|det J|。**这个密度差异本身就是过滤 inter-cluster 样本的天然信号。**

---

## 从代码与已有 Idea 中得到的背景判断

**代码分析**（`BreezeForest.train_forward()`，`MultiBF.train_forward()`，`BreezeForest.inverse_map()`）：

- BreezeForest 的训练 loss 是 `-log_det`，等价于最大化 `log|det J(x)|` on training data
- 对于训练数据中 cluster k 的点：`log|det J(x)|` 高（因为 CDF 在 cluster 处变化快，Jacobian 大）
- 对于 inter-cluster 区域的点：`log|det J(x)|` 低（CDF 在这里变化缓慢，Jacobian 小）

**因此**：如果我们用训练好的模型对生成样本计算 `log p(x)`（对单 BF 就是 `log|det J(x)|`），然后过滤掉低 density 样本，就能自然去除 inter-cluster 点。

**与现有推理 idea 的比较**：
- **LZR（2026-03-11 12:35）**：矩形 box 限制 latent 采样 → 只对 MultiBF 有效，且 box 估计不准确时效果差
- **Latent GMM Resampling（2026-03-12 01:51）**：GMM 限制 latent 采样 → 只对 MultiBF 有效，且需要良好的组件专一化才能拟合出好的 GMM

**SDRS 的独特优势**：
- 对**单 BF** 有效（现有所有推理 idea 均不支持单 BF）
- 不需要组件专一化（即使 soft-EM 训练后效果也比较好）
- 不需要任何额外拟合（不需要 GMM fitting）
- 可以与 Latent GMM Resampling **叠加使用**（Latent GMM 过滤 + SDRS 二次过滤）

**外部研究验证**：
- **Importance Corrected Neural JKO (Arbel et al., 2024, arxiv 2407.20444)**：在 continuous normalizing flow 中交替做 flow steps 和 rejection-resampling，用 importance weights 来接受/拒绝样本，解决多峰分布的采样问题。本 Idea 是其在已训练离散 BreezeForest 上的轻量化变体（不需要重训练）。
- **Stimper et al. (2022, AISTATS)**：通过学习 base distribution 的 rejection sampling 来修复 topology mismatch；本 Idea 在数据空间直接 rejection，不需要修改 base distribution 或重训练。
- **DiverseFlow（2025, arxiv 2504.07894）**：使用 DPP 提升流模型采样多样性；与本 Idea 正交但都属于推理时改进策略。

---

## 核心思路

**过采样 + 密度筛选（Oversample-then-Filter）**：

1. **过采样**：从训练好的模型生成 n_samples × oversample_ratio 个样本
2. **密度评估**：对每个生成样本 x，用训练好的模型计算 log p(x)
   - 对单 BF：`log p(x) = log|det J(x)|` （利用 `train_forward` 计算，但不做 backward）
   - 对 MultiBF：`log p(x) = logsumexp_k(log π_k + log|det J_k(x)|)`
3. **阈值筛选**：保留 `log p(x) ≥ threshold` 的样本
   - 阈值设定方式：取生成样本 log p(x) 的第 `density_percentile` 百分位数
4. **返回**：从筛选后的样本中随机取 n_samples 个返回

**核心参数**：
- `oversample_ratio`：过采样倍数（建议 3–10）
- `density_percentile`：筛选百分位数（建议 20–40）

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**密度视角的因果链**：

1. 训练好的 flow 模型的 Jacobian 在训练数据密集处（cluster 内部）大，在数据稀疏处（cluster 之间）小
2. `log|det J(x)|` 直接反映每个点处的模型密度（log p(x) = log|det J(x)| + const）
3. inter-cluster 点的 log p(x) 低（模型"不认识"这个区域）
4. 通过过滤掉低 density 样本，精确去除 inter-cluster 生成点

**理论保证**：
- 筛选后的样本集的经验分布近似于：p_filtered(x) ∝ p(x) * I[log p(x) ≥ threshold]
- 这等价于对 p(x) 做截尾（truncation），去除低密度区域
- 截尾分布在高密度区域（cluster 内部）与 p(x) 近似一致，在低密度区域（cluster 之间）密度为零

**与 Latent GMM Resampling 的理论区别**：
- Latent GMM：在 z 空间截尾（用 GMM 建模 z_k 的真实分布，跳过 z 的低密度区域）
- SDRS：在 x 空间截尾（直接用 log p(x) 过滤，不依赖 latent 空间结构）
- 两者互补：可先用 Latent GMM 减少无效 z 采样，再用 SDRS 二次过滤漏网的 inter-cluster 点

---

## 与历史 idea 的关系

| 历史 Idea | 关系 | 说明 |
|-----------|------|------|
| **LZR（2026-03-11 12:35）** | **补充（不替代）** | LZR 已被 Latent GMM 替代；SDRS 是独立的推理路径（data space filtering），与 latent space filtering 不冲突 |
| **Latent GMM Resampling（2026-03-12 01:51）** | **互补，可叠加** | Latent GMM 在 z 空间过滤；SDRS 在 x 空间二次过滤。两者叠加效果最好。SDRS 还填补了 Latent GMM 不支持单 BF 的空白 |
| **DAEM（2026-03-12 01:51）** | 效果叠加 | 训练越专一 → density contrast 越大 → SDRS 过滤效果越好；但 SDRS 即使在 soft-EM 训练后也有一定效果 |
| **K-Means Pre-Init / TAPC（本轮新增）** | 同上 | 更好的初始化 → 更好的训练 → 更清晰的密度分布 → SDRS 过滤更准确 |
| **ICDR（2026-03-11 12:40）** | 互补 | ICDR 在训练时降低组件交叉密度；SDRS 在推理时过滤低密度点。两者正交 |

**本 Idea 的替代关系**：不替代任何现有 idea；填补了**单 BF 场景的推理修复空白**，这是所有现有 idea 都没有覆盖的场景。

---

## 具体实现建议

### 步骤 1：为 BreezeForest 添加 `compute_log_density()` 方法

```python
# 在 BreezeForest 中添加
def compute_log_density(self, x):
    """
    Compute log p(x) = log|det J(x)| for each sample (up to an additive constant).
    Used for post-generation density filtering.
    
    :param x: input tensor (batch_size, dim)
    :return: per-sample log density (batch_size,)
    """
    epsilons = self.epsilon
    x_deltas = torch.cat([
        (x - epsilons).view(1, -1, x.size(1)),
        (x + epsilons).view(1, -1, x.size(1))
    ], dim=0)
    
    breeze_list = []
    self.forward(x, breeze_list)
    x_deltas_out = self.breeze_forward(x_deltas, breeze_list)
    
    du_dx = (x_deltas_out[1] - x_deltas_out[0]) / (2 * epsilons)
    du_dx = torch.abs(du_dx * self.dim_mask + 1 - self.dim_mask).clamp(min=0.001)
    
    # Sum over dimensions -> per-sample log density
    return torch.sum(torch.log(du_dx), dim=1)  # (batch_size,)
```

### 步骤 2：为 BreezeForest 添加 `inverse_map_with_density_filter()` 方法

```python
def inverse_map_with_density_filter(
    self,
    z,
    density_percentile=25,
    oversample_ratio=4,
    max_gap=1e-3,
    decay_ratio=1.0
):
    """
    Generate samples using density-filtered inverse mapping.
    
    :param z: latent codes (n_samples, dim) - the DESIRED number of output samples
    :param density_percentile: keep samples above this log-density percentile
    :param oversample_ratio: multiply n_samples by this for initial generation
    :return: filtered samples, approximately n_samples in count (may be fewer)
    """
    n_samples = z.size(0)
    n_gen = n_samples * oversample_ratio
    
    # Generate more latent codes
    z_over = torch.rand(n_gen, self.dim) * 0.98 + 0.01
    
    # Generate all samples
    with torch.no_grad():
        x_all = self.inverse_map(z_over, max_gap=max_gap, decay_ratio=decay_ratio)
        
        # Compute log density for each generated sample
        self.batch_example = x_all  # needed for bisection distribution
        log_dens = self.compute_log_density(x_all)  # (n_gen,)
        
        # Filter: keep top (100 - density_percentile)% samples
        threshold = torch.quantile(log_dens, density_percentile / 100.0)
        keep_mask = log_dens >= threshold
        x_filtered = x_all[keep_mask]
        
        # Return up to n_samples
        if len(x_filtered) >= n_samples:
            idx = torch.randperm(len(x_filtered))[:n_samples]
            return x_filtered[idx]
        else:
            # Fallback: return all filtered + some unfiltered to reach n_samples
            n_extra = n_samples - len(x_filtered)
            x_extra = x_all[~keep_mask][:n_extra]
            return torch.cat([x_filtered, x_extra], dim=0)
```

### 步骤 3：为 MultiBF 添加密度过滤采样

```python
# 在 MultiBF 中添加
def inverse_map_with_density_filter(
    self,
    n_samples,
    density_percentile=25,
    oversample_ratio=4,
    max_gap=1e-3,
    decay_ratio=1.0
):
    """
    Generate samples from mixture with post-generation density filtering.
    Works even without component specialization (complements Latent GMM).
    """
    n_gen = n_samples * oversample_ratio
    
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_gen, replacement=True)
    
    x_all = torch.zeros(n_gen, self.dim)
    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        z = torch.rand(n_k, self.dim) * 0.98 + 0.01
        x_k = self.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
        x_all[mask] = x_k
    
    # Compute mixture log density for all generated samples
    with torch.no_grad():
        log_pi = self.get_mixture_log_weights()  # (K,)
        component_log_probs = []
        for k, bf in enumerate(self.components):
            bf.batch_example = x_all
            ld = self._per_sample_log_det(bf, x_all)  # (n_gen,)
            component_log_probs.append(log_pi[k] + ld)
        stacked = torch.stack(component_log_probs, dim=0)  # (K, n_gen)
        log_prob = torch.logsumexp(stacked, dim=0)  # (n_gen,)
        
        threshold = torch.quantile(log_prob, density_percentile / 100.0)
        keep_mask = log_prob >= threshold
        x_filtered = x_all[keep_mask]
    
    if len(x_filtered) >= n_samples:
        idx = torch.randperm(len(x_filtered))[:n_samples]
        return x_filtered[idx]
    else:
        n_extra = n_samples - len(x_filtered)
        x_extra = x_all[~keep_mask][:n_extra]
        return torch.cat([x_filtered, x_extra], dim=0)
```

### 步骤 4：在 demo 中使用

```python
# 对单 BF (demo_functions.py 中的 generate_sample):
with torch.no_grad():
    z = torch.rand(sample_size * 4, 2) * 0.98 + 0.01
    generated = model.inverse_map_with_density_filter(
        z, density_percentile=25, oversample_ratio=4
    )

# 对 MultiBF (demo_multi_bf.py):
with torch.no_grad():
    samples = mbf.inverse_map_with_density_filter(
        n_samples=data_size,
        density_percentile=25,
        oversample_ratio=4
    )
```

### 超参数调优建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `density_percentile` | 20 – 35 | 越高过滤越严格；从 25 开始，根据可视化效果调整 |
| `oversample_ratio` | 3 – 8 | 越高过滤质量越好但速度越慢；建议 4–5 |
| 使用时机 | 任何时候 | 无需重训练，可立即用于现有模型 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **计算开销 × oversample_ratio** | 需要生成 3–8 倍的样本，每个样本都要运行 bisection + density eval | 对 MultiBF 用 `_per_sample_log_det` 复用 forward pass；减小 `oversample_ratio`（3 倍通常足够） |
| **密度平坦化问题** | 若 soft-EM 训练后模型密度在 inter-cluster 区域不够低，threshold 无法区分 | 与 DAEM/K-Means init 结合使用；或降低 `density_percentile` 到 15% |
| **样本数量不保证** | 过滤后可能样本数 < n_samples（若过滤率高） | 代码已有 fallback 机制（不足时取部分未过滤样本补充） |
| **density 计算的有限差分误差** | `compute_log_density` 用有限差分近似 Jacobian，与精确值有误差 | 误差是系统性的（所有样本都有相同的近似误差），不影响相对排序；如需精确，用 `train_forward_exact` |
| **单 BF 上的 inter-cluster 路径** | 单 BF 必须"拉伸"映射通过 inter-cluster 区域；这些区域的 density 不一定是 0，仅仅是低 | 这个问题单靠 SDRS 不能完全消除（但可以显著改善）；推荐与 TAPC/DAEM 结合 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（填补单 BF 推理修复空白，零成本实施）**

理由：
1. **填补现有 idea 的唯一空白**：现有所有推理阶段 idea（LZR、Latent GMM）仅支持 MultiBF；SDRS 是唯一支持**单 BreezeForest** 的推理修复方案
2. **零成本实施**：不需要重训练，不需要 GMM fitting，只需在现有 inverse_map 外加一层过滤
3. **理论严格**：基于 flow 模型的 density estimate，在 cluster 内部 log p(x) 高、inter-cluster 处低，这是 flow 训练目标的直接数学结论
4. **可叠加**：可与 Latent GMM Resampling 叠加（Latent GMM 减少无效 z 采样，SDRS 在 x 空间二次过滤）
5. **有外部文献支撑**：Importance Corrected Neural JKO（Arbel 2024）验证了在 flow 采样时使用 importance/density weight 进行 rejection sampling 的有效性

---

## 参考文献

- Arbel, M. et al. (2024). "Importance Corrected Neural JKO Sampling." *arxiv 2407.20444*. https://arxiv.org/abs/2407.20444  
  ← 直接支撑：在 continuous normalizing flow 中交替 flow steps 和 rejection-resampling；本 Idea 是其在已训练 BF 上的离散轻量化版本
- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a/stimper22a.pdf  
  ← 同样思路的 latent space 版本；本 Idea 的 data-space 版本
- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 113*.  
  ← Latent space guided sampling 同一思路；本 Idea 在 data space 实现类似约束
- Müller, T. et al. (2019). "Neural Importance Sampling." *SIGGRAPH Asia 2019*.  
  ← 用 normalizing flow 的 density 做 importance sampling 的经典应用
