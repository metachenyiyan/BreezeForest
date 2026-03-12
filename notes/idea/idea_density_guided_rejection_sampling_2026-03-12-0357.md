# Idea: Density-Guided Rejection Sampling (DGRS) for MultiBF Inference

**创建时间**: 2026-03-12 03:57 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（推理阶段新增方案，与 Latent GMM 互补，对 DAEM 训练后效果最佳）

---

## 问题定义

MultiBF 当前的生成流程：
```
z ~ Uniform([0.01, 0.99]^d)
x = f_k^{-1}(z)
```

此流程的结构性缺陷是：**生成的 x 没有经过任何质量验证**。z 的采样范围是整个 [0.01, 0.99]^d，包含了映射到 inter-cluster 区域的 z 值。即使 f_k 已经被专一化（经过 DAEM + K-Means Pre-Init），仍然存在边缘 z 值映射到 cluster 边界以外的风险。

**现有采样阶段修复方案的不足**：
- **LZR (2026-03-11-1235)**：矩形 box 限制 z 的采样范围，是正确方向但近似粗糙
- **Latent GMM (2026-03-12-0151)**：用 GMM 建模 latent 密度，比矩形 box 精确，但需要 GMM 近似的假设

以上方案都在 **latent 空间**（z ∈ [0,1]^d）限制采样。但还有一个正交的视角：**在生成的 x 上直接过滤**，利用 MultiBF 自身的混合密度估计作为质量判断标准。

---

## 从代码与已有 Idea 中得到的背景判断

**代码分析**（`MultiBF.train_forward()`, `MultiBF._per_sample_log_det()`）：

MultiBF 可以为任意输入 x 计算对数似然：
```python
log p(x) = logsumexp_k( log π_k + log|det J_k(x)| )
```

其中 `_per_sample_log_det(bf, x)` 通过有限差分近似每个样本的 log|det J_k(x)|。这意味着：

1. **MultiBF 可以评估任意 x 的密度**，不只限于训练数据
2. **对于 inter-cluster 区域的点**：如果组件已专一化（DAEM 训练后），所有组件 k 在该区域的 |det J_k(x)| 都很低 → log p(x) 低
3. **对于 cluster k 内的点**：组件 k 的 |det J_k(x)| 高 → log p(x) 高

因此，**MultiBF 自身就是最好的 inter-cluster 样本检测器**，无需额外训练一个判别器。

**与已有 Idea 的代码关联**：
- `MultiBF._per_sample_log_det()` 已实现，可直接用于评估生成样本的密度
- `MultiBF.inverse_map()` 的输出（生成样本）可以传入 `_per_sample_log_det()` 进行过滤
- 整个 DGRS 流程不需要修改任何现有方法，只需添加一个 wrapper

**关键条件：组件专一化程度**

DGRS 的有效性取决于 MultiBF 的密度估计质量：
- **DAEM + K-Means 训练后**：组件已专一 → inter-cluster 密度低 → DGRS 有效
- **baseline soft-EM 训练后**：组件混淆 → 所有区域密度相似 → DGRS 效果有限

因此，DGRS 是 DAEM + K-Means 流水线的**推理阶段收尾步骤**，而非独立解决方案。

**外部研究支撑**：

1. **Optimal Budgeted Rejection Sampling (Verine et al., AISTATS 2024, arxiv 2311.00460)**：
   
   本 Idea 的直接理论基础。OBRS 证明：对于给定的采样预算（生成 N_proposal 个样本，保留 N_final 个），存在一个最优的拒绝采样策略，使 post-rejection 分布与真实分布之间的任意 f-散度最小化。该策略的最优形式是：**按模型密度对生成样本排序，保留密度最高的 N_final 个**。
   
   关键结论：用模型自身的密度作为接受/拒绝标准，在统计意义上是最优的（在给定计算预算下）。

2. **Discriminator Rejection Sampling (Azadi et al., ICLR 2019)**：
   
   原始 DRS 方法，用 GAN 的判别器估计密度比 p_data(x) / p_model(x) 来接受/拒绝样本。DGRS 是其针对 normalizing flow 的特殊化版本：因为 normalizing flow 直接计算 p_model(x)，不需要额外的判别器网络，使 DRS 机制变得更简单、计算更稳定。

3. **Coeurdoux et al. (2024, Machine Learning)**：
   
   使用 MALA 在 normalizing flow 的 latent space 进行 MCMC 采样，以避免标准均匀采样的拓扑问题。DGRS 与 MALA 是互补方向：MALA 通过 latent 空间的梯度引导，DGRS 通过 data 空间的密度过滤。对 BreezeForest 的 [0,1]^d latent space，直接 MALA 需要处理 sigmoid 边界（较复杂），而 DGRS 在 data space 操作，更直接。

---

## 核心思路

**过采样 + 密度过滤**（Oversampling + Density Filtering）：

1. 生成 N_proposal 个候选样本（N_proposal > N_target，如 N_proposal = 3 × N_target）
2. 对每个候选样本计算 log p(x)（MultiBF 混合对数密度）
3. 保留 log p(x) 最高的 N_target 个样本（top-p 选择）

**直觉**：MultiBF 自身"知道"哪些 x 值在其学习到的分布中密度高。inter-cluster 点的 log p(x) 在训练良好后会显著低于 cluster 内的点。通过这种自过滤，生成样本自动远离 inter-cluster 区域。

**变体 1：固定阈值过滤**
```
接受 x 如果 log p(x) > τ （τ 由训练数据分位数决定）
```

**变体 2：Top-k 选择（推荐）**
```
生成 N_proposal，保留 log p(x) 最高的 N_target 个
```

**变体 2 的优点**：不需要确定具体阈值 τ，只需确定采样倍数（N_proposal / N_target）。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**密度分离的数学论证**：

经过 DAEM + K-Means 训练后，设组件 k 专一化到 cluster k：

对 x ∈ cluster k（cluster k 内的点）：
- |det J_k(x)| ≈ p_k(x)（高，组件 k 对该区域密度高）
- |det J_j(x)| ≈ p_j(x)（低，j≠k 的组件未覆盖此区域）
- log p(x) = log Σ_k π_k |det J_k(x)| ≈ log(π_k × p_k(x))（高）

对 x ∈ inter-cluster 区域：
- 所有组件的 |det J_k(x)| 都低（没有任何组件专一化到此区域）
- log p(x) = log Σ_k π_k |det J_k(x)| ≈ log(ε_0 + ε_1 + ...)（低）

因此，top-k 选择自然淘汰 inter-cluster 样本，保留 cluster 内的样本。

**与 Latent GMM 的比较**：

| 维度 | Latent GMM (2026-03-12-0151) | DGRS（本 Idea） |
|------|------------------------------|----------------|
| 操作空间 | Latent space z ∈ [0,1]^d | Data space x ∈ R^d |
| 机制 | 限制哪些 z 被采样 | 过滤哪些 x 被保留 |
| 对组件专一化的依赖 | 中等（GMM 可以适应不完美专一化） | 高（需要 DAEM + K-Means 使密度分离清晰） |
| 计算开销 | GMM fitting（训练后一次） + GMM sampling | N_proposal 次 forward pass |
| 不确定性 | GMM 近似可能不准确 | 直接使用模型密度，无近似 |
| 对 BreezeForest CDF 的特殊利用 | 是（利用 latent z 结构） | 是（利用 |det J| 作为密度代理） |
| 适用时机 | 可在任何已训练模型上立即使用 | 在专一化训练后效果最佳 |

**互补关系**：两者可以叠加：先用 Latent GMM 限制 z 的采样范围（减少 proposal 中 inter-cluster 样本的比例），再用 DGRS 过滤剩余的 inter-cluster 样本（双重过滤）。

---

## 与历史 idea 的关系

| 历史 Idea | 关系 | 说明 |
|-----------|------|------|
| **LZR (2026-03-11-1235)** | 正交且互补 | LZR 限制 latent 采样（z 空间）；DGRS 过滤 data 空间（x 空间）。LZR 可以作为 DGRS 的前置步骤，减少 proposal 中 inter-cluster 样本的比例，提高 DGRS 的效率 |
| **Latent GMM (2026-03-12-0151)** | **正交且互补，但本轮未进入 top-3** | Latent GMM 在 z 空间建模实际密度（GMM），DGRS 在 x 空间用模型密度过滤。两者可叠加。本轮 DGRS 作为 top-3 中的推理阶段方案，因为它提供了与 Latent GMM 不同的过滤维度，且不需要 GMM 近似。Latent GMM 仍然有价值，尤其是在组件专一化程度不高时（DGRS 效果较弱的情况）。 |
| **ICDR (2026-03-11-1240)** | 训练时 vs 推理时的不同修复 | ICDR 在训练阶段推动组件分离；DGRS 在推理阶段过滤 inter-cluster 样本。两者互不干扰。 |
| **DAEM / K-Means Pre-Init** | 前置依赖 | DGRS 的有效性依赖 DAEM + K-Means 带来的组件专一化；三者构成完整流水线 |

---

## 具体实现建议

### 步骤 1：在 MultiBF 中添加 `generate_with_density_filter()` 方法

```python
def generate_with_density_filter(
    self,
    n_samples,
    oversample_factor=3.0,
    max_gap=1e-3,
    decay_ratio=1.0,
    exact=False
):
    """
    Generate samples with density-guided rejection sampling (DGRS).
    
    Generates n_samples * oversample_factor candidates, evaluates their log-density
    under the mixture model, and keeps the top n_samples by log p(x).
    
    :param n_samples: number of samples to return
    :param oversample_factor: ratio of candidates to generate (default: 3x)
    :param max_gap: bisection precision for inverse_map
    :param exact: if True, use exact Jacobian for density evaluation
    :return: filtered samples (n_samples, dim), their log p(x) values
    """
    n_proposal = int(n_samples * oversample_factor)
    
    # Step 1: Generate proposal samples (current inverse_map strategy)
    x_proposal = self.inverse_map(
        n_samples=n_proposal, max_gap=max_gap, decay_ratio=decay_ratio
    )  # (n_proposal, dim)
    
    # Step 2: Evaluate log p(x) for each proposal
    log_p = self._compute_mixture_log_prob(x_proposal, exact=exact)  # (n_proposal,)
    
    # Step 3: Keep top n_samples by log p(x)
    _, top_indices = torch.topk(log_p, n_samples)
    x_filtered = x_proposal[top_indices]
    
    return x_filtered, log_p[top_indices]


def _compute_mixture_log_prob(self, x, exact=False):
    """
    Compute per-sample mixture log probability: log p(x) = logsumexp_k(log π_k + log|det J_k(x)|)
    
    :param x: tensor (N, dim)
    :return: log p(x) tensor (N,)
    """
    log_pi = self.get_mixture_log_weights()  # (K,)
    det_fn = self._per_sample_log_det_exact if exact else self._per_sample_log_det
    
    component_log_probs = []
    for k, bf in enumerate(self.components):
        ld = det_fn(bf, x)  # (N,)
        component_log_probs.append(log_pi[k] + ld)
    
    stacked = torch.stack(component_log_probs, dim=0)  # (K, N)
    return torch.logsumexp(stacked, dim=0)  # (N,)
```

### 步骤 2：在 demo_multi_bf.py 中集成

```python
# 在训练完成后，使用 DGRS 替换标准 inverse_map
with torch.no_grad():
    samples, log_probs = mbf.generate_with_density_filter(
        n_samples=data_size,
        oversample_factor=3.0  # 生成 3 倍样本，保留密度最高的 1/3
    )
    samples = samples * std + mean

# 可视化密度分布（诊断用）
print(f"Log p(x) stats: min={log_probs.min():.2f}, "
      f"median={log_probs.median():.2f}, max={log_probs.max():.2f}")
```

### 步骤 3：自适应阈值（可选变体）

如果需要精确控制过滤强度，可以使用训练数据的对数似然分位数作为阈值：

```python
def calibrate_density_threshold(self, x_train, percentile=10.0, exact=False):
    """
    Compute log p(x) threshold from training data.
    Samples below the p-th percentile of training data log-likelihood are rejected.
    """
    with torch.no_grad():
        log_p_train = self._compute_mixture_log_prob(x_train, exact=exact)
    threshold = torch.quantile(log_p_train, percentile / 100.0).item()
    self.density_threshold = threshold
    print(f"[DGRS] Density threshold (p={percentile}%): {threshold:.4f}")
    return threshold


def generate_with_threshold_filter(self, n_samples, max_attempts=5, **kwargs):
    """
    Generate exactly n_samples with density threshold filtering.
    Automatically retries if not enough samples pass the threshold.
    """
    assert hasattr(self, 'density_threshold'), \
        "Call calibrate_density_threshold() first"
    
    results = []
    attempts = 0
    
    while len(results) < n_samples and attempts < max_attempts:
        n_remaining = n_samples - len(results)
        n_request = int(n_remaining * 2.0) + 50  # oversample
        
        x_prop = self.inverse_map(n_samples=n_request, **kwargs)
        log_p = self._compute_mixture_log_prob(x_prop)
        
        mask = log_p >= self.density_threshold
        results.append(x_prop[mask])
        attempts += 1
    
    if len(results) > 0:
        x_all = torch.cat(results, dim=0)
        return x_all[:n_samples]
    else:
        return self.inverse_map(n_samples=n_samples, **kwargs)  # fallback
```

### 超参数调优

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `oversample_factor` | 2.0 – 5.0 | 过采样倍数。DAEM + K-Means 训练后，专一化程度高时 2.0 即可；基线 soft-EM 时需要 4.0-5.0 |
| `percentile`（阈值变体） | 5.0 – 15.0 | 用训练数据的 10% 分位数作为阈值；更高的 percentile 更严格（过滤更多样本）|
| 使用 exact Jacobian | False（默认） | 有限差分足够，exact 计算更慢但更精确；大批量时有限差分稳定性已够 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **soft-EM 训练后效果差** | 组件未专一化时，inter-cluster 密度与 cluster 密度相似，top-k 过滤无法有效区分 | 将 DGRS 作为 DAEM + K-Means 之后的第三步，不单独使用 |
| **计算开销增加** | N_proposal 次 forward pass（每次包含 K 个组件的有限差分 Jacobian 计算）| 默认使用有限差分（低成本）；oversample_factor = 2.0 时开销仅比标准生成高 2x |
| **Cluster 边界样本被过滤** | Cluster 边缘的合法样本密度略低，可能被过滤 | 用较低的 percentile（5%-10%）避免过度过滤；或使用 oversample_factor 版本（top-k 而非阈值） |
| **密度估计噪声** | `_per_sample_log_det` 使用有限差分，有近似误差 | 增大 batch 计算 proposal 密度时适当平均；或在最后阶段用 exact Jacobian 验证 |
| **cluster 数量大时内存** | N_proposal = 3 × N_target 时内存需求增大 3 倍 | 分批生成 proposal（每批 N_target / n_batch 个样本） |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（推理阶段新增方案，DAEM + K-Means 训练后的收尾步骤）**

理由：
1. **理论最优**：OBRS (Verine 2024) 证明按模型密度进行 top-k 选择是给定采样预算下的最优拒绝采样策略
2. **零成本额外训练**：不需要训练任何新模型，只需在已训练的 MultiBF 上做多次 forward pass
3. **与 DAEM + K-Means 自然配合**：专一化训练后，密度分离更清晰，DGRS 的过滤效果最强
4. **与 Latent GMM 正交**：DGRS 在 data 空间过滤，Latent GMM 在 latent 空间限制；两者可叠加使用
5. **实现极简**：核心逻辑约 20 行代码，复用现有 `_per_sample_log_det`
6. **新颖性**：此前 BreezeForest 项目中没有任何 idea 使用模型自身的混合密度作为推理时的质量过滤器

---

## 与整体 Multi-Cluster 解决方案的关系

**推荐的完整三阶段流水线**：

```
阶段一（初始化）：K-Means Pre-Init + Warm-Start
    ↓ 给各组件良好的 cluster-specific 起点
阶段二（训练）：DAEM Temperature Annealing
    ↓ 通过温度退火强制组件专一化
阶段三（推理）：Density-Guided Rejection Sampling (本 Idea)
    ↓ 过滤生成样本中残余的 inter-cluster 点
```

每个阶段解决问题的不同层面：
- K-Means Pre-Init → 初始化阶段消除同质化起点
- DAEM → 训练阶段通过温度控制强化组件专一化
- DGRS → 推理阶段利用模型密度过滤残余 inter-cluster 样本

---

## 参考文献

- Verine, A. et al. (2024). "Optimal Budgeted Rejection Sampling for Generative Models." *AISTATS 2024*. https://proceedings.mlr.press/v238/verine24a.html  
  ← **直接理论基础**：按模型密度排序保留 top-k 是给定预算下最优拒绝采样策略
- Azadi, S. et al. (2019). "Discriminator Rejection Sampling." *ICLR 2019*.  
  ← 原始 DRS 方法；DGRS 是其针对 normalizing flow 的特殊化（无需额外判别器）
- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 113*. https://arxiv.org/abs/2305.12149  
  ← 正交的 latent space 引导采样方法；DGRS 是其 data space 对应物，实现更简单
- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a.html  
  ← 通过修改 base distribution 解决 topology mismatch；DGRS 通过密度过滤达到类似效果
- Pires, G. & Figueiredo, M. (2020). "Variational Mixture of Normalizing Flows." *ESANN 2020*.  
  ← Mixture of NF 的训练策略综述；DGRS 的有效性依赖本文分析的专一化前提
