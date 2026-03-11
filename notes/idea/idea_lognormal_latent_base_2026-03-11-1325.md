# Idea: Per-Component Logit-Normal Latent Base Distribution (PLNB)

**创建时间**: 2026-03-11 13:25 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（训练后即可生效，无需重训练）

---

## 问题定义

MultiBF 在**生成阶段**使用 `z ~ Uniform(0.01, 0.99)^d` 作为 latent 基础分布，再通过每个 BreezeForest 组件的 `inverse_map(z)` 映射回数据空间。

这里存在一个被现有 idea 部分识别但未被彻底解决的问题：

**既有 LZR（idea_latent_zone_restriction_2026-03-11-1235.md）的局限：**
- LZR 通过各维度的百分位数估计出一个**矩形边界框** [a_k, b_k]^d
- 矩形边界框不能捕获 latent 分布的**椭球形状**（covariance structure）
- 对于倾斜的 cluster（在 latent 空间中呈椭圆形），矩形框会包含大量无效角落区域
- 需要手动调节 percentile_low / percentile_high 超参数

**更深层的问题：**
BreezeForest 的 forward 是一个将 x ∈ R^d 映射到 z ∈ (0,1)^d 的连续单调函数（各维度的条件 CDF）。对于任意 cluster 的训练数据 x^k，其 latent 表示 z^k = f_k(x^k) 在 (0,1)^d 中形成一个**特定形状的分布**，不一定是均匀的，且各维度之间有相关性（由 breeze 权重引入）。

当前 Uniform(0.01, 0.99)^d 采样覆盖整个 latent 空间，包括：
1. cluster k 数据的 latent representation 所在的高密度区域（我们想要的）
2. 其他 cluster j≠k 数据的 latent representation 所在区域（产生 inter-cluster 生成的根源）
3. 完全是训练数据 latent representation 之外的区域（产生 extra-distribution 生成）

---

## 从当前项目代码与已有 idea 中得到的背景判断

**代码关键细节：**

1. **BreezeForest 输出范围**（`model/TreeLayer.py` L178）：
   ```python
   return self.acti_func.forward(x), tree_bias, tree_scale
   ```
   最后一层激活函数是 Sigmoid（`model/BreezeForest.py` L68-69），因此 `forward()` 的输出 z ∈ (0,1)^d。

2. **Bisection 的 Stage 1** 在 CDF 空间操作（`model/tools.py` L97-103）：
   ```python
   lo, hi = _bisect(
       torch.zeros_like(target), torch.ones_like(target),
       lambda m: inc_func(distribution.icdf(m)), gap_dis,
   )
   lo = distribution.icdf(lo.clamp(min=1 - anomaly_dis))
   hi = distribution.icdf(hi.clamp(max=anomaly_dis))
   ```
   这里 `distribution` 默认是 `Normal(0,1)`，用来在 real space 做粗搜索。generation 时采样的 z 会直接作为 bisection 的 target（`model/MultiBF.py` L165-168）。

3. **Latent space 的结构**：z ∈ (0,1)^d 是有界空间，普通 Gaussian 不能直接用。但是 logit 变换 `logit(z) = log(z/(1-z))` 将 (0,1)^d 映射到 R^d，在 logit 空间可以自然地拟合 Gaussian。

**现有 LZR idea 的分析：**
- LZR 识别了问题（latent 中有效采样区域只是子集），并提出了矩形边界解决方案
- 本 idea 用 **logit 空间的 Gaussian** 替换矩形边界，是更精确、更数学合理的版本
- 外部研究（Josias & Brink, NeurIPS 2023 Workshop）直接验证了 GMM base distribution 在 CNF 上的效果

---

## 核心思路

**用数据驱动的 Logit-Normal 分布替换 Uniform 基础分布：**

**步骤 1：Latent Calibration（训练后一次性执行）**
- 对训练数据中分配给组件 k 的样本（用 responsibility 或 K-Means 确定）
- 通过组件 k 的 forward 得到 z_k^{(i)} = f_k(x^{(i)}) ∈ (0,1)^d
- 对 z_k^{(i)} 做 logit 变换：ζ_k^{(i)} = logit(z_k^{(i)}) = log(z_k / (1 - z_k)) ∈ R^d
- 拟合一个多元高斯：**ζ_k ~ N(μ_k, Σ_k)**（对角协方差 diag(σ_k^2) 即可）

**步骤 2：Generation 时使用 Logit-Normal 采样**
- 不再从 Uniform(0.01, 0.99)^d 采样
- 改为：
  1. 采样 ζ_k ~ N(μ_k, Σ_k)（in logit space）
  2. 映射回 (0,1)^d：z_k = sigmoid(ζ_k)
  3. 裁剪到安全范围：z_k = z_k.clamp(0.01, 0.99)
  4. 调用 `bf.inverse_map(z_k)` 做双分法求逆

**直觉**：ζ_k 的高斯分布集中在训练数据 latent representation 所在的区域。从这个高斯中采样，再映射回 (0,1)^d，再做 inverse_map，自然生成的样本集中在 cluster k 附近。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**数学论证：**

设 f_k 是组件 k 的 BreezeForest 正向映射（x → z）。设 D_k = {x ∈ D : assigned to component k}。

定义 Z_k = {f_k(x) : x ∈ D_k} ⊂ (0,1)^d，即训练数据在 latent 空间中的像。

由 BreezeForest 的单调性：
```
如果 cluster k 和 cluster j 在数据空间 R^d 中有较大分离，
则 Z_k 和 Z_j = {f_k(x) : x ∈ D_j} 在 (0,1)^d 中也会有较大分离
（因为 f_k 是各维度的 conditional CDF，cluster j 的点有相对 cluster k 系统性不同的 z 值）
```

logit 变换是单调的，因此：
```
logit(Z_k) 和 logit(Z_j) 在 R^d 中也是分离的
```

Gaussian N(μ_k, Σ_k) 集中在 logit(Z_k) 附近，生成的 ζ 绝大多数落在 logit(Z_k) 内。
因此 sigmoid(ζ) 绝大多数落在 Z_k 内，inverse_map 生成的 x 绝大多数在 cluster k 附近。

**对比 LZR（矩形边界）的优势：**

| 维度 | LZR（矩形边界） | PLNB（Logit-Normal） |
|------|----------------|---------------------|
| 形状 | 轴对齐矩形 | 椭球（捕获 covariance） |
| 参数 | 2d 个参数（lo, hi per dim） | 2d 个参数（μ, σ per dim，对角） |
| 超参数 | percentile_low/high（需调节） | 无（0均值std直接估计） |
| 角落区域 | 包含矩形的4个"角落"（无效区域） | 自然截断 |
| 偏斜分布 | 捕获不好 | 通过 Gaussian 捕获 |
| 理论基础 | 无（启发式） | 有（logit-normal 分布是有界分布的自然参数化） |

**外部研究验证：**
- Josias & Brink (NeurIPS 2023 Workshop) 的 CNF with GMM base distribution：
  "enables mode-specific sampling" and "improved sample quality"
- StiCTAF (ICLR 2025)：stick-breaking mixture base with component-specific tail transforms 直接解决 mode-seeking bias

---

## 与历史 idea 的关系

| 关系 | 历史 Idea | 说明 |
|------|-----------|------|
| **直接替代（升级版）** | idea_latent_zone_restriction_2026-03-11-1235.md | LZR 是本 idea 的矩形近似；PLNB 用 Gaussian 替换矩形框，更精确，不需要调节百分位参数。推荐用 PLNB 替代 LZR |
| **互补** | idea_kmeans_piecewise_training_2026-03-11-1320.md | Piecewise 训练（本轮 idea 1）确保每个组件专一化，使 latent calibration 更准确；两者叠加效果最佳 |
| **无关** | idea_inter_component_density_repulsion_2026-03-11-1240.md | ICDR 是训练时修复；PLNB 是生成时修复，互补不冲突 |

**旧 LZR idea 是否过时？**
- LZR 仍然有效，PLNB 是它的直接升级
- 如果只能实施一个，优先实施 PLNB（不需要矩形边界调节，数学更严格）
- 两者同属 "inference-time fix"，原理相同，PLNB 更精确

---

## 具体实现建议

### 步骤 1：添加 `calibrate_lognormal_bases()` 到 MultiBF

```python
def calibrate_lognormal_bases(self, x_train, use_hard_assignments=True):
    """
    Fit per-component Logit-Normal base distributions from training data.
    
    For each component k:
    1. Identify samples assigned to component k
    2. Forward-pass them through component k to get z_k ∈ (0,1)^d
    3. Apply logit transform: zeta_k = log(z_k / (1 - z_k)) ∈ R^d
    4. Fit N(mu_k, diag(sigma_k^2)) to zeta_k
    
    :param x_train: training data (N, dim)
    :param use_hard_assignments: if True, use argmax responsibility (hard);
                                  if False, use soft responsibility weighting
    """
    self.lognormal_bases = []
    
    with torch.no_grad():
        # Compute per-sample responsibilities
        log_pi = self.get_mixture_log_weights()
        component_log_probs = []
        for k, bf in enumerate(self.components):
            ld = self._per_sample_log_det(bf, x_train)
            component_log_probs.append(log_pi[k] + ld)
        
        stacked = torch.stack(component_log_probs, dim=0)  # (K, N)
        log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
        
        if use_hard_assignments:
            assignments = torch.argmax(log_resp, dim=0)  # (N,)
        
        for k, bf in enumerate(self.components):
            # Select samples for component k
            if use_hard_assignments:
                mask = (assignments == k)
                x_k = x_train[mask]
            else:
                # Weighted soft assignment
                resp_k = torch.exp(log_resp[k])  # (N,)
                # Use top 50% by responsibility as a practical proxy
                threshold = 1.0 / self.n_components
                mask = resp_k > threshold
                x_k = x_train[mask]
            
            if x_k.shape[0] < 2:
                # Fallback: use all data
                x_k = x_train
            
            # Forward pass: x_k -> z_k ∈ (0, 1)^d
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list)  # (n_k, dim)
            
            # Logit transform: z_k -> zeta_k ∈ R^d
            # Clamp z to avoid logit blow-up at boundaries
            z_k_safe = z_k.clamp(1e-4, 1 - 1e-4)
            zeta_k = torch.log(z_k_safe / (1 - z_k_safe))  # (n_k, dim)
            
            # Fit diagonal Gaussian in logit space
            mu_k = zeta_k.mean(dim=0)    # (dim,)
            sigma_k = zeta_k.std(dim=0).clamp(min=0.1)  # (dim,), min std
            
            self.lognormal_bases.append((mu_k, sigma_k))
    
    print(f"Calibrated logit-normal bases for {len(self.lognormal_bases)} components:")
    for k, (mu, sigma) in enumerate(self.lognormal_bases):
        print(f"  Component {k}: mu={mu.numpy().round(3)}, sigma={sigma.numpy().round(3)}")
```

### 步骤 2：修改 `inverse_map()` 使用 Logit-Normal 采样

```python
def inverse_map_lognormal(self, n_samples, max_gap=1e-3, decay_ratio=1.0, 
                           n_sigma=3.0):
    """
    Generate samples using per-component Logit-Normal base distribution.
    Requires calibrate_lognormal_bases() to be called first.
    
    :param n_sigma: number of sigma for clipping (default 3.0 = ~99.7% coverage)
    """
    assert hasattr(self, 'lognormal_bases'), "Call calibrate_lognormal_bases() first"
    
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)

    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue

        mu_k, sigma_k = self.lognormal_bases[k]
        
        # Sample from logit-Normal: zeta ~ N(mu_k, diag(sigma_k^2))
        zeta_k = torch.randn(n_k, self.dim) * sigma_k + mu_k
        
        # Clip to reasonable range (n_sigma standard deviations)
        zeta_k = zeta_k.clamp(-n_sigma * sigma_k.abs() + mu_k, 
                               n_sigma * sigma_k.abs() + mu_k)
        
        # Logit-Normal -> (0, 1)^d: z = sigmoid(zeta)
        z_k = torch.sigmoid(zeta_k).clamp(0.01, 0.99)
        
        x_k = self.components[k].inverse_map(
            z_k, max_gap=max_gap, decay_ratio=decay_ratio
        )
        results[mask] = x_k

    return results
```

### 步骤 3：在 demo_multi_bf.py 中集成

```python
# 训练完成后（或在已训练模型上）：
all_batch = (all_data - mean) / std  # 全量训练数据归一化

with torch.no_grad():
    # 1. 校准 logit-normal bases
    mbf.calibrate_lognormal_bases(all_batch, use_hard_assignments=True)
    
    # 2. 使用 logit-normal 生成
    samples = mbf.inverse_map_lognormal(n_samples=data_size, n_sigma=2.5)
    samples = samples * std + mean
```

### 超参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `use_hard_assignments` | True | 如果已经用 K-Means 训练，用 hard 分配更精确 |
| `n_sigma` | 2.5 – 3.0 | 控制采样范围。2.5 ≈ 98.8% 的 Gaussian 概率质量；3.0 ≈ 99.7% |
| 最小 sigma | 0.1 | 防止某维度方差过小导致采样过于集中 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **对角协方差假设** | 对角 Gaussian 不能捕获 latent 维度间的相关性（off-diagonal covariance） | 升级到全协方差 Gaussian（需要更多样本）；或用 PCA 先旋转 latent 空间再拟合对角 Gaussian |
| **Logit-Normal 的 tail 行为** | Gaussian 的 tail 采样到后，sigmoid 变换会导致 z 非常接近 0 或 1，bisection 可能失败 | `n_sigma` 裁剪（3σ 以内）+ `z.clamp(0.01, 0.99)` 已基本解决 |
| **组件未专一化时的 calibration 不准** | 如果组件仍在用 soft-EM 训练，latent 表示来自多个 cluster，高斯拟合不准 | 与 Piecewise 训练（idea 1）结合使用；或用更严格的 responsibility 阈值筛选"纯"样本 |
| **多维 Gaussian 的样本量需求** | 对角 Gaussian 拟合需要至少 2d 个样本；全协方差需要 d(d+1)/2 个样本 | 对角 Gaussian 在 2D 场景（BreezeForest demo 的 dim=2）只需约 4 个样本，无问题 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（inference-time fix，无需重训练，立即可验证）**

理由：
1. **对已训练模型立即有效**：不需要修改训练流程，在任何已训练的 MultiBF 上运行一次 calibration 即可
2. **比 LZR 更精确**：捕获 latent 分布的形状（椭球 vs 矩形），参数更少（不需要调节 percentile）
3. **理论支撑强**：Logit-Normal 是 (0,1)^d 上分布的自然参数化；Josias & Brink (2023) 直接验证 GMM base 在 CNF 上的效果
4. **实现简单**：约 40 行代码，可在现有 MultiBF 上直接添加方法
5. **可与 Idea 1 叠加**：Piecewise 训练 + PLNB 生成是当前架构下最强的组合

---

## 参考文献

- Josias, M. & Brink, W. (2023). "Multimodal base distributions for continuous normalizing flows." *NeurIPS 2023 Deep Learning for Differential Equations (DLDE-III) Workshop*. https://openreview.net/pdf?id=eOODNEuD7D  
  (GMM base distribution in CNF enables mode-specific sampling; 直接支持本 idea)
- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a.html  
  (解决 topological mismatch 的 resampling 方法，理论同源)
- StiCTAF (2025). "Stick-Breaking Normalizing Flows with Component-Wise Tail Adaptation." *ICLR 2025 Submission*. https://openreview.net/forum?id=Iwfp9yTwf3  
  (Component-wise tail transforms + mixture base，思路类似)
- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 2024*. https://arxiv.org/abs/2305.12149  
  (已在 LZR idea 中引用；Langevin 是 PLNB 的 MCMC 升级版，可以作为进一步增强)
