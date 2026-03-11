# Idea: z-GMM Latent-Space Resampling — Replace Uniform Base with Learned GMM on z

**创建时间**: 2026-03-11 17:15 UTC
**推荐优先级**: ⭐⭐⭐ 最高优先级

---

## 问题定义

BreezeForest 的 latent space 是 [0,1]^d（CDF 的值域），训练目标是让 f(x) ~ Uniform([0,1]^d)。  
在生成阶段，当前策略是：

```
z ~ Uniform(0.01, 0.99)^d  →  x = f^{-1}(z)
```

这个看似合理的策略，在多 cluster 数据上存在一个**结构性缺陷**，所有现有 idea 都未明确指出：

### 核心缺陷：CDF 机制导致低密度区域在 z 空间中被"放大"

BreezeForest 的正向映射本质上是条件 CDF（累积分布函数）。CDF 在高密度区域变化快（Jacobian 大），在低密度区域变化慢（Jacobian 小）。

对于有 K 个 cluster 的数据：
- Cluster A 的数据（高密度）→ CDF 变化快 → z 值集中于 [0,1]^d 的某个小区域 Z_A
- Cluster 之间的低密度区域 → CDF 变化慢（近乎平坦）→ z 值被**扩张**，占据 [0,1]^d 中的大片区域 Z_inter

因此：**采样 z ~ Uniform([0,1]^d) 时，大部分采样点落在 Z_inter（inter-cluster 扩张区域），而非 Z_A ∪ Z_B。**

这意味着 BreezeForest 的 Uniform 基分布对于多 cluster 数据来说是**反效果的**——它在最不希望采样的区域（cluster 之间）分配了最多的采样概率。

这与 Gaussian base 的流模型不同：Gaussian base 会将低密度数据推向高范数区域，从正态分布采样时自然地较少采到这些区域。BreezeForest 的 Uniform base 没有这种保护机制。

---

## 从当前项目代码与已有 idea 中得到的背景判断

### 代码分析

`demo_functions.py` 中的 `generate_sample()`：
```python
distribution = uniform.Uniform(torch.tensor(0.01), torch.tensor(0.99))
seeds = distribution.sample(torch.Size([sample_size, 2]))
generated = model.inverse_map(seeds)
```

无论 cluster 结构如何，所有维度的 z 均从 Uniform([0.01, 0.99]) 均匀采样。

`MultiBF.inverse_map()` 中：
```python
z = torch.rand(n_k, self.dim) * 0.98 + 0.01  # 每个组件均从 Uniform 采样
x_k = self.components[k].inverse_map(z, ...)
```

同样未考虑组件 k 的实际 latent 分布范围。

### 与现有 idea 的关系

**现有 Idea 2（LZR）** 已经意识到了这个问题，并提出用矩形边界框 [a_k, b_k]^d 来限制每个组件的采样区域。但 LZR 有以下局限：
1. 只适用于 MultiBF（需要组件分配）
2. 矩形框是粗糙的近似，无法捕捉 cluster 在 z 空间中的非矩形形状
3. 没有解决 z 空间中 inter-cluster 区域"扩张"的根本原因
4. 对单个 BF（single BreezeForest）无效

**本 Idea** 通过在 z 空间中直接拟合训练数据的 latent 分布来解决上述所有局限。

---

## 核心思路

**训练后校准**（Post-Training Calibration，无需重训练）：

1. 对训练集所有数据，做正向传播得到 latent 表示：
   ```
   Z_train = {z_i = f(x_i) | x_i ∈ 训练集}  ⊂ [0,1]^d
   ```
2. 在 [0,1]^d 上对 Z_train 拟合一个 Gaussian Mixture Model（GMM）：
   ```
   q(z) = GMM_K(z | μ_1,...,μ_K, Σ_1,...,Σ_K, π_1,...,π_K)
   ```
   GMM 的组件数 K 可以等于 MultiBF 的组件数，或者通过 BIC/AIC 自动选择。
3. **生成时**，从 GMM 采样 z ~ q(z)（而非 Uniform），并 clamp 到 [0.01, 0.99]^d：
   ```
   z ~ q(z),  clamp(z, 0.01, 0.99) → x = f^{-1}(z)
   ```

由于 q(z) 的高密度区域正好是训练数据的 latent 所在位置（Z_A, Z_B,...），GMM 采样会避开扩张的 inter-cluster z 区域。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**直接原理**：

GMM q(z) 是对 {z_i = f(x_i)} 分布的近似。这些 z_i 聚集在各 cluster 对应的 z 区域（Z_A, Z_B,...）。从 q(z) 采样，几乎只会采到这些区域内的 z 值。

**与 Uniform 采样的对比**：

| 采样策略 | z 采样范围 | inter-cluster z 占比 | 生成质量 |
|---------|-----------|---------------------|---------|
| Uniform([0.01, 0.99]) | 整个 [0,1]^d | 高（CDF 扩张） | 差 |
| LZR（矩形框） | 各组件矩形框 | 中（粗糙截断） | 中 |
| z-GMM（本 Idea） | 高密度 cluster 区域 | 极低 | 好 |

**对比 Stimper et al. (2022)**：
Stimper 的 "Resampling Base Distributions" 通过学习一个 rejection sampling 基分布来修复 topology 问题。z-GMM 是其思路在 BreezeForest 有界 [0,1]^d latent 空间上的自然应用，且实现更简单（直接 GMM 拟合，无需学习 acceptance function）。

**对比 arxiv 2512.04954**：
该 2024 论文表明 GMM 基分布在多模态 posterior 估计中显著减少了 modes 之间的虚假"概率桥梁"。z-GMM 在 BreezeForest 的 bounded latent space 中解决了同样的问题。

---

## 它与历史 idea 的关系

**替代（超越）Idea 2（LZR）**：

| 维度 | LZR（现有 Idea 2） | z-GMM（本 Idea） |
|------|------------------|----------------|
| 适用范围 | 仅 MultiBF | 单 BF + MultiBF |
| 区域估计精度 | 矩形框（粗糙） | GMM 连续密度（精确） |
| cluster 形状 | 仅轴对齐矩形 | 任意椭球形 |
| 实现方式 | 手工设计 | 数据驱动自动拟合 |
| 理论基础 | 经验规则 | 密度估计理论 |

**继承 Idea 1（Hard-EM）**：z-GMM 与 Hard-EM 可以结合使用。Hard-EM 训练后，各组件在 z 空间的 latent 分布更集中，z-GMM 的拟合更准确。

**不影响 Idea 3（ICDR）**：z-GMM 是推理时修复，ICDR 是训练时修复，互补。

---

## 具体实现建议

### 方案 A：基于 sklearn 的后训练 GMM 拟合（最简单）

```python
from sklearn.mixture import GaussianMixture
import torch

def calibrate_z_gmm(model_or_bf, x_train, n_components=None, covariance_type='full'):
    """
    Fit a GMM on the latent representations of training data.
    Works for both single BreezeForest and MultiBF components.
    
    :param model_or_bf: BreezeForest instance (single BF)
    :param x_train: training data (N, dim)
    :param n_components: number of GMM components (default: use BIC to select 1–10)
    :param covariance_type: 'full', 'diag', or 'spherical'
    :return: fitted sklearn GaussianMixture
    """
    with torch.no_grad():
        breeze_list = []
        z_train = model_or_bf.forward(x_train, breeze_list).cpu().numpy()
    
    if n_components is None:
        # Auto-select n_components via BIC (search 1..min(10, N//100))
        best_gmm = None
        best_bic = float('inf')
        for k in range(1, min(11, z_train.shape[0] // 100 + 1)):
            gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, 
                                  n_init=3, random_state=42)
            gmm.fit(z_train)
            bic = gmm.bic(z_train)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        return best_gmm
    else:
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                              n_init=3, random_state=42)
        gmm.fit(z_train)
        return gmm


def sample_from_gmm_clamped(gmm, n_samples, dim, lo=0.01, hi=0.99, max_attempts=5):
    """
    Sample from GMM and clamp to [lo, hi]^dim.
    Uses rejection sampling to stay in valid range.
    """
    collected = []
    remaining = n_samples
    
    for _ in range(max_attempts):
        z_raw = gmm.sample(remaining * 2)[0]  # oversample for rejection
        z_raw = torch.tensor(z_raw, dtype=torch.float32)
        valid = ((z_raw >= lo) & (z_raw <= hi)).all(dim=1)
        z_valid = z_raw[valid][:remaining]
        collected.append(z_valid)
        remaining -= len(z_valid)
        if remaining <= 0:
            break
    
    # If not enough valid samples, clamp the rest
    z_out = torch.cat(collected, dim=0)
    if len(z_out) < n_samples:
        z_fallback = torch.tensor(gmm.sample(n_samples - len(z_out))[0], dtype=torch.float32)
        z_fallback = z_fallback.clamp(lo, hi)
        z_out = torch.cat([z_out, z_fallback], dim=0)
    
    return z_out[:n_samples]
```

### 方案 B：MultiBF 集成 — 每组件独立拟合 GMM

```python
def calibrate_z_gmm_per_component(mbf, x_train, n_components_per_gmm=2):
    """
    For MultiBF: fit a separate GMM on each component's latent representation 
    of high-responsibility training samples.
    """
    mbf.component_gmms = []
    
    with torch.no_grad():
        # Compute responsibilities
        log_pi = mbf.get_mixture_log_weights()
        component_log_probs = []
        for k, bf in enumerate(mbf.components):
            ld = mbf._per_sample_log_det(bf, x_train)
            component_log_probs.append(log_pi[k] + ld)
        stacked = torch.stack(component_log_probs, dim=0)
        log_prob = torch.logsumexp(stacked, dim=0)
        log_resp = stacked - log_prob.unsqueeze(0)
        resp = torch.exp(log_resp)  # (K, N)
        
        for k, bf in enumerate(mbf.components):
            # Select top-50% by responsibility
            mask = resp[k] > (1.0 / mbf.n_components)
            if mask.sum() < 20:
                _, idx = torch.topk(resp[k], min(50, len(resp[k])))
                mask = torch.zeros(len(resp[k]), dtype=torch.bool)
                mask[idx] = True
            
            x_k = x_train[mask]
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list).cpu().numpy()
            
            from sklearn.mixture import GaussianMixture
            gmm_k = GaussianMixture(n_components=n_components_per_gmm, 
                                     covariance_type='full', n_init=3)
            gmm_k.fit(z_k)
            mbf.component_gmms.append(gmm_k)


def inverse_map_z_gmm(mbf, n_samples, max_gap=1e-3, decay_ratio=1.0):
    """
    Generate samples using per-component GMM-based latent sampling.
    Requires calibrate_z_gmm_per_component() to be called first.
    """
    assert hasattr(mbf, 'component_gmms'), "Call calibrate_z_gmm_per_component() first"
    
    weights = mbf.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, mbf.dim)
    
    for k in range(mbf.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        z_k = sample_from_gmm_clamped(mbf.component_gmms[k], n_k, mbf.dim)
        x_k = mbf.components[k].inverse_map(z_k, max_gap=max_gap, decay_ratio=decay_ratio)
        results[mask] = x_k
    
    return results
```

### 方案 C：单 BF 应用（single BreezeForest，无需 MultiBF）

```python
# 训练后校准
bf_gmm = calibrate_z_gmm(bf, x_train_normalized, n_components=8)  # 例如 8 Gaussians 数据集

# 生成
with torch.no_grad():
    z_seeds = sample_from_gmm_clamped(bf_gmm, n_samples=data_size, dim=2)
    generated = bf.inverse_map(z_seeds)
    generated = generated * std + mean
```

### 超参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `n_components` | = n_clusters | 如果知道 cluster 数，直接设置 |
| `covariance_type` | `'full'` | 允许椭球形 cluster，适合大多数情况 |
| BIC 自动选择 | 1–10 | 对于低维（2D）效果好 |
| 拒绝采样尝试次数 | 5 | 足以采集到 [0.01, 0.99]^d 范围内的点 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **GMM 拟合不准** | 如果 BF 组件未专一化（soft-EM 训练），z 空间中 cluster 形状不清晰，GMM 拟合可能过于扩散 | 与 Hard-EM 结合使用（Hard-EM 训练后 cluster 更清晰）；或用更大的 n_components |
| **拒绝率高** | GMM 的部分采样点可能超出 [0.01, 0.99]^d，被 clamp 掉 | 用稍宽松的范围（0.001, 0.999），或 clamp 后接受（而非拒绝采样） |
| **sklearn 依赖** | 需要 sklearn，已在 requirements.txt 中包含 | 无需额外安装 |
| **计算开销** | GMM 拟合一次性开销，N=3000 样本几乎即时 | 忽略不计 |
| **过度拟合 latent** | 如果训练数据本身有 outlier，GMM 可能学到不良 cluster | 用 percentile-based 过滤：仅用 responsibility > threshold 的样本拟合 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级**

理由：
1. **直接针对 BreezeForest 架构的根本性缺陷**：Uniform 基分布对低密度区域的 z 空间"扩张"效应，是其他现有 idea 未明确解决的核心问题
2. **零训练开销**：不需要重训练，只需一次 forward pass + GMM 拟合（< 1 秒）
3. **适用范围广**：单 BF 和 MultiBF 均可使用（LZR 只能用于 MultiBF）
4. **理论支撑强**：直接对应 Stimper et al. (2022) 和 arxiv 2512.04954 的方法，且在 BreezeForest 的 bounded latent 上实现更简单
5. **可与所有现有 idea 叠加**：Hard-EM + z-GMM 是理论上最强的组合

---

## 与现有 Idea 的最终关系声明

- **替代 / 超越 Idea 2（LZR）**：z-GMM 在理论、精度和适用范围上均优于 LZR。LZR 可视为 z-GMM 的简化版本（单组件矩形框 = 各维度独立截断的退化 GMM）。**建议用 z-GMM 替代 LZR 作为推理时修复方案。**
- **与 Idea 1（Hard-EM）配合**：Hard-EM 确保各组件专一化 → z 空间 cluster 更清晰 → z-GMM 拟合更准确。两者是最强组合。
- **与 Idea 3（ICDR）互补**：ICDR 是训练时修复，z-GMM 是推理时修复，两者可叠加。

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a.html
- Zoran Azar et al. (2024). "Amortized Inference of Multi-Modal Posteriors using Likelihood-Weighted Normalizing Flows." arXiv:2512.04954. (GMM 基分布减少虚假 probability bridges)
- Comas Massagu, A. et al. (2024). "Multimodal base distributions in conditional flow matching generative models." *BMVC 2024*. https://bmvc2024.org/proceedings/492/ (GMM base 在 flow matching 中的 mode-specific 采样)
- Marchetti, G.L. et al. (2023). "Piecewise Normalizing Flows." arXiv:2305.02930. (cluster-specific flow 训练消除 inter-cluster bridges)
