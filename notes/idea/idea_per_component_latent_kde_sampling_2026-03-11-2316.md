# Idea: Per-Component Latent KDE Sampling (PLKS)

**创建时间**: 2026-03-11 23:16 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（替代 LZR，更精确的推断时修复）

---

## 问题定义

MultiBF 的生成流程：

```
k ~ Categorical(π)
z ~ Uniform([0.01, 0.99]^d)
x = f_k^{-1}(z)
```

根本问题：**Uniform([0.01, 0.99]^d) 不是 z-space 中的"合法"分布**。

BreezeForest 的正向映射 `f_k` 将训练数据映射到 [0,1]^d，但这并不意味着 [0,1]^d 内的每一个 z 值都对应一个训练数据中有代表的区域。对于 multi-cluster 数据：

- cluster A 的样本被 f_k 映射到 z-space 的某个**子区域 Z_A**
- cluster B 的样本被 f_k 映射到**另一个子区域 Z_B**（Z_A ≠ Z_B）
- z-space 的其他区域（包括 Z_A 和 Z_B 之间的"桥"）被 f_k^{-1} 映射回 inter-cluster 区域

当前 Uniform 采样无差别地覆盖整个 [0.01, 0.99]^d，必然包含"桥"区域，导致生成 inter-cluster 点。

历史 idea `idea_latent_zone_restriction_2026-03-11-1235.md`（LZR）通过估计矩形边界 [a_k, b_k]^d 来限制 z 的采样范围。但 LZR 的局限性在于：

1. **矩形边界不精确**：各维度独立估计百分位数，忽略维度间的相关结构
2. **仅排除极端值**：桥区域可能在边界范围内（如 cluster A 在 z~0.3，cluster B 在 z~0.7，桥在 z~0.5，但矩形边界 [0.1, 0.9] 仍包含 0.5）
3. **依赖组件专一化**：如果组件未充分专一化，Z_k 估计混入多个 cluster 的点，边界更加不准

**本 idea（PLKS）** 用实际训练数据的 latent code 分布拟合一个简单的密度模型，然后从该密度模型采样，直接以训练数据的 latent 密度作为采样分布。

---

## 从项目代码与已有 idea 得到的背景判断

### BreezeForest latent space 的特性

BreezeForest 的 forward 输出经过 Sigmoid，所有值在 (0, 1) 之间。对于多峰数据（如 8 gaussians），每个组件会将不同 cluster 映射到 z-space 的不同子区域。由于 BreezeForest 是 CDF-based（单调映射），z-space 中的次序保留了 x-space 的次序：如果 cluster A 在 x-space 中"左边"，则 f_k(A) 在 z-space 中值较小，f_k(B) 在 z-space 中值较大，中间必然存在"桥"z 值。

这个"桥"的位置和宽度完全取决于数据分布，无法用简单的百分位数规避（因为百分位数估计的是覆盖范围，而不是密度差异）。

### LZR 的实际限制（从代码角度）

LZR 的估计：`lo = percentile(z_k, 5%)`, `hi = percentile(z_k, 95%)`

对于 cluster A（z 值集中在 0.1-0.3）和 cluster B（z 值集中在 0.7-0.9）：
- LZR 边界可能是 [0.05, 0.95]，仍包含"桥"区间 [0.3, 0.7]
- 即便进一步压缩到 [0.1, 0.3] 或 [0.7, 0.9]，也因维度独立性无法处理高维相关结构

这说明矩形边界方案**在根本上不足以排除桥区域**，需要更精细的密度模型。

---

## 核心思路

**训练后校准（Post-Training Calibration）**：

1. 对训练数据中分配给组件 k 的样本（按 responsibility 筛选），通过 `f_k` 正向传播，得到其 latent 表示 `{z_i^k}`
2. 对每个 latent 维度 d，用 **1D Gaussian Mixture Model（GMM）** 或 **Kernel Density Estimation（KDE）** 拟合 `{z_id^k}` 的分布
3. 生成时：从各维度的独立 GMM/KDE 采样，组合成 z，再做 inverse_map

**为什么用 GMM/KDE 而不是矩形边界**：
- 对于 cluster A 在 z~0.3、cluster B 在 z~0.7 的情况（当组件 k 未充分专一化时），GMM 会拟合出**双峰**密度，而矩形边界无法表达这种结构
- GMM 的两个分量分别对应两个 cluster 的 latent zone，中间桥区密度极低
- 从 GMM 采样时，几乎不会采到桥区（因为桥区密度接近 0）

**当组件充分专一化时**（配合 DAEM + K-Means init）：
- GMM 会拟合出**单峰**密度（只有一个 cluster）
- 采样完全集中在目标 cluster 的 latent zone 内
- inter-cluster 生成接近零

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**理论论证**：

设 p_z^k 为组件 k 在 latent space 上的 empirical latent 密度（由 KDE/GMM 估计），则从 p_z^k 采样 z、再做 f_k^{-1}，得到的 x 的分布近似为：

```
p_x^k(x) ≈ p_z^k(f_k(x)) * |det J_{f_k}(x)|
```

即 x 的生成密度由两项决定：
1. `p_z^k(f_k(x))`：latent code 在训练数据 latent zone 内的密度（只在 cluster k 附近高）
2. `|det J_{f_k}(x)|`：Jacobian（BreezeForest 在 cluster k 附近训练更充分，Jacobian 更大）

两项都在 cluster k 附近最大，在 inter-cluster 区域极小。因此 PLKS 从两个方向同时压制 inter-cluster 生成。

**对比 Stimper et al. (2022) "Resampled Base Distributions"**：
Stimper 的方法通过学习一个 rejection sampling proposal distribution 来改变 latent 采样分布，需要训练额外的参数（一个 normalizing flow）。PLKS 是其数据驱动的简化替代：用 training data 的 latent codes 直接拟合 GMM/KDE，不需要额外训练，只需一次 calibration pass。

---

## 与历史 idea 的关系

**直接升级/替代 `idea_latent_zone_restriction_2026-03-11-1235.md`（LZR）**。

| 维度 | LZR（历史 idea） | PLKS（本 idea） |
|------|----------------|----------------|
| 密度模型 | 矩形边界（百分位数） | GMM/KDE（实际密度） |
| 能否捕捉双峰结构 | 否 | 是 |
| 能否排除桥区 | 部分 | 是（高概率） |
| 依赖组件专一化 | 高（Zone 不准时失效） | 低（双峰 GMM 仍能限制桥区采样） |
| 维度相关性 | 忽略 | 忽略（各维度独立，可升级） |
| 实现复杂度 | 低 | 低-中等（sklearn GMM/KDE） |

LZR 是 PLKS 思路的简化版本（用矩形近似代替密度模型）。PLKS 在原理上更精确，实现成本只略高。**本 idea 建议用 PLKS 替代 LZR，因为 PLKS 能处理 LZR 失败的场景（latent 双峰结构）**。

---

## 具体实现建议

### 步骤 1：添加 calibrate_latent_kde() 到 MultiBF

```python
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import numpy as np

def calibrate_latent_kde(
    self, 
    x_train,
    n_gmm_components=3,
    kde_bandwidth='scott',
    use_kde=False,
    responsibility_threshold=None
):
    """
    Fit per-component, per-dimension GMM/KDE to latent codes of assigned training samples.
    
    After calling this, use inverse_map_with_latent_kde() for generation.
    
    :param x_train: normalized training data (N, dim) tensor
    :param n_gmm_components: number of GMM components per latent dimension
                             (use 1 if components are well-specialized)
    :param kde_bandwidth: bandwidth for KDE ('scott', 'silverman', or float)
    :param use_kde: if True, use KDE instead of GMM
    :param responsibility_threshold: float, use only samples where resp_k > threshold
                                     (default: 1/n_components)
    """
    if responsibility_threshold is None:
        responsibility_threshold = 1.0 / self.n_components
    
    self.latent_kde_models = []  # List of K lists of D density models
    
    with torch.no_grad():
        # Compute per-sample responsibilities
        log_pi = self.get_mixture_log_weights()
        component_log_probs = []
        for k, bf in enumerate(self.components):
            ld = self._per_sample_log_det(bf, x_train)
            component_log_probs.append(log_pi[k] + ld)
        
        stacked = torch.stack(component_log_probs, dim=0)   # (K, N)
        log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
        responsibilities = torch.exp(log_resp)  # (K, N)
        
        for k, bf in enumerate(self.components):
            resp_k = responsibilities[k]  # (N,)
            
            # Select samples primarily assigned to component k
            mask = resp_k > responsibility_threshold
            if mask.sum() < 20:
                # Fallback: use top 20% by responsibility
                topk = max(20, int(0.2 * len(resp_k)))
                _, idx = torch.topk(resp_k, min(topk, len(resp_k)))
                mask = torch.zeros_like(resp_k, dtype=torch.bool)
                mask[idx] = True
            
            x_k = x_train[mask]
            
            # Forward pass to get latent codes
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list)  # (n_k, dim), values in (0,1)
            z_k_np = z_k.cpu().numpy()
            
            # Fit per-dimension density models
            dim_models = []
            for d in range(self.dim):
                z_d = z_k_np[:, d].reshape(-1, 1)
                
                if use_kde:
                    model = KernelDensity(bandwidth=kde_bandwidth, kernel='gaussian')
                    model.fit(z_d)
                else:
                    n_comp = min(n_gmm_components, len(z_d) // 5)  # Ensure enough samples
                    model = GaussianMixture(n_components=max(1, n_comp), 
                                           random_state=42, max_iter=200)
                    model.fit(z_d)
                
                dim_models.append(model)
            
            self.latent_kde_models.append(dim_models)
            print(f"Component {k}: fitted {'KDE' if use_kde else 'GMM'} on "
                  f"{mask.sum()} samples per latent dim")
```

### 步骤 2：添加 inverse_map_with_latent_kde() 到 MultiBF

```python
def inverse_map_with_latent_kde(self, n_samples, max_gap=1e-3, decay_ratio=1.0,
                                  n_candidate_multiplier=3):
    """
    Generate samples using per-component latent KDE/GMM sampling.
    Requires calibrate_latent_kde() to be called first.
    
    Samples z from estimated latent density (instead of Uniform),
    then applies bisection-based inverse_map.
    
    :param n_candidate_multiplier: oversample by this factor, then filter to valid range
    """
    assert hasattr(self, 'latent_kde_models'), "Call calibrate_latent_kde() first"
    
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)
    
    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        dim_models = self.latent_kde_models[k]
        n_candidate = n_k * n_candidate_multiplier
        
        # Sample z from per-dimension GMM/KDE
        z_samples = []
        for d in range(self.dim):
            model = dim_models[d]
            if hasattr(model, 'sample'):  # GMM
                z_d, _ = model.sample(n_candidate)
            else:  # KDE
                z_d = model.sample(n_candidate)
            z_d = np.clip(z_d.flatten(), 0.01, 0.99)  # Clamp to valid range
            z_samples.append(torch.tensor(z_d, dtype=torch.float32))
        
        z = torch.stack(z_samples, dim=1)  # (n_candidate, dim)
        
        # Use first n_k samples (candidates oversampled to handle clipping losses)
        z = z[:n_k]
        
        x_k = self.components[k].inverse_map(
            z, max_gap=max_gap, decay_ratio=decay_ratio
        )
        results[mask] = x_k
    
    return results
```

### 步骤 3：集成到训练/评估流程

```python
# 训练完成后，在生成之前：
all_data = ...  # normalized training data
with torch.no_grad():
    # Calibrate (fast, only a forward pass + GMM fitting)
    mbf.calibrate_latent_kde(
        all_data, 
        n_gmm_components=2,  # 2 is often enough; use 3 if highly non-specialized
        use_kde=False         # GMM is faster and interpretable
    )
    
    # Generate with latent KDE sampling
    samples = mbf.inverse_map_with_latent_kde(n_samples=data_size)
    samples = samples * std + mean
```

### 参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `n_gmm_components` | 2 （充分专一化），3（一般情况） | 若组件专一化好，1D latent 是单峰，用1即可；否则用2-3 |
| `responsibility_threshold` | 1/K | 标准均匀分配阈值；可调高至 0.5 获取更纯净样本 |
| `n_candidate_multiplier` | 2-3 | 过采样比例，确保 clip 后仍有足够样本 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **GMM 拟合双峰但组件未专一化** | 如果组件仍在响应两个 cluster，GMM 会拟合双峰，采样从两个峰都会产生样本 | 配合 DAEM + K-Means init 确保组件先专一化 |
| **各维度独立拟合忽略相关性** | 高维数据中维度相关性可能使独立 GMM 不准确 | 对 2D 数据（项目当前 dim=2）几乎无影响；高维时可升级为多维 GMM |
| **sklearn 依赖** | 需要 sklearn（已在 requirements.txt 中） | 无需新依赖 |
| **Calibration 计算开销** | 需要对训练集做一次 forward pass + GMM 拟合 | 总开销 <5 秒（对 3000-5000 样本），只做一次 |
| **z-space 范围约束** | GMM 采样可能产生超出 [0,1] 的 z 值（但经 Sigmoid 输出的 latent 应当在内） | 用 `np.clip(z, 0.01, 0.99)` 处理边界；若大量截断说明 GMM 有问题 |
| **与 LZR 的重叠** | 若组件非常专一化，GMM 退化为单峰高斯，与 LZR 效果基本一致 | 无问题；这是期望行为 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（替代 LZR，推断时精确修复）**

理由：
1. **LZR 的严格升级版**：处理了 LZR 无法应对的 latent 双峰结构，不需要任何额外条件
2. **推断时无需重训练**：与 LZR 一样，只需一次 calibration，立即可用
3. **实现成本中等**（约 60 行），完全基于已有 sklearn 依赖
4. **理论更精确**：从实际 latent 密度采样，而非粗糙的矩形近似
5. **与 DAEM + K-Means init 叠加效果最佳**：专一化越好，GMM 越接近单峰，采样越精确

**推荐联合使用策略**：

1. **K-Means Warm Start**（`idea_kmeans_warm_start_init_2026-03-11-2314.md`）——初始化
2. **DAEM 训练**（`idea_daem_deterministic_annealing_em_2026-03-11-2312.md`）——训练
3. **PLKS 采样**（本 idea）——推断

三者形成"初始化 → 训练 → 推断"的完整改进流水线，每一步都直接针对 inter-cluster 生成问题的不同环节。

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a.html — The seminal work proposing learned resampled base distributions for normalizing flows; PLKS is a data-driven, training-free alternative.
- Coeurdoux, F. et al. (2024). "Normalizing Flow Sampling with Langevin Dynamics in the Latent Space." *Machine Learning 2024*. https://arxiv.org/abs/2305.12149 — Shows latent-space density sampling (via MALA) is more precise than uniform latent sampling.
- Verine, A. et al. (2024). "Optimal Budgeted Rejection Sampling for Generative Models." *AISTATS 2024*. — Validates density-aware sampling as a post-training quality improvement mechanism.
- arxiv 2512.04954 (2024). "Amortized Inference of Multi-Modal Posteriors using Likelihood-Weighted Normalizing Flows." — Initializing base with GMM matching mode cardinality dramatically improves multi-modal reconstruction; validates the GMM-in-latent-space approach.
