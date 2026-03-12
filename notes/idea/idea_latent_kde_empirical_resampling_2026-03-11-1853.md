# Idea: Latent-Space Empirical KDE Resampling (EDR)

**创建时间**: 2026-03-11 18:53 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（替代 / 超越 LZR，适用于单 BF 和 MultiBF）

---

## 问题定义

BreezeForest 的生成流程为：
1. z ~ Uniform(0.01, 0.99)^d
2. x = f^{-1}(z)（二分搜索）

根本问题：**Uniform(0.01, 0.99)^d 对 latent 空间中所有 z 值一视同仁，但实际上只有部分 z 值对应真实数据区域，其余 z 值对应 cluster 之间的低密度间隙。**

对于 d=2，z_train = {f(x_i)} 的分布如下：
- 对应 cluster A 的数据：z 值集中在某个区域 Z_A ⊂ [0,1]^2
- 对应 cluster B 的数据：z 值集中在另一个区域 Z_B ⊂ [0,1]^2
- Z_A 和 Z_B 之间存在"间隙"——没有或极少训练数据对应的 z 值
- **当 z ~ Uniform 采样到间隙区域时，f^{-1}(z) 映射到 inter-cluster 的无效 x**

LZR（已有 idea 1235）尝试用矩形框约束每个 MultiBF 组件的 z 范围，方向正确，但存在以下局限：
- 只适用于 MultiBF（不适用于单 BF）
- 矩形框是对实际 Z_k 分布形状的粗糙近似（真实 Z_k 可能是不规则形状）
- 矩形框估计依赖组件的 responsibility，若组件未专一化则 Z_k 估计不准

---

## 从当前项目代码与已有 idea 中得到的背景判断

**代码层面**：
- `BreezeForest.forward(x)` 是高效可微的正向映射 x → z ∈ [0,1]^d
- 正向传播比反向（bisection）快得多（O(1) forward vs O(log(1/ε)) bisection）
- `generate_sample()` 使用 `uniform.Uniform(0.01, 0.99)` 的扁平采样
- `inverse_map()` 中的 `compute_dis()` 使用训练 batch 的均值/方差作为 bisection prior，说明项目已意识到使用数据统计量的价值
- 对于单 BF：latent space 是 [0,1]^d（bounded）；对于 MultiBF：每个组件的 latent space 独立是 [0,1]^d

**已有 idea 分析**：
- LZR（1235）用矩形框约束每个组件的 z 范围，已捕捉到核心洞察，但近似粗糙
- Hard-EM（1230）是 training-time 修复，不解决 inference 时的问题
- ICDR（1240）同样是 training-time，不解决 inference 时的采样问题

**文献验证**：
- Stimper et al. (2022) "Resampling Base Distributions of Normalizing Flows"：通过**学习**一个拒绝采样的 base distribution 来修复 topology 问题。本 Idea 用数据驱动的 KDE 代替学习，无需重训练。
- Amortized Multi-Modal Posteriors（arXiv 2512.04954, 2025）：明确指出"从 z_train 的分布形状采样而不是均匀采样"可显著减少 inter-mode 桥接。
- Langevin dynamics in latent space (Coeurdoux, 2024)：通过 MALA 在 latent space 中采样，也是基于同样洞察：z_train 的高密度区域才是有效生成区域。

---

## 核心思路

**训练后校准（Post-Training Calibration），无需重训练**：

### Step 1：收集 Latent 训练样本

对所有训练数据运行正向传播，得到 latent 表示：
```
z_train = {f(x_i) | x_i ∈ D_train} ⊂ [0,1]^d
```

### Step 2：在 Logit 空间拟合 KDE 或 GMM

由于 z ∈ [0,1]^d 是有界空间，直接在 z 空间做 KDE 会在边界附近产生偏差。改为在 logit 变换后的空间拟合：
```
w_i = logit(z_i) = log(z_i / (1 - z_i))  ∈ ℝ^d
```
在 w 空间拟合 KDE（或 GMM）。

### Step 3：从 KDE 采样

从拟合的 KDE/GMM 中采样 w，还原 z = sigmoid(w)，再做 inverse_map：
```
w ~ KDE(w_train)
z = sigmoid(w).clamp(0.01, 0.99)
x = f^{-1}(z)
```

**关键效果**：
- z_train 在 cluster 间隙区域**稀疏**（因为间隙区域没有训练数据的正向映射结果）
- 从 KDE(w_train) 采样会**自动跳过**间隙区域（KDE 密度在这些区域接近 0）
- 最终生成的 x = f^{-1}(z) 几乎不含 inter-cluster 点

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**直觉论证**：

设训练数据来自两个 cluster A 和 B，正向映射后：
- cluster A 数据 → z_A，密集分布在 [0.1, 0.4]^2（示例）
- cluster B 数据 → z_B，密集分布在 [0.6, 0.9]^2（示例）
- 间隙区域 → [0.4, 0.6]^2 几乎没有 z_train 值

KDE(w_train)（w = logit(z_train)）在对应 [0.4, 0.6]^2 的区域密度接近 0。从 KDE 采样时：
- 约 50% 的样本来自 z_A 区域 → inverse_map 到 cluster A
- 约 50% 的样本来自 z_B 区域 → inverse_map 到 cluster B
- 极少数来自间隙 → inter-cluster 点大幅减少

**与 Uniform 采样的对比**：

| 采样方式 | 间隙 z 被采到的概率 | 结果 |
|---------|------------------|------|
| Uniform(0.01, 0.99)^d | 与 cluster 面积成正比，可能高 | 大量 inter-cluster 点 |
| KDE(z_train) | 接近 0（间隙稀疏）| 极少 inter-cluster 点 |

**适用范围（关键优势）**：
- **单 BF**：直接对单个模型的 z_train 拟合 KDE，无需 MultiBF
- **MultiBF**：对每个组件的 z_train（用 responsibility 筛选）分别拟合 KDE
- 与 LZR 相比：KDE 捕捉实际形状，LZR 只用矩形框

---

## 它与历史 idea 的关系

**替代 / 超越 LZR（1235）**：
- LZR 是本 Idea 的粗糙近似版：矩形框 ≈ KDE 的 bounding box
- 本 Idea 捕捉实际分布形状（非矩形），避免 LZR 的过切割问题
- 本 Idea 适用于单 BF（LZR 只适用于 MultiBF）
- LZR 在 MultiBF 组件未专一化时估计不准；本 Idea 直接从前向传播结果估计，更稳健

**与 Hard-EM（1230）的关系**：互补。本 Idea 是 inference-time 修复，Hard-EM 是 training-time 修复。两者可叠加。

**与 Piecewise BF（本轮 Idea 1）的关系**：
- Piecewise BF 训练完后，每个组件的 z_train 会更加集中（因为组件专一化）
- 在 Piecewise BF 生成时，对每个组件分别用 EDR，可进一步提升生成质量
- 两者叠加是目前最强的组合

**替代关系**：对于单 BF 的 inter-cluster 问题，本 Idea 是当前最直接的修复方案（无需 Hard-EM 等 training-time 修复）。

---

## 具体实现建议

### 步骤 1：收集 z_train

```python
def collect_latent(bf, x_train, batch_size=1000):
    """Run forward pass on training data to collect latent representations."""
    bf.eval()
    z_list = []
    with torch.no_grad():
        for i in range(0, len(x_train), batch_size):
            batch = x_train[i:i+batch_size]
            breeze_list = []
            z_batch = bf.forward(batch, breeze_list)
            z_list.append(z_batch.cpu())
    return torch.cat(z_list, dim=0)  # (N, dim)
```

### 步骤 2：在 Logit 空间拟合 KDE

```python
from sklearn.neighbors import KernelDensity
import numpy as np

def fit_latent_kde(z_train, bandwidth=0.3):
    """
    Fit KDE in logit space.
    z_train: tensor (N, dim), values in (0, 1)
    Returns: sklearn KDE object (fitted in logit space)
    """
    z_np = z_train.clamp(0.01, 0.99).numpy()
    w_np = np.log(z_np / (1 - z_np))  # logit transform -> R^d
    
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(w_np)
    return kde

def fit_latent_gmm(z_train, n_components=None, max_components=8):
    """
    Alternative: fit GMM in logit space (better for well-separated clusters).
    If n_components is None, select by BIC.
    """
    from sklearn.mixture import GaussianMixture
    z_np = z_train.clamp(0.01, 0.99).numpy()
    w_np = np.log(z_np / (1 - z_np))
    
    if n_components is None:
        best_bic = np.inf
        best_gmm = None
        for k in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=k, n_init=3)
            gmm.fit(w_np)
            if gmm.bic(w_np) < best_bic:
                best_bic = gmm.bic(w_np)
                best_gmm = gmm
        return best_gmm
    else:
        gmm = GaussianMixture(n_components=n_components, n_init=5)
        gmm.fit(w_np)
        return gmm
```

### 步骤 3：EDR 生成函数

```python
def generate_edr(bf, kde_or_gmm, n_samples, dim, max_gap=1e-3):
    """
    Generate samples using Empirical Density Resampling (EDR).
    
    :param bf: trained BreezeForest (or per-component BF in Piecewise BF)
    :param kde_or_gmm: fitted KDE or GMM in logit space
    :param n_samples: number of samples to generate
    :param dim: data dimensionality
    """
    # Sample w from KDE/GMM in logit space
    w_samples = kde_or_gmm.sample(n_samples)  # (n_samples, dim)
    
    # Convert back to z ∈ (0.01, 0.99)
    z_samples = torch.sigmoid(torch.tensor(w_samples, dtype=torch.float))
    z_samples = z_samples.clamp(0.01, 0.99)
    
    # Map back to x via bisection
    bf.eval()
    with torch.no_grad():
        x_generated = bf.inverse_map(z_samples, max_gap=max_gap)
    
    return x_generated
```

### 步骤 4：在 demo 中整合（替代原始 generate_sample）

```python
# 训练完成后：
# 1. 收集 latent z_train
z_train = collect_latent(bf, x_train_normalized)

# 2. 拟合 GMM（推荐 GMM over KDE，对多 cluster 数据更准确）
gmm = fit_latent_gmm(z_train, n_components=None)  # BIC 自动选 K

# 3. 生成样本
samples = generate_edr(bf, gmm, n_samples=data_size, dim=2)
samples = samples * std + mean  # 反归一化
```

### 对 MultiBF 的适配

```python
def generate_edr_multi_bf(mbf, x_train_normalized, n_samples):
    """Per-component EDR for MultiBF."""
    from model.MultiBF import MultiBF
    
    # 1. Compute responsibilities
    with torch.no_grad():
        log_pi = mbf.get_mixture_log_weights()
        component_log_probs = []
        for k, bf in enumerate(mbf.components):
            ld = mbf._per_sample_log_det(bf, x_train_normalized)
            component_log_probs.append(log_pi[k] + ld)
        stacked = torch.stack(component_log_probs, dim=0)
        log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
        responsibilities = torch.exp(log_resp)  # (K, N)
    
    # 2. For each component, fit KDE on its latent z (using threshold)
    kdes = []
    for k, bf_k in enumerate(mbf.components):
        threshold = 1.0 / mbf.n_components
        mask = responsibilities[k] > threshold
        x_k = x_train_normalized[mask]
        
        with torch.no_grad():
            z_k = bf_k.forward(x_k, [])
        kde_k = fit_latent_gmm(z_k.cpu())
        kdes.append(kde_k)
    
    # 3. Generate
    weights = mbf.get_mixture_weights().detach()
    comp_idx = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, mbf.dim)
    
    for k in range(mbf.n_components):
        mask = (comp_idx == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        x_k = generate_edr(mbf.components[k], kdes[k], n_k, mbf.dim)
        results[mask] = x_k
    
    return results
```

### 超参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| KDE bandwidth | 0.1–0.5（logit 空间）| 用交叉验证（sklearn）自动选；GMM 无需此参数 |
| GMM n_components | BIC 自动选 | 一般等于真实 cluster 数 |
| KDE kernel | 'gaussian' | 最常用，效果好 |
| logit clamp | 0.01 / 0.99 | 防止 logit 趋于无穷大 |

**KDE vs GMM 选择**：
- 数据 cluster 清晰（如 8gaussians）→ GMM（更准确，更快采样）
- 数据分布不规则（如 spirals, moons）→ KDE（更灵活）

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **KDE 带宽选择敏感** | 带宽过大 → 间隙被平滑掉，EDR 效果减弱；带宽过小 → 生成样本过于集中 | 用 sklearn 的交叉验证 `KernelDensity` 或 Scott's rule 自动选 |
| **z_train 量少时 KDE 不准** | 训练数据量少时 z_train 稀疏，KDE 估计有噪声 | 最少 1000 个训练样本；或对 z_train 做 bootstrap 增广 |
| **BF 未很好建模时 z_train 可能有噪声** | 欠拟合的 BF 的 z_train 分布混乱，无法区分 cluster | 先确保训练 loss 收敛，再做 EDR |
| **GMM 组件数估计错误** | BIC 有时低估 GMM 组件数，无法完全分离 cluster 间隙 | 显式设置 n_components = 真实 cluster 数（如已知）|
| **logit 变换在边界处不稳定** | z 值过于接近 0 或 1 时 logit 趋于 ±∞ | 已有 clamp(0.01, 0.99) 防护；实际训练数据的 z 一般不会到达边界 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（同 Piecewise BF 并列，且更快部署）**

理由：
1. **零重训练成本**：只需在已训练模型上运行一次 forward pass + sklearn 拟合，5–10 分钟完成
2. **唯一适用于单 BF 的 inference-time 修复**：LZR 只支持 MultiBF；EDR 直接对单 BF 也有效
3. **直接针对根因**：z_train 的分布稀疏性直接反映了 cluster 间隙，KDE 采样自动规避间隙
4. **超越 LZR**：捕捉实际分布形状，不依赖矩形框近似，不依赖 responsibility 估计
5. **有文献理论基础**：Stimper (2022) 的 resampled base distribution 是其学习版本；本 Idea 是后处理的数据驱动版本，无需重训练

**建议部署顺序**：
1. **先部署 EDR**（本 Idea）：在已有训练模型上立即验证效果
2. **再考虑 Piecewise BF**（本轮 Idea 1）：重训练，获得更根本的修复

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.  
  https://proceedings.mlr.press/v151/stimper22a.html  
  （本 Idea 是其无需重训练的数据驱动版本）
- Coeurdoux, F. et al. (2024). "Normalizing Flow Sampling with Langevin Dynamics in the Latent Space." *Machine Learning, 113*, 8301–8326.  
  https://arxiv.org/abs/2305.12149  
  （同一根洞察：对 latent z-space 进行更智能的采样）
- Marchetti, G.L. et al. (2023). "Piecewise Normalizing Flows." *arXiv 2305.02930*.  
  （确认在 latent 层面分离 cluster 是解决 inter-cluster 生成问题的正确方向）
- Amortized Multi-Modal Posteriors. (2025). *arXiv 2512.04954*.  
  （GMM base 初始化显著改善 multi-modal flow 的重构质量）
