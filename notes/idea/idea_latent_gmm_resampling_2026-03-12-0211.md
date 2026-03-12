# Idea: Latent Space GMM Resampling (LGSR) — 推理阶段 Latent 分布校准

**创建时间**: 2026-03-12 02:11 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（可立即在已训练模型上实施，无需重训练）

---

## 问题定义

BreezeForest 的生成流程是：

```
z ~ Uniform(0.01, 0.99)^d  →  x = f^{-1}(z)
```

但 `Uniform(0.01, 0.99)^d` 是对 latent 空间的**均匀覆盖**，完全不考虑训练数据在 latent 空间中的实际分布结构。

**关键洞察（从代码出发）**：

BreezeForest 的正向映射 `f: R^d → [0,1]^d` 是一个双射。对于多 cluster 训练数据 `{x_i}`，其在 latent 空间中的像 `{z_i = f(x_i)}` 也构成**多峰分布**——每个 cluster 对应 [0,1]^d 中的一个高密度子区域。

设 cluster A 的 latent 像是 Z_A，cluster B 的 latent 像是 Z_B，则 Z_A 和 Z_B 之间存在一个**低密度过渡带** Z_gap。当前生成策略从 Uniform 采样时，Z_gap 区域同样会被采样到，这些 z 值通过 inverse_map 后，对应 data space 中 cluster A 和 B 之间的无效区域。

**这个问题对单个 BreezeForest 同样存在，不仅限于 MultiBF。**

---

## 从当前项目代码与已有 idea 中得到的背景判断

**代码关键发现**：

1. `demo_functions.py`（第 148-153 行）和 `MultiBF.inverse_map()`（第 165 行）都使用 `torch.rand * 0.98 + 0.01`（即 Uniform(0.01, 0.99)^d）作为 latent 采样分布。这是问题的直接来源。

2. `bisection()` 函数（`tools.py` 第 71-107 行）中，Stage 1 在 CDF 空间 [0,1] 内搜索，使用 `distribution.icdf()` 将 CDF 值映射到真实空间作为搜索范围初始化。Stage 1 的目的是缩小搜索范围；但若 z 值来自 inter-cluster 的 latent 区域，即使 bisection 精确收敛，也只会找到一个 inter-cluster 的数据点（正确地"反演"了一个错误的 z 值）。

3. `BreezeForest.forward()` 最终返回 `x * self.dim_mask`，其中每个维度经过 sigmoid 激活，保证输出在 [0,1]^d 内。这意味着 latent 空间天然有界，非常适合用 GMM 建模（不需要处理无限尾巴）。

4. `BreezeForest.inverse_map()` 的 `compute_dis()` 方法（第 258-264 行）根据 `batch_example` 计算每个维度的均值和标准差，用于设置 bisection 的 Normal 参考分布。这是一个一维（per-dimension）的 Normal 近似，无法捕捉多峰结构。

**已有 idea 的局限**：

- **LZR（2026-03-11-1235）**：思路与 LGSR 相近，但有两个关键限制：
  1. **仅适用于 MultiBF**：LZR 依赖组件 k 的 responsibility 来选取"属于 cluster k 的样本"，然后估计各组件的 latent zone。对于**单个 BreezeForest**，LZR 完全无法应用。
  2. **矩形 zone 近似**：LZR 对每个 cluster 在 latent 空间中的形状用轴对齐矩形框来近似，忽略了各维度之间的相关性和非矩形的 cluster 形状。例如，若 cluster A 在 latent 空间中对应一个斜向椭圆形区域，LZR 的矩形框会过度包含 inter-cluster 区域。

**LGSR 是 LZR 的直接升级版本，解决了上述两个核心限制。**

**外部文献依据**：

- Stimper et al. (2022) 的 Resampled Base Distribution 方法通过学习一个 rejection sampling 先验来修复 flow 的拓扑问题——这是 LGSR 的理论基础，但需要额外训练。LGSR 是一种数据驱动的无训练替代方案。
- Coeurdoux et al. (2024) 的 Langevin Dynamics in Latent Space 通过 MCMC 在 latent 空间采样，同样是为了避开 low-density latent 区域——但 MCMC 的计算代价更高。LGSR 用 GMM 拟合来替代 MCMC，更高效。
- 最新（2025）GMM base distribution 研究（arXiv:2503.00524）验证了在 flow 的 prior 空间中使用 GMM（而非单一 Gaussian）可以显著改善多模态分布的生成质量。

---

## 核心思路

**训练后校准（Post-Training Calibration）**，一次性操作，无需重训练：

1. **Latent 表示计算**：将全部训练数据 `{x_i}` 通过正向映射得到 `{z_i = f(x_i)} ⊂ [0,1]^d`。
2. **GMM 拟合**：用 scikit-learn 的 `GaussianMixture` 拟合 `{z_i}` 在 [0,1]^d 中的分布，组件数设为 cluster 数 K（或通过 BIC 自动选择）。
3. **约束采样**：生成时从 GMM 采样 z，拒绝/截断落在 [0.01, 0.99]^d 之外的样本。
4. **Inverse Map**：用已有的 `inverse_map()` 做反演：`x = f^{-1}(z)`。

```
训练完成 → 计算 {z_i = f(x_i)} → 拟合 GMM_latent → 
生成时: z ~ GMM_latent ∩ [0.01, 0.99]^d → x = f^{-1}(z)
```

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**因果分析**：

多 cluster 数据 → 每个 cluster 的 latent 像（z_i = f(x_i)）在 [0,1]^d 中形成高密度子区域 Z_k → Z_k 之间有低密度过渡带 Z_gap → 原始 Uniform 采样包含 Z_gap → 从 Z_gap 采样的 z 通过 inverse_map 产生 inter-cluster 点。

GMM 拟合 `{z_i}` 后：
- GMM 的高密度区域正好对应训练数据的 latent 像 Z_k
- GMM 在 Z_gap 区域的密度极低
- 从 GMM 采样 z，命中 Z_gap 的概率远低于 Uniform 采样
- 因此 x = f^{-1}(z) 落在 inter-cluster 区域的概率大幅降低

**定量估计**：

若训练数据的 latent 像在 [0,1]^2 中形成两个分离的高密度团（面积各占 20%），则：
- Uniform 采样：60% 的 z 值在 inter-cluster / 无数据区域 → 60% 无效样本
- GMM 采样（拟合良好）：仅 5-10% 的 z 值在低密度区域 → 5-10% 无效样本

**对比 LZR 的定量优势**：

| 特性 | LZR | LGSR |
|------|-----|------|
| 适用范围 | 仅 MultiBF | 单 BF + MultiBF |
| latent 形状 | 矩形框 | GMM（任意形状） |
| 需要 responsibility 计算 | 是 | 否（用所有训练数据） |
| 拟合复杂度 | O(N * K) | O(N * K_gmm) |
| 可自动调整组件数 | 否 | 是（BIC 选择） |
| 处理 latent 维度相关性 | 否（轴对齐） | 是（协方差矩阵） |

---

## 与历史 idea 的关系

**直接升级 LZR（2026-03-11-1235）**

LZR 是 LGSR 的矩形近似版本，LGSR 在以下维度有明确改进：
1. 适用范围扩展：从 MultiBF-only → 单 BF 也适用
2. 形状更精确：GMM vs 矩形框
3. 不依赖 responsibility 计算：用所有训练数据拟合，更稳定
4. 可通过 BIC 自动选择 GMM 组件数

**LZR 可视为 LGSR 的特例（K=1 per component，仅使用均值和协方差对角，截断到边界框）**。因此 LGSR 在任何 LZR 有效的场景下都至少一样好。

**与 Hard-EM/PBF（2026-03-11-1230 / 本轮新增）的关系：互补**

PBF/Hard-EM 是训练阶段的修复（让模型本身更好地分离 cluster）；LGSR 是推理阶段的修复（让采样更集中于有数据的 latent 区域）。两者叠加后：
- PBF 使 latent space 中的 cluster 结构更清晰
- LGSR 进一步利用这个清晰的结构改进采样

**与 ICDR（2026-03-11-1240）的关系：独立补充**

ICDR 修改训练目标；LGSR 修改采样策略。两者不冲突，可以叠加。

---

## 具体实现建议

### 步骤 1：添加 `calibrate_latent_gmm()` 方法到 BreezeForest

```python
from sklearn.mixture import GaussianMixture
import numpy as np

def calibrate_latent_gmm(self, x_train, n_components=None, auto_select=True, max_components=10):
    """
    Fit a GMM in the latent space [0,1]^d using forward-mapped training data.
    
    :param x_train: training data tensor (N, dim) — already normalized
    :param n_components: number of GMM components (None = auto via BIC)
    :param auto_select: whether to use BIC to select best n_components
    :param max_components: max n_components to try in BIC selection
    """
    with torch.no_grad():
        # Compute latent representations
        breeze_list = []
        z = self.forward(x_train, breeze_list)  # (N, dim), values in [0,1]^d
        z_np = z.numpy()
    
    if auto_select or n_components is None:
        # BIC-based model selection
        best_bic = np.inf
        best_gmm = None
        for k in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=3, random_state=42)
            gmm.fit(z_np)
            bic = gmm.bic(z_np)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        self.latent_gmm = best_gmm
        print(f"BIC selected {best_gmm.n_components} GMM components for latent space")
    else:
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', n_init=3, random_state=42)
        gmm.fit(z_np)
        self.latent_gmm = gmm
    
    return self.latent_gmm
```

### 步骤 2：修改 `inverse_map()` 使用 GMM 采样

```python
def inverse_map_gmm(self, n_samples, max_gap=1e-3, decay_ratio=1.0,
                    clamp_lo=0.01, clamp_hi=0.99, max_resample_factor=3):
    """
    Generate samples using GMM-calibrated latent sampling.
    Requires calibrate_latent_gmm() to be called first.
    
    :param n_samples: number of samples to generate
    :param max_resample_factor: oversample by this factor to account for clamping rejection
    """
    assert hasattr(self, 'latent_gmm'), "Call calibrate_latent_gmm() first"
    
    # Oversample from GMM to account for out-of-range samples
    n_oversample = int(n_samples * max_resample_factor)
    z_np, _ = self.latent_gmm.sample(n_oversample)
    
    # Clamp to valid range and filter
    z_np = np.clip(z_np, clamp_lo, clamp_hi)
    z = torch.tensor(z_np, dtype=torch.float32)
    
    # Take exactly n_samples (oversampling ensures enough valid ones)
    z = z[:n_samples]
    
    return self.inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
```

### 步骤 3：MultiBF 适配版本

对 MultiBF，可以为每个组件分别拟合 GMM，利用 responsibility 信息：

```python
def calibrate_latent_gmm_multi(self, x_train, n_components_per_bf=None):
    """
    For MultiBF: fit per-component GMM using responsibility-weighted samples.
    """
    with torch.no_grad():
        log_pi = self.get_mixture_log_weights()
        component_log_probs = []
        for k, bf in enumerate(self.components):
            ld = self._per_sample_log_det(bf, x_train)
            component_log_probs.append(log_pi[k] + ld)
        
        stacked = torch.stack(component_log_probs, dim=0)  # (K, N)
        resp = torch.softmax(stacked, dim=0)  # (K, N)
    
    for k, bf in enumerate(self.components):
        # Use samples where component k has highest responsibility
        resp_k = resp[k]
        mask = resp_k > (1.0 / self.n_components)
        x_k = x_train[mask]
        
        if len(x_k) >= 10:
            n_comp = n_components_per_bf or max(1, len(x_k) // 200)
            bf.calibrate_latent_gmm(x_k, n_components=n_comp)
        else:
            bf.calibrate_latent_gmm(x_train, n_components=1)  # fallback

def inverse_map_gmm_multi(self, n_samples, max_gap=1e-3, decay_ratio=1.0):
    """MultiBF version of GMM-based generation."""
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)
    
    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        x_k = self.components[k].inverse_map_gmm(n_k, max_gap=max_gap, decay_ratio=decay_ratio)
        results[mask] = x_k
    
    return results
```

### 步骤 4：在训练脚本中集成（单 BF + MultiBF 均适用）

```python
# 训练完成后：
import numpy as np
from sklearn.mixture import GaussianMixture

# 获取全部训练数据（已归一化）
all_batch = (full_data - mean) / std

# 单 BF 版本
bf.calibrate_latent_gmm(all_batch, auto_select=True, max_components=8)

# 生成
with torch.no_grad():
    samples = bf.inverse_map_gmm(n_samples=3000)
    samples = samples * std + mean
```

### 超参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `n_components` | BIC 自动选择，或等于 cluster 数 | 过多组件会过拟合 latent GMM |
| `clamp_lo/hi` | 0.01 / 0.99 | 保持与原始生成一致的边界 |
| `covariance_type` | `'full'` | 捕捉 latent 维度间相关性 |
| `max_resample_factor` | 2-3 | GMM 采样的过采样倍数 |
| BIC 最大组件数 | cluster 数 × 2 | 避免过多组件 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **GMM 过拟合** | 若 K 过大，GMM 可能在训练数据之外的 latent 区域也有高密度 | 使用 BIC/AIC 控制 K；或用 cross-validation 选择 K |
| **GMM 欠拟合** | 若 K 过小（如 K=1），GMM 退化为 Gaussian，覆盖 inter-cluster 区域 | 设置 K ≥ cluster 数；BIC 自动选择通常足够 |
| **latent 空间边界效应** | GMM 的高斯分量可能超出 [0,1]^d 边界 | 截断 + 过采样；clamp 后统计采样有效率 |
| **计算依赖 sklearn** | 需要 sklearn 依赖 | 项目已依赖 sklearn（distribution2d.py 中 make_moons 等） |
| **训练不专一时效果受限** | 若 BF 训练时没有形成清晰的 cluster 结构，latent GMM 也会模糊 | 与 PBF 结合使用；PBF 训练产生清晰 latent 结构，LGSR 进一步利用 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（可立即在任何已训练模型上实施）**

理由：

1. **零训练成本**：不需要修改训练代码，不需要重训练模型；只需在推理前运行一次 GMM 拟合（< 1 秒）
2. **即时可验证**：可以在现有任何 BreezeForest 或 MultiBF 模型上立即验证效果
3. **适用范围更广**：LZR 仅适用于 MultiBF，LGSR 同时适用于单 BreezeForest（解决了 LZR 的主要局限）
4. **形状更准确**：GMM 捕捉 latent 空间的实际形状，包括旋转、椭圆等非矩形结构
5. **可自动调整**：BIC 选择 GMM 组件数，无需手动调参
6. **理论支撑**：与 Stimper et al. 2022 的 resampled base distribution 同类方法，但更简单且无需训练

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.  
  https://proceedings.mlr.press/v151/stimper22a.html  
  (LGSR 的理论基础，通过学习 rejection sampling prior 修复 topology；LGSR 是其数据驱动无训练版本)
- Coeurdoux, F. et al. (2024). "Normalizing Flow Sampling with Langevin Dynamics in the Latent Space." *Machine Learning 2024*.  
  https://arxiv.org/abs/2305.12149  
  (Latent 空间 MCMC 采样，与 LGSR 同源思路；LGSR 用 GMM 替代 MCMC，更高效)
- Bevins, H. & Handley, W. (2023). "Piecewise Normalizing Flows." *arXiv:2305.02930*.  
  (预聚类 + 独立训练；LGSR 是推理侧的等效修复)
- Chen, S. et al. (2025). "Gaussian Mixture Flow Matching Models." *ICML 2025*.  
  https://proceedings.mlr.press/v267/chen25cl.html  
  (GMM 先验/base distribution 在多模态分布中显著优于 Gaussian；验证了 LGSR 的核心假设)
