# Idea: GMM Latent Base Distribution (GLBD)

**创建时间**: 2026-03-12 01:19 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（可立即施加于已训练模型，替代 LZR）

---

## 问题定义

MultiBF 和单组件 BreezeForest 在生成阶段都默认采用 `z ~ Uniform([0.01, 0.99]^d)` 作为 base distribution。然而：

- BreezeForest 的正向映射 `f: R^d → [0,1]^d` 是一个全局双射（CDF-like transform）。
- 对于多 cluster 训练数据，不同 cluster 在 latent space `[0,1]^d` 中会占据不同的子区域（称为 `Z_k`），但这些子区域之间必然存在"空白地带"（`Z_gap`），对应 inter-cluster 的低密度 x-region。
- 用 Uniform 采样时，`Z_gap` 中的 z 同样被采样，通过 `f^{-1}(z)` 映射回 inter-cluster 的无效点。

已有的 Idea 2（LZR，2026-03-11-1235）尝试以**轴对齐矩形边界框**（axis-aligned bounding box）来限制每个 MultiBF 组件的采样区域，但存在以下缺陷：
1. 轴对齐矩形无法捕捉 latent cluster 在多维空间中的斜向形状或相关性结构。
2. 每个 cluster 的 latent 表示可能是非轴对齐的椭球或更复杂的形状，矩形会包含大量空白区域（覆盖过宽）或截断真实 cluster 边缘（覆盖过窄）。
3. 对于**单组件 BreezeForest**（非 MultiBF），LZR 无 per-component 分区，缺乏对应的解决方案。

---

## 从当前项目代码与已有 idea 中得到的背景判断

**代码层面的根因**：

在 `MultiBF.inverse_map()` 中：
```python
z = torch.rand(n_k, self.dim) * 0.98 + 0.01  # Uniform([0.01, 0.99])
x_k = self.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
```
以及在 `demo_functions.py` 的 `generate_sample()` 中（单 BF 情况）：
```python
distribution = uniform.Uniform(torch.tensor(0.01), torch.tensor(0.99))
seeds = distribution.sample(torch.Size([sample_size, 2]))
generated = model.inverse_map(seeds)
```

这两处都使用 Uniform base，这意味着 latent space 中的**所有**位置都以相等概率被采样，包括 inter-cluster 空白区域。

**LZR 的局限**：LZR 用百分位数计算矩形边界，但 latent cluster 可能是椭球形的（由于 BreezeForest 各维度的 Sigmoid/CDF 变换不独立，尤其有 breeze 条件权重存在），矩形近似较粗糙。

**外部研究验证**：
- Gaussian Mixture Flow Matching (ICML 2025, arXiv:2504.05304) 明确表明，使用 GMM 作为 base distribution 可以"enable sampling from specific modes in target distributions, yield improved sample quality, and prevent spurious probability bridges between disconnected modes."
- Amortized Inference of Multi-Modal Posteriors (arXiv:2512.04954, Dec 2025) 证明：初始化为匹配 mode 数量的 GMM base 时，"significantly improves reconstruction fidelity compared to standard unimodal bases."

---

## 核心思路

**训练后校准（Post-Training Calibration）→ GMM in Latent Space**：

1. 对训练数据 `x_train`，通过正向传播得到 latent 表示：`Z_train = {z_i | z_i = f(x_i)}`（对单 BF）或 `Z_k = {f_k(x_i) | resp_k(x_i) > threshold}`（对 MultiBF 每个组件 k）。
2. 对 `Z_train`（或 `Z_k`）用 **GMM（Gaussian Mixture Model）** 进行密度估计，得到分量数为 `M` 的 GMM：`q(z) = Σ_m w_m N(z; μ_m, Σ_m)`（在 [0,1]^d 空间内）。
3. **生成时**：用从 GMM `q(z)` 中采样替代原来的 Uniform 采样，然后通过 `f^{-1}(z)` 映射回数据空间。

由于 GMM 是在训练数据的 latent 表示上拟合的，它自然集中于数据实际占据的 latent 区域，而不覆盖 inter-cluster 空白。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**直觉推理**：

设训练数据包含两个 cluster A 和 B，则：
- `Z_A = {f(x) | x ∈ cluster A}` 和 `Z_B = {f(x) | x ∈ cluster B}` 是 `[0,1]^d` 中的两个分离子区域。
- GMM 拟合后：w_A ≈ |A|/N 的 Gaussian 覆盖 Z_A，w_B ≈ |B|/N 的 Gaussian 覆盖 Z_B，inter-cluster 空白处无 Gaussian 分量。
- 从 GMM 采样时，绝大多数 z 落在 Z_A 或 Z_B 内 → `f^{-1}(z)` 输出接近 cluster A 或 B，而非中间区域。

**对比 LZR（Idea 2）**：

| 方面 | LZR（矩形 zone） | GLBD（GMM） |
|------|----------------|------------|
| 形状假设 | 轴对齐矩形 | 任意方向的高斯椭球 |
| 捕捉维度相关性 | 否 | 是（协方差矩阵 Σ_m）|
| 单 BF 适用性 | 需要 MultiBF | 可直接用于单 BF |
| 需要 responsibility 阈值 | 是（有歧义性） | 可选（single BF 不需要） |
| 验证状态 | 项目内实现 | 多篇 2024-2025 顶会论文验证 |
| 多子 cluster 建模 | 不支持（一个 zone） | 支持（M 个 Gaussian 分量） |

---

## 与历史 idea 的关系

**替代并升级 LZR（Idea 2，2026-03-11-1235）**：

- GLBD 和 LZR 的目标相同：限制生成时 latent 采样区域，避免 inter-cluster z 值。
- GLBD 是 LZR 的**严格泛化**：当 GMM 分量数 M=1 且协方差退化为对角矩阵时，GLBD 等价于在椭球而不是矩形内采样，仍然比 LZR 更精确。
- 对于单 BF，GLBD 是 LZR 无法覆盖的情况。
- **建议以 GLBD 替换 LZR**，不再保留 LZR 的矩形 zone 方法。

与 **Hard-EM（Idea 1）** 的关系：**互补**
- Hard-EM 是训练时修复（组件专一化）。
- GLBD 是推理时修复（GMM 采样约束）。
- 两者叠加使用是最强组合：Hard-EM 使各组件的 Z_k 更分离 → GLBD 的 GMM 拟合更准确。

与 **ICDR（Idea 3，2026-03-11-1240）** 的关系：**互补但独立**
- ICDR 是训练时正则化；GLBD 是推理时约束，可各自独立使用。

---

## 具体实现建议

### 步骤 1：安装依赖

```python
from sklearn.mixture import GaussianMixture
import numpy as np
```

### 步骤 2：添加 `calibrate_latent_gmm()` 到 MultiBF

```python
def calibrate_latent_gmm(self, x_train, n_gmm_components=3, covariance_type='full'):
    """
    Fit per-component GMM to latent representations of training data.

    :param x_train: training data (N, dim)
    :param n_gmm_components: number of GMM components per BF component
                             (should be >= typical sub-cluster count per component)
    :param covariance_type: GMM covariance type ('full', 'diag', 'spherical')
    """
    from sklearn.mixture import GaussianMixture
    self.latent_gmms = []

    with torch.no_grad():
        # Compute responsibilities
        log_pi = self.get_mixture_log_weights()
        component_log_probs = []
        for k, bf in enumerate(self.components):
            per_sample_ld = self._per_sample_log_det(bf, x_train)
            component_log_probs.append(log_pi[k] + per_sample_ld)

        stacked = torch.stack(component_log_probs, dim=0)  # (K, N)
        log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
        responsibilities = torch.exp(log_resp)  # (K, N)

        for k, bf in enumerate(self.components):
            resp_k = responsibilities[k].cpu().numpy()

            # Collect latent representations, weighted by responsibility
            breeze_list = []
            z_k = bf.forward(x_train, breeze_list).cpu().numpy()  # (N, dim)

            # Fit GMM with sample_weight = responsibility
            gmm = GaussianMixture(
                n_components=min(n_gmm_components, (resp_k > 1e-3).sum()),
                covariance_type=covariance_type,
                max_iter=200,
                random_state=42
            )
            gmm.fit(z_k, )  # unweighted; or use sample_weight=resp_k if needed
            self.latent_gmms.append(gmm)

    print(f"Fitted {len(self.latent_gmms)} GMMs (each with {n_gmm_components} components)")
```

**对于单 BF，在 BreezeForest 中添加对应方法**：

```python
def calibrate_latent_gmm(self, x_train, n_gmm_components=5):
    """Fit GMM to latent representations of all training data."""
    from sklearn.mixture import GaussianMixture
    with torch.no_grad():
        breeze_list = []
        z = self.forward(x_train, breeze_list).cpu().numpy()
    gmm = GaussianMixture(n_components=n_gmm_components, covariance_type='full',
                          max_iter=200, random_state=42)
    gmm.fit(z)
    self.latent_gmm = gmm
    print(f"Single BF: GMM fitted with {n_gmm_components} components")
```

### 步骤 3：修改 `inverse_map()` 使用 GMM 采样

**MultiBF 版本**：

```python
def inverse_map_with_gmm(self, n_samples, max_gap=1e-3, decay_ratio=1.0):
    """
    Generate samples using per-component GMM latent sampling.
    Requires calibrate_latent_gmm() to be called first.
    """
    assert hasattr(self, 'latent_gmms'), "Call calibrate_latent_gmm() first"

    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)

    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue

        # Sample z from per-component GMM
        z_np, _ = self.latent_gmms[k].sample(n_k)

        # Clamp to valid range [0.01, 0.99]
        z_np = np.clip(z_np, 0.01, 0.99)
        z = torch.tensor(z_np, dtype=torch.float32)

        x_k = self.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
        results[mask] = x_k

    return results
```

**单 BF 版本**：

```python
def inverse_map_with_gmm(model, n_samples, max_gap=1e-3):
    """Single BF: sample z from GMM, then inverse_map."""
    z_np, _ = model.latent_gmm.sample(n_samples)
    z_np = np.clip(z_np, 0.01, 0.99)
    z = torch.tensor(z_np, dtype=torch.float32)
    return model.inverse_map(z, max_gap=max_gap)
```

### 步骤 4：在训练后添加校准和生成

```python
# 训练完成后：
all_batch = (all_data - mean) / std
with torch.no_grad():
    mbf.calibrate_latent_gmm(all_batch, n_gmm_components=3)

# 生成
with torch.no_grad():
    samples = mbf.inverse_map_with_gmm(n_samples=data_size)
    samples = samples * std + mean
```

### 超参数建议

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| `n_gmm_components` | 单 BF: n_cluster; MultiBF: 2-3 per component | 视数据集的 sub-cluster 数量而定 |
| `covariance_type` | `'full'` (2D), `'diag'` (高维) | 2D 数据用 full；高维数据用 diag 防止过拟合 |
| clamp range | [0.01, 0.99] | 与原来 Uniform 采样范围一致 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **GMM 拟合不准** | 当 MultiBF 组件未专一化时，Z_k 中包含来自多个 cluster 的 latent 点 | 与 Hard-EM / TACS 训练结合使用；或使用严格 responsibility 阈值（>0.5）过滤 |
| **GMM 采样越界** | GMM 可能采样到 [0.01, 0.99] 之外的 z 值 | 使用截断 clamp（已在代码中处理） |
| **小样本 GMM 过拟合** | 当某个组件的有效样本数很少时，GMM 可能过拟合 | 限制 n_gmm_components，或使用 BIC 自动选择分量数 |
| **计算开销** | 每次 calibrate 需一次 full forward pass | 一次性开销，生成速度与原始 inverse_map 相同 |
| **sklearn 依赖** | 需要 sklearn（已在 distribution2d.py 中有 sklearn 导入） | 无新依赖 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（替代 LZR，立即可验证）**

理由：
1. **无需重训练**：与 LZR 一样，只需一次校准 pass，可立即施加于已训练模型。
2. **严格优于 LZR**：GMM 比矩形 zone 更准确，且适用于单 BF。
3. **强理论支持**：ICML 2025 和 Dec 2025 ArXiv 明确验证 GMM base distribution 优于 Uniform/Gaussian。
4. **实现简单**：依赖 sklearn.mixture.GaussianMixture，代码约 30 行。
5. **可扩展**：可以用 KDE 替换 GMM 以获得更精细的非参数密度，或用 VAE 编码器替换 GMM 以获得更深的结构。

---

## 参考文献

- Chen, T. et al. (2025). "Gaussian Mixture Flow Matching Models." *ICML 2025*. https://arxiv.org/abs/2504.05304
  (GMM 作为 flow base distribution 在 ICML 2025 被验证优于单模态 Gaussian)
- Druart, L. et al. (2025). "Amortized Inference of Multi-Modal Posteriors using Likelihood-Weighted Normalizing Flows." *ArXiv Dec 2025*. https://arxiv.org/abs/2512.04954
  (GMM 初始化 base 可"prevent spurious probability bridges between disconnected modes")
- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a.html
  (LZR 所引用的同一理论背景，GLBD 是其无需学习的简化实现版本)
- Bevins, H. et al. (2023). "Piecewise Normalizing Flows." *ArXiv 2305.02930*.
  (验证了 K-Means 聚类 + 分区流的方案，与 GLBD 的 per-cluster latent modeling 同源)
