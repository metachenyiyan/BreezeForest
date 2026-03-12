# Idea: GMM Latent Base Distribution — 用高斯混合模型替代均匀分布作为每组件的推理采样基

**创建时间**: 2026-03-12 01:35 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（推理阶段的精确基分布替换，可立即实施无需重训练）

---

## 问题定义

MultiBF 的 `inverse_map()` 在采样时对每个组件 k 执行：

```python
z = torch.rand(n_k, self.dim) * 0.98 + 0.01  # z ~ Uniform([0.01, 0.99]^d)
x_k = self.components[k].inverse_map(z)
```

这里存在一个根本性的**基分布不匹配（Base Distribution Mismatch）**问题：

- BreezeForest 组件 k 的正向映射 $f_k$ 将数据空间中的 **cluster k** 映射到 latent 空间 $[0,1]^d$ 的某个子区域 $Z_k$（cluster k 的 CDF 范围集中的地方）
- 然而当 $f_k$ 训练于所有数据（soft-EM）或至少初始化于全量数据时，**其 latent 空间 $[0,1]^d$ 还容纳了其他 cluster 和 inter-cluster 区域的映射**
- 从 Uniform([0.01, 0.99]^d) 采样，等价于不加区分地从 latent 空间采样——其中包含了 cluster k 对应的 $Z_k$ 和所有其他区域

**LZR（历史 idea 1235）** 已认识到这个问题，并提出用每维独立的百分位数边界 $[a_k^d, b_k^d]$ 估计 $Z_k$。LZR 的局限性：
1. **轴对齐矩形框**：忽略了 $Z_k$ 在不同维度之间的相关性（latent 表示可能是斜向椭圆形）
2. **硬截断边界**：在边界处密度突变为零，可能截断 cluster 边缘的合法样本
3. **对组件混淆的敏感性**：若 soft-EM 训练的组件同时覆盖多个 cluster，$Z_k$ 的百分位估计会包含多个 cluster 的 latent 像，导致边界不准

---

## 从当前项目代码与已有 idea 中得到的背景判断

`BreezeForest.forward()` 通过 Sigmoid 激活（`model/tools.py` 中的 `Sigmoid` 类），将输出压缩到 $(0, 1)^d$。因此 latent 空间天然有界。

`generate_sample()` 在 `demo_functions.py` 中明确使用：
```python
distribution = uniform.Uniform(torch.tensor(0.01), torch.tensor(0.99))
seeds = distribution.sample(torch.Size([sample_size, 2]))
generated = model.inverse_map(seeds)
```

这是全 latent 空间均匀采样，没有任何密度约束。

历史 idea 1235（LZR）的校准方法通过 `torch.quantile` 逐维估计百分位边界，忽略了维度间相关性。其 `calibrate_latent_zones()` 实现中，对每个维度 d 独立计算分位数，这对于相关性强的 latent 分布（如 2D 高斯沿对角线分布的 cluster）会高估 $Z_k$ 的范围。

**本 Idea 的改进目标**：将 LZR 的轴对齐矩形升级为 **per-component GMM（高斯混合模型）latent 基分布**，从而精确捕获 $Z_k$ 的形状和相关结构。

---

## 核心思路

**训练后校准（与 LZR 相同思路，但基分布更精确）**：

1. 对训练完成的 MultiBF，计算每个样本对每个组件的 responsibility
2. 对组件 k，选取 responsibility 最高的训练样本 $\{x_i : r_{ik} > threshold\}$
3. 通过正向映射得到这批样本的 latent 表示：$z_i^k = f_k(x_i)$
4. 对 $\{z_i^k\}$ 拟合一个 GMM（通常 1-3 个成分即可）
5. 将拟合的 GMM 存为组件 k 的 **latent base distribution**

**推理时的采样策略**：

```
z ~ GMM_k  (而非 z ~ Uniform([0.01, 0.99]^d))
z = clip(z, 0.01, 0.99)  (确保在 BreezeForest 的有效 latent 范围内)
x = f_k^{-1}(z)
```

由于 $\text{GMM}_k$ 是从 cluster k 的数据的 latent 像拟合的，采样 $z \sim \text{GMM}_k$ 天然集中在 $Z_k$ 附近，而不在 inter-cluster 的 latent 区域。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**数学严格性**：

设 cluster k 的训练数据 $D_k$ 的 latent 像为 $Z_k = \{f_k(x) : x \in D_k\}$。训练时 $f_k$ 被优化使得 $f_k(D_k)$ 的分布尽量接近 Uniform([0,1]^d)。

因此 $Z_k$ 几乎遍布整个 $[0,1]^d$——但 cluster k 对应的高密度部分（即 $Z_k^{dense}$）会**在 $[0,1]^d$ 内形成一个紧凑的子区域**，对应 $D_k$ 在 CDF 空间的映射（CDF 值密集分布的区域）。

对于 multi-cluster 数据，如果 soft-EM 训练导致 $f_k$ 同时覆盖多个 cluster，$Z_k^{dense}$ 会包含多个斑块。GMM 拟合 $Z_k$ 时，每个 cluster 对应的高密度区域会成为 GMM 的一个成分。推理时从 GMM 采样，采样点集中在高密度区域，避开 inter-cluster 的低密度 latent 区域。

**优势对比（vs LZR, vs Uniform Sampling）**：

| 维度 | Uniform 采样（当前） | LZR（idea 1235） | GMM Latent Base（本 Idea） |
|------|---------------------|-----------------|---------------------------|
| 采样范围 | 全 [0.01, 0.99]^d | 轴对齐矩形框 Z_k | GMM 概率密度下的高密度区域 |
| 维度相关性 | 忽略 | 忽略（逐维独立） | **捕获**（GMM 协方差矩阵）|
| 边界处理 | 均匀截断 | 硬截断 | **软衰减**（高斯尾部自然衰减）|
| 包含 inter-cluster 的 z | 多 | 少（取决于框大小） | **极少**（仅 GMM 尾部泄漏）|
| 是否需要重训练 | 否 | 否 | **否**（校准步骤在训练后运行）|
| 实现复杂度 | 无 | 低 | 低（sklearn.mixture.GaussianMixture）|

**外部文献验证**：

arxiv 2512.04954（Amortized Inference of Multi-Modal Posteriors using Likelihood-Weighted Normalizing Flows, 2024）明确指出：当 normalizing flow 使用 GMM 基分布（cardinality 匹配目标 cluster 数量）时，重建保真度显著提升，spurious bridge 现象基本消除。

BMVC 2024（Multimodal base distributions in conditional flow matching）进一步验证：GMM 基分布在 in-distribution likelihood 与标准高斯基相当，但在 out-of-distribution 和多峰精度上显著更好。

---

## 它与历史 idea 的关系

**升级 LZR（idea 1235）**，保留其核心直觉（限制 latent 采样范围），修正其轴对齐矩形的局限。

具体关系：

1. **继承**：与 LZR 一样，在训练完成后通过一次 calibration 步骤确定每组件的 latent 采样区域，无需重训练
2. **改进**：将 LZR 的轴对齐矩形升级为 GMM，从而捕获：
   - 维度间的相关性（协方差矩阵）
   - 非矩形的 Z_k 形状
   - 软边界（GMM 尾部自然衰减，不截断合法样本）
3. **不替代**：LZR 仍然是一个有效的轻量化方案，当 latent 分布接近轴对齐时（如独立维度的 BreezeForest）LZR 几乎等价于 GMM Latent Base

**对 ICDR（idea 1240）的关系**：无直接关系（ICDR 是训练时的 repulsion，本 Idea 是推理时的采样策略）

---

## 具体实现建议

### 步骤 1：添加 `calibrate_gmm_latent_bases()` 到 MultiBF

```python
from sklearn.mixture import GaussianMixture
import numpy as np

def calibrate_gmm_latent_bases(
    self, 
    x_train, 
    n_gmm_components=1,
    responsibility_threshold=None
):
    """
    Fit a GMM to each component's latent representations, to use as
    the sampling base distribution at inference time.
    
    :param x_train: training data (N, dim), normalized
    :param n_gmm_components: number of GMM components per flow component
    :param responsibility_threshold: if None, use 1/K as threshold
    """
    if responsibility_threshold is None:
        responsibility_threshold = 1.0 / self.n_components

    self.latent_gmm_bases = []

    with torch.no_grad():
        # Compute soft responsibilities
        log_pi = self.get_mixture_log_weights()
        component_log_probs = []
        for k, bf in enumerate(self.components):
            ld = self._per_sample_log_det(bf, x_train)
            component_log_probs.append(log_pi[k] + ld)
        
        stacked = torch.stack(component_log_probs, dim=0)  # (K, N)
        log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
        responsibilities = torch.exp(log_resp)  # (K, N)

        for k, bf in enumerate(self.components):
            resp_k = responsibilities[k]  # (N,)
            
            # Select samples well-assigned to component k
            mask = resp_k > responsibility_threshold
            if mask.sum() < max(10, n_gmm_components * 5):
                # Fallback: top 20% by responsibility
                topk = max(int(0.2 * len(resp_k)), n_gmm_components * 5)
                _, idx = torch.topk(resp_k, min(topk, len(resp_k)))
                mask = torch.zeros_like(resp_k, dtype=torch.bool)
                mask[idx] = True

            x_k = x_train[mask]
            
            # Forward map to latent space
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list).detach().numpy()  # (n_k, dim)
            
            # Fit GMM in latent space
            n_fit_components = min(n_gmm_components, len(z_k) // 5)
            gmm = GaussianMixture(
                n_components=max(1, n_fit_components),
                covariance_type='full',
                random_state=42,
                max_iter=200
            )
            gmm.fit(z_k)
            self.latent_gmm_bases.append(gmm)
    
    print(f"Calibrated GMM latent bases for {len(self.latent_gmm_bases)} components")
    for k, gmm in enumerate(self.latent_gmm_bases):
        print(f"  Component {k}: GMM means = {np.round(gmm.means_, 3)}")
```

### 步骤 2：修改 `inverse_map()` 使用 GMM 基

```python
def inverse_map_with_gmm(self, n_samples, max_gap=1e-3, decay_ratio=1.0):
    """
    Generate samples using per-component GMM latent base distributions.
    Requires calibrate_gmm_latent_bases() to be called first.
    """
    assert hasattr(self, 'latent_gmm_bases'), \
        "Call calibrate_gmm_latent_bases() first"
    
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)

    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        # Sample from component k's GMM latent base
        z_samples, _ = self.latent_gmm_bases[k].sample(n_k)  # (n_k, dim)
        # Clamp to valid BreezeForest latent range
        z_samples = np.clip(z_samples, 0.01, 0.99)
        z = torch.tensor(z_samples, dtype=torch.float32)

        x_k = self.components[k].inverse_map(
            z, max_gap=max_gap, decay_ratio=decay_ratio
        )
        results[mask] = x_k

    return results
```

### 步骤 3：在 demo_multi_bf.py 中集成

```python
# 训练完成后：
all_batch = (all_data - mean) / std  # 使用标准化后的训练数据

with torch.no_grad():
    # 校准 GMM latent 基（n_gmm_components=1 通常够用，组件已专一时）
    mbf.calibrate_gmm_latent_bases(
        all_batch, 
        n_gmm_components=1  # 若 Piecewise 训练则用 1；若 soft-EM 训练则可用 2-3
    )

# 使用 GMM 基生成
with torch.no_grad():
    samples = mbf.inverse_map_with_gmm(n_samples=data_size)
    samples = samples * std + mean
```

### 超参数建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `n_gmm_components` | 1（Piecewise 训练后），2-3（soft-EM 训练后）| soft-EM 下每个组件可能有多个 cluster 的 latent 像 |
| `responsibility_threshold` | 1/K（默认）| 决定哪些样本参与 calibration |
| GMM covariance type | `'full'`（dim ≤ 5），`'diag'`（dim > 5）| 高维时 diag 更稳定 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **GMM 拟合不准** | 若分配给组件 k 的样本太少（<10），GMM 拟合退化 | 用 fallback：当 n_k < 10 时降级为 LZR（轴对齐矩形）|
| **GMM 边界泄漏** | GMM 的尾部在 [0,1] 边界之外的采样被 clamp 到 0.01/0.99，集中在边界处 | 将 clamp 替换为 truncated Gaussian 采样（scipy.stats.truncnorm）|
| **多组件覆盖同一 cluster** | 若 soft-EM 训练使两个组件都覆盖同一 cluster，GMM 会有重叠 | 先做 Piecewise 训练（本轮 idea 1），再用 n_gmm_components=1 |
| **维度增长时 GMM 拟合开销** | 高维时 GMM full covariance 矩阵参数量为 O(dim^2) | dim > 5 时改用 'diag' 协方差；dim > 20 时考虑 PCA 降维后拟合 |
| **sklearn 依赖** | 需要 sklearn.mixture.GaussianMixture | sklearn 在 requirements.txt 中已有（sklearn.datasets 被 distribution2d.py 使用）|

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（与 Piecewise K-Means 并列，且无需重训练）**

理由：
1. **零成本升级**：在已有 LZR（1235）的基础上，用 10 行代码替换轴对齐框为 GMM，无需重训练
2. **直接修复 LZR 的核心限制**：维度相关性和硬截断问题
3. **理论支撑充分**：arxiv 2512.04954（2024）明确验证 GMM base distribution 在 multi-cluster normalizing flow 上的优势
4. **与 Piecewise 训练完美互补**：Piecewise 训练后，每个组件的 latent 分布近似单峰，GMM with n_components=1 精确拟合，采样质量最优
5. **可与 LZR 兼容**：GMM Latent Base 作为 LZR 的超集，当 GMM covariance 为对角且等宽时退化为 LZR

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.  
  [https://proceedings.mlr.press/v151/stimper22a.html](https://proceedings.mlr.press/v151/stimper22a.html)  
  最早提出学习 base distribution 修复多峰 topology 问题（本 Idea 是其数据驱动的非学习简化版）
- (2024). "Amortized Inference of Multi-Modal Posteriors using Likelihood-Weighted Normalizing Flows." *arXiv:2512.04954*.  
  [https://arxiv.org/abs/2512.04954](https://arxiv.org/abs/2512.04954)  
  明确验证：GMM base distribution cardinality 匹配 cluster 数量后，spurious bridge 显著消除
- (2024). "Multimodal base distributions in conditional flow matching generative models." *BMVC 2024*.  
  验证 GMM 基分布在 in-distribution 上与高斯基相当，在多峰精度上更优
- Bevins, H.T.J. & Handley, W. (2023). "Piecewise Normalizing Flows." *arXiv:2305.02930*.  
  支持：分片训练后每个组件的 latent 像更紧凑，GMM 拟合更精确
