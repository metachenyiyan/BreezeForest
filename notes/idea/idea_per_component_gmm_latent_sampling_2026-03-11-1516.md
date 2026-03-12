# Idea: Per-Component GMM Latent Sampling（组件级 GMM 潜空间采样）

**创建时间**: 2026-03-11 15:16 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（升级旧 LZR Idea，可立即用于已训练模型）

---

## 问题定义

MultiBF 的生成阶段对每个组件 k 使用 `z ~ Uniform([0.01, 0.99]^d)` 作为 latent 采样分布。即使在 Hard-EM 或顺序专家训练后，组件 k 建模了 cluster k，其 `inverse_map_k` 仍然对整个 `[0.01, 0.99]^d` 空间均匀采样。

这产生两个问题：
1. **训练不完美时**：组件 k 的 CDF 变换在 cluster k 外部仍然有定义，均匀采样会触发这些区域，生成 inter-cluster 或 out-of-cluster 样本。
2. **latent 空间分布不均匀**：即使 BF_k 在 cluster k 上训练完美，cluster k 的数据在 z-space 中并非均匀分布于整个 `[0,1]^d`——它们集中在一个子区域（由于 CDF 的局部映射特性），而 `[0,1]^d` 的其他区域对应的 x 值在 cluster k 的分布支撑之外。

旧 LZR idea（`idea_latent_zone_restriction_2026-03-11-1235.md`）通过统计每个组件分配样本的 z-space 百分位数边界，将采样限制在 `[lo_k, hi_k]^d` 的超矩形内。这有两个显著缺陷：
- **超矩形 ≠ 真实 z-cluster 形状**：z-space 中的 cluster 分布通常是椭球形甚至非凸形的，超矩形会包含大量超出真实 z-cluster 的区域。
- **维度独立性假设**：按维度分别取百分位数忽略了维度间的相关性，即便所有维度都在边界内，组合起来的 z 也可能落在真实 cluster 的外部。

---

## 从当前项目代码与已有 idea 中得到的背景判断

### 代码分析

`MultiBF.inverse_map()` 的采样逻辑：
```python
z = torch.rand(n_k, self.dim) * 0.98 + 0.01  # Uniform([0.01, 0.99]^d)
x_k = self.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
```

问题在于 `z` 没有利用任何关于 cluster k 的先验知识。

`BreezeForest.inverse_map()` 中的 `compute_dis()` 只计算了 batch 级别的 Normal 分布（用于二分搜索的初始化范围），并未约束生成的 z 值本身。

### z-space 的几何理解

对于组件 k（假设它被专一化为 cluster k）：
- 前向传播 `z_i = BF_k(x_i)`：cluster k 的数据 `x_i` 映射到 `[0,1]^d` 的某个子集 `Z_k`
- `Z_k` 的形状取决于 cluster k 的形态：如果 cluster k 是一个轴对齐的椭球，`Z_k` 也大致是椭球形
- `[0,1]^d \ Z_k` 对应的 `f_k^{-1}(z)` 是 cluster k 分布支撑之外的点（包括 inter-cluster void）

旧 LZR 用超矩形 `[lo_k, hi_k]^d` 近似 `Z_k`，本 Idea 用 **高斯混合模型（GMM）** 更精确地建模 `Z_k`。

### 外部调研验证

- Stimper et al. (AISTATS 2022) 的 resampling base distributions 方法通过学习 rejection sampling 来修复 base distribution，与本 Idea 思路一致，但本 Idea 用数据驱动的 GMM 代替了额外的神经网络学习步骤，更简单且无需重训练。
- Coeurdoux et al. (ML 2024, arXiv:2305.12149) 使用 MALA 在 latent space 中做 MCMC 采样。本 Idea 是一个更简单的等价方案：用 GMM 近似 latent 分布，直接采样而无需 MCMC。

---

## 核心思路

**训练后校准（Post-Training Calibration）+ GMM 拟合**：

1. 对每个组件 k，获取分配给 cluster k 的训练数据 `{x_i : assignment_i == k}`
2. 通过 BF_k 前向传播得到它们的 latent 表示 `{z_i = BF_k(x_i)}`
3. 用 sklearn.mixture.GaussianMixture 在 `{z_i}` 上拟合一个 GMM（n_components_z = 1 或 2 即可）
4. **生成时**：从该 GMM（clamp 到 [0.01, 0.99]^d）采样 z，再通过 `BF_k^{-1}` 映射回 x-space

GMM 天然地：
- 建模了 z-space 中 cluster k 的真实分布形状（椭球、旋转、各维度相关性）
- 在 Z_k 外部分配极低的采样概率
- 避免了超矩形逼近的"角落泄漏"问题

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**数学直觉**：

设 `p_k^z` 为 cluster k 数据在 z-space 中的真实分布（即 `z = BF_k(x_i)` 的分布）：

```
如果 BF_k 专一于 cluster k:
  - 对 x_i ~ cluster k: z_i = BF_k(x_i) ~ p_k^z  (高密度区域)
  - 对 x_j ~ inter-cluster void: z_j = BF_k(x_j) ~ 某个低密度区域

从 p_k^z 采样 z:
  - z 落在 cluster k 的高密度 latent 区域
  - BF_k^{-1}(z) 映射回 cluster k 附近
  - inter-cluster void 的 latent 区域几乎不被采样
```

**GMM vs 超矩形（旧 LZR）对比**：

| 方面 | 旧 LZR（超矩形） | 本 Idea（GMM） |
|------|-----------------|---------------|
| z-cluster 形状建模 | 轴对齐矩形，忽略相关性 | 椭球（含旋转），捕获维度相关 |
| 角落泄漏 | 有（超矩形的角落可能在 Z_k 外） | 无（GMM 的低密度区域贡献极少） |
| 参数调优 | 需要手动调 percentile_low/high | GMM 参数从数据学习，无需手调 |
| 多模 z-cluster | 不支持 | 支持（增加 z-GMM 组件数） |
| 实现复杂度 | 简单（torch 分位数） | 稍复杂（sklearn GMM + 自定义采样） |

---

## 与历史 idea 的关系

**升级旧 LZR idea（`idea_latent_zone_restriction_2026-03-11-1235.md`）**：

本 Idea 保留了 LZR 的核心直觉（限制 z-space 采样到 cluster k 对应的区域），但用 GMM 替换了超矩形，提升了精度和鲁棒性。

与 **Hard-EM / Sequential Expert Training** 的关系：**互补**：
- Hard-EM / 顺序专家训练（新 Idea 1）确保组件专一化，使 `{z_i = BF_k(x_i)}` 集中在清晰的 latent cluster 区域
- 本 Idea 在此基础上，精确建模这个 latent cluster 区域，进一步提升生成质量

**即使不用 Hard-EM（仅 soft-EM 训练），本 Idea 也能改善效果**：对 soft-EM 训练的 MultiBF，responsibility 最高的样本也会在 latent space 中形成相对集中的区域，GMM 可以近似这个区域。

---

## 具体实现建议

### 步骤 1：添加 calibrate_latent_gmms() 到 MultiBF

```python
from sklearn.mixture import GaussianMixture
import numpy as np

def calibrate_latent_gmms(self, x_train, n_gmm_components=1, 
                           responsibility_threshold=None):
    """
    Fit per-component GMM in latent z-space using data from assigned samples.
    
    :param x_train: training data (N, dim) tensor
    :param n_gmm_components: number of Gaussian components for z-space GMM
                             (1 usually sufficient after Hard-EM; 2-3 for soft-EM)
    :param responsibility_threshold: min responsibility to include sample
                                     (default: 1/K for uniform baseline)
    """
    if responsibility_threshold is None:
        responsibility_threshold = 1.0 / self.n_components
    
    self.latent_gmms = []
    
    with torch.no_grad():
        # Compute responsibilities
        log_pi = self.get_mixture_log_weights()
        component_log_probs = []
        for k, bf in enumerate(self.components):
            ld = self._per_sample_log_det(bf, x_train)
            component_log_probs.append(log_pi[k] + ld)
        
        stacked = torch.stack(component_log_probs, dim=0)   # (K, N)
        log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
        resp = torch.exp(log_resp)  # (K, N)
        
        for k, bf in enumerate(self.components):
            # Select high-responsibility samples for component k
            resp_k = resp[k]
            
            # Hard assignment: use top-responsibility samples
            assignments = torch.argmax(resp, dim=0)  # (N,)
            mask = (assignments == k)
            
            # Fallback if too few samples
            if mask.sum() < max(10, n_gmm_components * 5):
                _, idx = torch.topk(resp_k, min(200, len(resp_k)))
                mask = torch.zeros(len(resp_k), dtype=torch.bool)
                mask[idx] = True
            
            x_k = x_train[mask]
            
            # Forward pass to get z-space representation
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list).detach().cpu().numpy()
            
            # Fit GMM in z-space
            gmm = GaussianMixture(
                n_components=n_gmm_components, 
                covariance_type='full',
                n_init=5, 
                random_state=42
            )
            gmm.fit(z_k)
            self.latent_gmms.append(gmm)
        
        print(f"Fitted {len(self.latent_gmms)} latent GMMs.")
        for k, gmm in enumerate(self.latent_gmms):
            print(f"  Component {k}: GMM means = {gmm.means_.round(3)}")

def sample_from_latent_gmm(self, k, n_samples):
    """
    Sample from component k's fitted latent GMM, clamped to [0.01, 0.99]^d.
    Returns z tensor of shape (n_samples, dim).
    """
    gmm = self.latent_gmms[k]
    z_np, _ = gmm.sample(n_samples)  # (n_samples, dim)
    z = torch.tensor(z_np, dtype=torch.float32).clamp(0.01, 0.99)
    return z
```

### 步骤 2：修改 inverse_map() 使用 GMM 采样

```python
def inverse_map_with_gmm(self, n_samples, max_gap=1e-3, decay_ratio=1.0):
    """
    Generate samples using per-component latent GMM sampling.
    Requires calibrate_latent_gmms() to be called first.
    """
    assert hasattr(self, 'latent_gmms'), "Call calibrate_latent_gmms() first"
    
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)

    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        # Sample from component k's latent GMM (instead of Uniform)
        z = self.sample_from_latent_gmm(k, n_k)
        
        x_k = self.components[k].inverse_map(
            z, max_gap=max_gap, decay_ratio=decay_ratio
        )
        results[mask] = x_k

    return results
```

### 步骤 3：超参数建议

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_gmm_components` | 1 | Hard-EM 后，每个组件的 z-cluster 是单峰的，用 1 即可。soft-EM 后用 2-3。 |
| `covariance_type` | 'full' | 捕获维度间相关性。如果 dim > 20，改用 'diag' 降低参数量。 |
| `clamp` 范围 | [0.01, 0.99] | 避免 icdf 在 0/1 边界处爆炸 |

### 步骤 4：验证建议

用以下方式验证 GMM 是否有效建模了 latent cluster：
```python
# 检查 GMM 的 samples 是否覆盖了真实 z-cluster
z_real = bf_k.forward(x_train_k).detach().cpu().numpy()  # 真实 z-cluster
z_gmm, _ = latent_gmms[k].sample(len(z_real))            # GMM 采样

# 两者的分布应该相近（用 KDE 可视化对比）
```

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **GMM 过拟合 z-cluster** | 用过多组件或过少数据，GMM 可能 overfit z-cluster | 用 n_gmm_components=1，使用 BIC 选择最优 K |
| **z-space GMM 与 x-space 关系非线性** | GMM 在 z-space 是高斯，但对应的 x-space 分布不是高斯 | 这是优点：GMM 在 z-space 的简单形状即可对应 x-space 中复杂的 cluster 形状 |
| **soft-EM 训练后 z-clusters 不清晰** | 如果组件未专一化，z_k 混入多个 cluster 的点，GMM 拟合不准 | 先做 Hard-EM / 顺序专家训练（新 Idea 1），再做 GMM 校准 |
| **Sklearn 依赖** | 需要 sklearn.mixture.GaussianMixture | sklearn 是 BreezeForest 现有依赖（distribution2d.py 已用 make_moons 等），无需新增 |
| **GMM 样本超出 [0.01, 0.99]^d** | GMM 是无界分布，偶尔采出极端值 | clamp 到 [0.01, 0.99] 后传入 bisection，不影响 inverse_map 的正确性 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（可立即在已训练模型上验证，无需重训练）**

理由：
1. **即时可验证**：对任何已训练的 MultiBF 模型运行 `calibrate_latent_gmms()` 后立即可用
2. **比旧 LZR 精度更高**：GMM 捕获 z-cluster 的形状、方向和维度相关性，超矩形做不到
3. **实现简单**：核心代码约 40 行，依赖现有 sklearn
4. **与 Idea 1 形成最强组合**：顺序专家训练（Idea 1）使 z-clusters 清晰 → GMM 精准建模 z-clusters → 采样质量大幅提升
5. **应用范围广**：适用于任何 soft-EM 或 Hard-EM 训练的 MultiBF，也可单独使用

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.  
  https://proceedings.mlr.press/v151/stimper22a.html  
  (Foundational work on non-uniform base distributions for flows; this Idea is a simpler data-driven version)
- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 2024*. arXiv:2305.12149  
  (Latent-space MCMC for multimodal flows; GMM sampling is a simpler alternative to MALA)
- Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*, Chapter 9 (GMM and EM algorithm).
