# Idea: Empirical Latent Histogram Sampling (ELHS)

**创建时间**: 2026-03-11 21:28 UTC  
**推荐优先级**: ⭐⭐ 高优先级（推断阶段升级，与 KDCT+RTAT 正交互补）

---

## 问题定义

MultiBF 当前的生成策略（`inverse_map` 中的 `z ~ Uniform[0.01, 0.99]^d`）存在一个**隐式假设**：

> 每个组件 k 的 latent 空间中，[0.01, 0.99]^d 的所有 z 值对应的 x 都是"合法的 cluster k 样本"。

这个假设是错误的。原因：

**BreezeForest 是 ℝ^d 到 (0,1)^d 的全局双射**。当组件 k 在全量数据（多 cluster 混合）上训练时：
- Cluster k 的数据 → 映射到 (0,1)^d 的**某个子区域 Z_k**（高密度区域）
- 其他 cluster 的数据 + inter-cluster void → 映射到 Z_k 的补集 **Z_k^c**
- Uniform[0.01, 0.99]^d 的采样包含了 Z_k^c 中的 z 值 → 反映射回去得到 inter-cluster 或其他 cluster 的点

**现有 LZR idea（2026-03-11 12:35）的修复**：用 cluster k 训练数据的 **各维度百分位数** 估计 Z_k 为一个**轴对齐矩形框**，在框内均匀采样。

**LZR 的局限**（本 idea 针对的问题）：
1. **矩形框假设太强**：实际 Z_k 可能是非矩形、非凸的形状（当 cluster k 本身呈现 L 型、弧形等分布时）
2. **独立维度估计**：各维度边界独立估计，忽略了 z 空间中各维度的相关性
3. **均匀采样**：即使在矩形框内，不同 z 值对应的实际数据密度也不均匀，均匀采样会低估 cluster 核心、高估 cluster 边缘
4. **对训练质量敏感**：如果组件未专一化（soft-EM 训练），Z_k 包含多个 cluster，矩形框会非常宽泛

**本 idea 的修复**：用**训练数据在 latent 空间的实际经验分布（直方图）**替代均匀采样，从根本上解决采样分布与实际数据 latent 分布不匹配的问题。

---

## 从代码与已有 idea 中得到的背景判断

**代码关键路径**：

`BreezeForest.forward()` 将 x 映射到 z ∈ (0,1)^d（因为最后一层是 Sigmoid 激活）。

`MultiBF.inverse_map()` 中：
```python
z = torch.rand(n_k, self.dim) * 0.98 + 0.01  # Uniform[0.01, 0.99]^d
x_k = self.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
```

**关键观察**：
1. `z = bf.forward(x_train_k)` 是 cluster k 训练数据在 latent 空间的实际分布
2. 这个分布已经**天然包含了 cluster k 的完整结构信息**
3. 如果我们从这个分布采样 z，而不是从 Uniform 采样，生成的 x 会更忠实于 cluster k

**与 LZR 的关键区别**：
- LZR：从训练数据的 latent codes 计算 [lo, hi] 框 → 在框内均匀采样
- ELHS：**直接将训练数据的 latent codes 作为 empirical distribution** → 有放回地从这些 latent codes 中重采样

**与 KDCT 的协同**：KDCT 训练后，每个组件 k 的训练数据恰好是 cluster k。Z_k = f_k(cluster_k data) 就是 cluster k 在 latent 空间的精确表示。ELHS 直接利用这一点。

---

## 核心思路

**Empirical Latent Distribution Sampling**：

1. **训练后校准**：对组件 k，将 cluster k 的训练数据通过 f_k 正向传播，得到 latent codes：
   ```
   Z_k = {z_i : z_i = f_k(x_i), x_i ∈ cluster_k}
   ```

2. **构建 latent 直方图**：将 (0,1)^d 划分为 G^d 个网格（对于 dim=2，G=50 → 50×50=2500 个格子），统计 Z_k 落入每个格子的数量，归一化为概率分布 π_k^{grid}

3. **直方图采样**：生成时，先根据 π_k^{grid} 选择一个格子，再在格内均匀采样 z → 再通过 inverse_map 得到 x

4. **自动适配 cluster 形状**：由于 π_k^{grid} 是 Z_k 的精确经验分布，它自然包含了 cluster k 的形状、密度分布和空洞（如果有的话）

**关键性质**：
- 高密度 z 区域（cluster 核心的 latent 表示）被**更频繁采样** → 生成更多 cluster 核心样本
- 低密度 z 区域（cluster 边缘的 latent 表示）被**较少采样** → 减少 cluster 边缘和 inter-cluster 的生成
- Z_k 未覆盖的区域（inter-cluster 和其他 cluster 的 latent 表示）被**完全不采样** → 彻底消除 inter-cluster 生成

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**定量论证**：

设训练数据分布为 P_data，组件 k 的 latent 表示为：
```
Z_k = f_k(cluster_k data)  ⊂ (0,1)^d
```

当 component k 使用 KDCT 专一化训练时：
- f_k 的 CDF 将 cluster k 的数据展开映射到 (0,1)^d 的大部分区域
- inter-cluster 和其他 cluster 的数据映射到 (0,1)^d 的边缘区域（z 值接近 0 或 1）

采样策略比较：

| 策略 | 采样的 z 范围 | 生成的 x 来源 |
|------|-------------|-------------|
| Uniform[0.01, 0.99]^d | 整个 [0.01, 0.99]^d | Cluster k + 部分边缘区域 |
| LZR（矩形框） | Z_k 的轴对齐包围盒 | Cluster k + 包围盒内的非 Z_k 区域 |
| **ELHS（本 idea）** | **Z_k 的实际分布** | **几乎完全是 Cluster k 本身** |

**对比 Stimper et al. (2022) 的 Resampled Base Distribution**：
- Stimper et al. 通过学习一个 acceptance function 来 rejection-sample from uniform base → 需要额外训练
- ELHS 直接用经验分布（直方图），无需训练，且更精确（不受 acceptance function 近似误差影响）

---

## 与历史 idea 的关系

**与 LZR (2026-03-11 12:35) 的关系：显著升级**

| 维度 | LZR（现有 Idea 2） | ELHS（本 Idea） |
|------|-----------------|----------------|
| Zone 形状 | 轴对齐矩形框（维度独立） | **任意形状**（直方图） |
| 维度相关性 | 忽略 | **自然建模** |
| Zone 内采样 | 均匀 | **按实际密度采样** |
| 对训练质量的依赖 | 高（组件未专一化时框太宽） | 同样（但精度更高） |
| 实现复杂度 | 低（quantile 计算） | 中（直方图构建 + 采样） |
| 与 KDCT 的协同 | 一般（KDCT 后 LZR 效果提升） | **极好**（KDCT 后 Z_k 精确对应 cluster k） |

LZR 是矩形盒版本，ELHS 是精确版本。**如果已经用了 KDCT，ELHS 的优势更加突出**：KDCT 保证 Z_k 是 cluster k 的精确 latent 表示，ELHS 完整利用这一信息。

**与 Hard-EM (2026-03-11 12:30) 的关系**：互补（Hard-EM 是训练阶段，ELHS 是推断阶段）。在 Hard-EM 训练后使用 ELHS 比 LZR 更能利用组件专一化的结果。

**不替代 KDCT 或 RTAT**：ELHS 是推断阶段修复，KDCT+RTAT 是训练阶段修复。最强组合：KDCT+RTAT 训练 + ELHS 推断。

---

## 具体实现建议

### 步骤 1：添加 `build_latent_histogram()` 方法到 MultiBF

```python
def build_latent_histogram(self, cluster_data_list, grid_size=50):
    """
    Build per-component latent histogram from training data.
    
    :param cluster_data_list: list of (N_k, dim) tensors, one per component
                              (from KDCT pre-clustering, or from responsibility-based assignment)
    :param grid_size: number of bins per dimension (default 50 → 50×50 grid for 2D)
    """
    self.latent_histograms = []
    self.grid_size = grid_size
    
    with torch.no_grad():
        for k, bf in enumerate(self.components):
            x_k = cluster_data_list[k]
            if len(x_k) == 0:
                # Fallback: uniform histogram
                self.latent_histograms.append(None)
                continue
            
            # Forward pass: get latent codes for cluster k's training data
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list)  # (N_k, dim), values in (0,1)
            
            # Build histogram in [0, 1]^dim
            # For dim=2: 2D histogram
            if self.dim == 2:
                hist, _, _ = np.histogram2d(
                    z_k[:, 0].numpy(),
                    z_k[:, 1].numpy(),
                    bins=grid_size,
                    range=[[0.01, 0.99], [0.01, 0.99]],
                    density=False
                )
                # Normalize to probability distribution
                hist = hist.astype(np.float32) + 1e-6  # Laplace smoothing
                hist = hist / hist.sum()
                self.latent_histograms.append(hist)
            
            else:
                # For higher dims: use independent 1D histograms as approximation
                # (product of marginals)
                marginals = []
                for d in range(self.dim):
                    hist_d, _ = np.histogram(
                        z_k[:, d].numpy(),
                        bins=grid_size,
                        range=(0.01, 0.99),
                        density=False
                    )
                    hist_d = hist_d.astype(np.float32) + 1e-6
                    hist_d = hist_d / hist_d.sum()
                    marginals.append(hist_d)
                self.latent_histograms.append(marginals)
    
    print(f"Built latent histograms for {len(self.latent_histograms)} components (grid_size={grid_size})")
```

### 步骤 2：直方图采样函数

```python
def _sample_from_histogram_2d(self, hist_2d, n_samples):
    """
    Sample z from 2D histogram distribution.
    
    :param hist_2d: (G, G) numpy array, normalized probability distribution
    :param n_samples: number of samples
    :return: (n_samples, 2) tensor of z values in [0.01, 0.99]^2
    """
    G = hist_2d.shape[0]
    
    # Flatten histogram for multinomial sampling
    flat_hist = hist_2d.flatten()
    
    # Sample grid cells
    cell_indices = np.random.choice(len(flat_hist), size=n_samples, p=flat_hist)
    row_indices = cell_indices // G
    col_indices = cell_indices % G
    
    # Convert to [0, 1] coordinates (center of each cell + small jitter)
    bin_width = (0.99 - 0.01) / G
    z1 = 0.01 + (row_indices + np.random.uniform(0, 1, n_samples)) * bin_width
    z2 = 0.01 + (col_indices + np.random.uniform(0, 1, n_samples)) * bin_width
    
    z = torch.tensor(np.stack([z1, z2], axis=1), dtype=torch.float32)
    return z.clamp(0.01, 0.99)
```

### 步骤 3：修改 `inverse_map` 使用直方图采样

```python
def inverse_map_with_histogram(self, n_samples, max_gap=1e-3, decay_ratio=1.0):
    """
    Generate samples using per-component empirical latent histogram sampling.
    Requires build_latent_histogram() to be called first.
    """
    assert hasattr(self, 'latent_histograms'), "Call build_latent_histogram() first"
    
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)

    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue

        if self.latent_histograms[k] is None:
            # Fallback to uniform
            z = torch.rand(n_k, self.dim) * 0.98 + 0.01
        elif self.dim == 2:
            z = self._sample_from_histogram_2d(self.latent_histograms[k], n_k)
        else:
            # For higher dims: sample from product of marginals
            z_parts = []
            for d in range(self.dim):
                G = len(self.latent_histograms[k][d])
                bin_width = (0.99 - 0.01) / G
                cell_indices = np.random.choice(G, size=n_k, p=self.latent_histograms[k][d])
                z_d = 0.01 + (cell_indices + np.random.uniform(0, 1, n_k)) * bin_width
                z_parts.append(torch.tensor(z_d, dtype=torch.float32))
            z = torch.stack(z_parts, dim=1).clamp(0.01, 0.99)

        x_k = self.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
        results[mask] = x_k

    return results
```

### 步骤 4：在训练后添加直方图构建

```python
# 训练完成后（在 demo_multi_bf.py 中）：

# 如果使用了 KDCT，cluster_data 已经按 cluster 分好
with torch.no_grad():
    mbf.build_latent_histogram(cluster_data, grid_size=50)

# 生成
with torch.no_grad():
    samples = mbf.inverse_map_with_histogram(n_samples=data_size)
    samples = samples * std + mean

# 或者：与 LZR 对比验证
with torch.no_grad():
    samples_lzr = mbf.inverse_map_with_zones(n_samples=data_size)
    samples_elhs = mbf.inverse_map_with_histogram(n_samples=data_size)
```

### 超参数建议

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| `grid_size` | 30-100 | 太小：估计精度低；太大：稀疏问题（需要更多训练数据）。对 2D 数据推荐 50。 |
| Laplace 平滑 | 1e-6 | 防止零概率格子（已设为默认值） |
| 对高维数据 | 独立边缘 | 对 dim > 4 的数据，联合直方图维度灾难，用独立边缘乘积近似 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **稀疏直方图** | cluster 数据量少时，直方图可能有大量空格子（稀疏），采样退化 | 增大 Laplace 平滑（如 0.01）；或降低 grid_size；或切换到 KDE |
| **维度灾难** | 高维数据（dim > 4）的联合直方图格子数爆炸 | 使用独立边缘近似，或 PCA 降维后在低维空间建直方图 |
| **间接依赖训练质量** | 如果组件未专一化，Z_k 包含多个 cluster → 直方图散乱 → 采样效果有限 | 与 KDCT 或 RTAT 结合使用，确保组件专一化后再构建直方图 |
| **离散化误差** | 直方图是离散近似，格内均匀抖动引入小误差 | 对于低维数据（2-4D），grid_size=50-100 的误差可忽略 |
| **计算和存储开销** | 50×50 直方图只有 2500 个 float，K 个组件合计 K×2500 → 极轻量 | 无明显开销 |

---

## 推荐优先级

**⭐⭐ 高优先级（推断阶段升级，与 KDCT+RTAT 互补）**

理由：
1. **精准定位**：ELHS 直接利用训练数据在 latent 空间的实际分布，是对 LZR 矩形框的精确替代
2. **零训练成本**：只需一次前向传播 + 直方图统计，无需任何额外训练
3. **对 KDCT 的完美配合**：KDCT 训练后 Z_k 精确对应 cluster k，ELHS 完整利用这一精确对应
4. **可独立验证**：可以直接在已训练的 MultiBF 模型上验证效果，无需重训练
5. **理论支撑**：与 Stimper et al. (2022) 的 resampled base distribution 思路同源，但更简单高效

**建议使用顺序**：
1. KDCT（训练阶段专一化）
2. RTAT（训练过程稳定化）
3. ELHS（推断阶段精准采样）

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a/stimper22a.html  
  (同源方法，通过学习 acceptance function 修复 latent 采样分布)
- Liu, S. et al. (2024). "Multimodal base distributions in conditional flow matching generative models." *BMVC 2024*. https://bmvc2024.org/proceedings/492/  
  (使用 GMM base distribution 而非 uniform，匹配 cluster 基数可提升重建精度)
- arXiv 2512.04954 (2024). "Amortized Inference of Multi-Modal Posteriors using Likelihood-Weighted Normalizing Flows."  
  (使用 GMM 初始化 flow 的 base distribution 匹配目标模态的基数，直接支持 empirical latent distribution 思路)
- Bevins, H. et al. (2023). "Piecewise Normalizing Flows." *arXiv 2305.02930*.  
  (说明每组件独立 flow 训练后，latent 分布精确对应各 cluster，验证 ELHS 的前提假设)
