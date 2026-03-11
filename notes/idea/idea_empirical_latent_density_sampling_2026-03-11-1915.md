# Idea: Empirical Latent Density Sampling (ELD-S)

**创建时间**: 2026-03-11 19:15 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（升级 LZR，同时适用于单 BreezeForest 和 MultiBF）

---

## 问题定义

BreezeForest（单组件）和 MultiBF 在**生成阶段**的核心问题是：采样使用 `z ~ Uniform([0.01, 0.99]^d)`，然后通过 bisection 的 `inverse_map` 将 z 映射回数据空间。

这个均匀采样策略有一个根本缺陷：

**训练数据的正向映射并不会均匀覆盖 [0.01, 0.99]^d。**

对于多 cluster 数据：
- cluster A 的数据通过正向映射 f 映射到 [0.01, 0.99]^d 的某个**局部高密度子区域 Z_A**
- cluster B 的数据映射到另一个子区域 **Z_B**
- cluster 之间的"空白区域"（低密度区域）在 latent space 中也有对应的 z 范围——这些 z 是"无数据对应"的，反向映射后会产生 inter-cluster 区域的点

当采样 z ~ Uniform 时，不可避免地采到这些"无数据对应"的 z 值，逆映射得到 inter-cluster 点。

**关键洞察**：不需要改变 flow 模型本身，只需要在生成时**用训练数据的实际 latent 分布替代 Uniform**，即可自然规避这些有问题的 z 区域。

---

## 从当前项目代码与已有 idea 中得到的背景判断

**代码观察**：
- `demo_functions.py` 的 `generate_sample()` 函数：`seeds = distribution.sample(torch.Size([sample_size, 2]))` 使用 `torch.distributions.uniform.Uniform(0.01, 0.99)`，完全均匀采样。
- `MultiBF.inverse_map()` 中：`z = torch.rand(n_k, self.dim) * 0.98 + 0.01`，同样是均匀采样。
- 没有任何对 latent 空间分布的建模或约束。

**已有 idea 分析**：
- `idea_latent_zone_restriction_2026-03-11-1235.md`（LZR）：通过对训练数据做正向映射，估计每个组件的 latent zone，然后只在该 zone 内均匀采样。
  - **LZR 的局限性**：
    1. 使用每维独立的百分位数（矩形框），忽略维度间的相关性
    2. 矩形框可能包含 cluster 之间的 latent 区域（两个 cluster 的 z_x 重叠但 z_y 不重叠时，矩形框仍会覆盖它们中间的部分）
    3. **不适用于单 BreezeForest 场景**（LZR 依赖多组件的 responsibility 计算）
    4. 需要调整 `percentile_low` / `percentile_high` 超参数

**综合判断**：  
LZR 的核心思路是正确的（从训练数据的 latent 表示出发），但实现方式过于简化。本 Idea 提供更精确的实现：**直接用经验 latent 分布（empirical latent distribution）替代 Uniform 作为采样分布**。

---

## 核心思路

**训练后一次性校准 + 生成时使用经验分布采样**：

1. **Latent 经验分布估计**：  
   训练完成后，将训练数据正向传播：`{z_i = f(x_i)}`，得到训练数据在 latent 空间的经验分布。

2. **每维度经验 CDF 构建**：  
   对每个维度 d，将 `{z_i^d}` 的经验分布表示为排好序的分位数数组（100 个百分位点）。

3. **逆 CDF 采样（概率积分变换）**：  
   生成时：
   - 采样 `u ~ Uniform(0, 1)^d`（在分位数空间均匀）
   - 对每维用线性插值将 u 映射到对应的 z 值（即逆 CDF）
   - 得到的 z 就是来自经验 latent 分布的样本
   - 再通过 `inverse_map(z)` 映射回数据空间

对于 MultiBF，对每个组件 k 分别建立经验 CDF（使用分配给组件 k 的训练样本的 latent 表示）。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**直觉论证**：

对于训练好的 BreezeForest，训练数据的 latent 表示 `{z_i}` 在 [0,1]^d 内形成多个高密度局部区域（对应多个 cluster）。这些高密度区域之间存在低密度"间隙"（对应 inter-cluster 空间）。

- **Uniform 采样**：平等地采样所有 z，包括间隙区域，产生 inter-cluster 点
- **ELD-S**：按经验分布采样，高密度区域采到更多 z，间隙区域几乎采不到 z

等价地：对于一个 1D CDF 函数 F(x)，若数据来自两个分离的 cluster，F(x) 在 cluster 之间的区域变化极慢（几乎水平）。当我们用逆 CDF 采样时，这段"水平区域"在 u 轴上占据的范围很小，因此很少被采到。这正是 ELD-S 的工作原理。

**与 LZR 的对比**：

| 维度 | LZR（旧 Idea 2） | ELD-S（本 Idea） |
|------|----------------|----------------|
| 分布估计 | 每维独立百分位数矩形框 | **每维经验 CDF（精确形状）** |
| 维度相关性 | 忽略 | 忽略（可升级） |
| 超参数 | percentile_low, percentile_high | **无**（分位数数组自动确定） |
| 适用场景 | MultiBF 多组件 | **单 BF + MultiBF 均适用** |
| 区域外样本排除 | 通过矩形裁剪 | **通过密度自然加权，无需裁剪** |
| 跨 cluster 情况 | 矩形框可能包含两 cluster 的 z 区域 | **经验 CDF 自然按数据密度采样** |
| 理论支撑 | 直觉式设计 | **概率积分变换**（严格数学基础） |

**与外部文献连接**：
- Stimper et al. (2022) "Resampling Base Distributions" 通过**学习**一个新的 base distribution 解决同样问题。ELD-S 是其无学习版本：直接用训练数据的经验分布，不需要额外训练任何参数。
- Coeurdoux et al. (2023/2024) "Normalizing flow sampling with Langevin dynamics in the latent space" 使用 MALA 在 latent 空间做 MCMC，通过 Jacobian 引导游走。ELD-S 比 MALA 更简单（无需 MCMC 迭代），在 BreezeForest 的 [0,1]^d 有界 latent 空间中尤为适合。

---

## 它与历史 idea 的关系

**升级替代 `idea_latent_zone_restriction_2026-03-11-1235.md`（LZR）**。

LZR 的核心思路（从训练数据 latent 表示出发）是正确的，ELD-S 是其更精确、无超参数、适用范围更广的升级版：

- LZR 用矩形框（每维独立百分位数） → ELD-S 用经验 CDF（精确形状）
- LZR 只适用于 MultiBF → ELD-S 也适用于单 BreezeForest
- LZR 需要设置 `percentile_low`/`percentile_high` → ELD-S 无需超参数

若已实现 LZR，ELD-S 的实现代价仅略高（需构建逐维经验 CDF 而非简单裁剪），但效果显著改善。

**与 `idea_kmeans_prepartition_dedicated_training_2026-03-11-1912.md`（K-Means 预划分）的关系**：  
互补。K-Means 预划分改善训练阶段（使组件更专一化），ELD-S 改善生成阶段。组合使用：
- 更专一化的组件 → 其 latent 分布更紧凑集中 → ELD-S 的经验 CDF 更能有效区分 cluster
- 效果叠加，互不冲突

**与 `idea_inter_component_density_repulsion_2026-03-11-1240.md`（ICDR）的关系**：  
独立互补。ICDR 作用在训练时，ELD-S 作用在生成时，无耦合。

---

## 具体实现建议

### 步骤 1：为单 BreezeForest 添加 ELD-S 支持

```python
class BreezeForest(torch.nn.Module):
    # ... 现有代码 ...

    def calibrate_latent_empirical_cdf(self, x_train, n_quantiles=200):
        """
        Estimate per-dimension empirical CDF of latent representations.
        
        :param x_train: training data tensor (N, dim)
        :param n_quantiles: number of quantile points to store (resolution)
        """
        with torch.no_grad():
            breeze_list = []
            z = self.forward(x_train, breeze_list)  # (N, dim)
        
        self.latent_empirical_quantiles = []
        q_levels = torch.linspace(0, 1, n_quantiles + 1)  # 0, 1/n, ..., 1
        
        for d in range(self.dim):
            z_d = z[:, d]
            q_vals = torch.quantile(z_d, q_levels)  # (n_quantiles+1,)
            self.latent_empirical_quantiles.append(q_vals)
    
    def sample_latent_empirical(self, n_samples):
        """
        Sample latent vectors from empirical distribution using inverse CDF.
        
        :return: z samples (n_samples, dim) drawn from empirical latent distribution
        """
        assert hasattr(self, 'latent_empirical_quantiles'), \
            "Call calibrate_latent_empirical_cdf() first"
        
        n_q = len(self.latent_empirical_quantiles[0]) - 1
        z_samples = torch.zeros(n_samples, self.dim)
        
        for d in range(self.dim):
            q = self.latent_empirical_quantiles[d]  # (n_q+1,)
            
            # Sample uniform in [0, 1] and map through empirical inverse CDF
            u = torch.rand(n_samples)            # (n_samples,) in [0, 1]
            
            # Linear interpolation between quantile values
            # u * n_q gives fractional index into quantile array
            idx = (u * n_q).long().clamp(0, n_q - 1)
            alpha = (u * n_q) - idx.float()
            
            z_d = q[idx] + alpha * (q[idx + 1] - q[idx])
            z_samples[:, d] = z_d
        
        return z_samples
    
    def inverse_map_empirical(self, n_samples, max_gap=1e-3, decay_ratio=1.0):
        """
        Generate samples using empirical latent distribution.
        Requires calibrate_latent_empirical_cdf() to be called first.
        """
        z = self.sample_latent_empirical(n_samples)
        return self.inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
```

### 步骤 2：为 MultiBF 添加 ELD-S 支持

```python
class MultiBF(torch.nn.Module):
    # ... 现有代码 ...

    def calibrate_latent_empirical_cdfs(self, x_train, n_quantiles=200):
        """
        Per-component empirical CDF of latent representations.
        Uses responsibility-weighted assignment to assign training data to components.
        
        :param x_train: training data (N, dim)
        :param n_quantiles: resolution of quantile representation
        """
        with torch.no_grad():
            # Compute component assignments via argmax responsibility
            log_pi = self.get_mixture_log_weights()
            comp_log_probs = []
            for k, bf in enumerate(self.components):
                ld = self._per_sample_log_det(bf, x_train)
                comp_log_probs.append(log_pi[k] + ld)
            stacked = torch.stack(comp_log_probs, dim=0)  # (K, N)
            assignments = torch.argmax(stacked, dim=0)    # (N,)
            
            self.component_latent_quantiles = []
            q_levels = torch.linspace(0, 1, n_quantiles + 1)
            
            for k, bf in enumerate(self.components):
                mask = (assignments == k)
                if mask.sum() < 10:
                    # Fallback: use all data
                    x_k = x_train
                else:
                    x_k = x_train[mask]
                
                breeze_list = []
                z_k = bf.forward(x_k, breeze_list)  # (n_k, dim)
                
                dim_quantiles = []
                for d in range(self.dim):
                    q = torch.quantile(z_k[:, d], q_levels)
                    dim_quantiles.append(q)
                
                self.component_latent_quantiles.append(dim_quantiles)
    
    def _sample_latent_empirical_k(self, k, n_samples):
        """Sample latent vectors for component k from empirical CDF."""
        dim_quantiles = self.component_latent_quantiles[k]
        n_q = len(dim_quantiles[0]) - 1
        
        z = torch.zeros(n_samples, self.dim)
        for d in range(self.dim):
            q = dim_quantiles[d]
            u = torch.rand(n_samples)
            idx = (u * n_q).long().clamp(0, n_q - 1)
            alpha = (u * n_q) - idx.float()
            z[:, d] = q[idx] + alpha * (q[idx + 1] - q[idx])
        return z
    
    def inverse_map_empirical(self, n_samples, max_gap=1e-3, decay_ratio=1.0):
        """
        Generate samples using per-component empirical latent distributions.
        Requires calibrate_latent_empirical_cdfs() to be called first.
        """
        assert hasattr(self, 'component_latent_quantiles'), \
            "Call calibrate_latent_empirical_cdfs() first"
        
        weights = self.get_mixture_weights().detach()
        component_indices = torch.multinomial(weights, n_samples, replacement=True)
        results = torch.zeros(n_samples, self.dim)
        
        for k in range(self.n_components):
            mask = (component_indices == k)
            n_k = mask.sum().item()
            if n_k == 0:
                continue
            
            z_k = self._sample_latent_empirical_k(k, n_k)
            x_k = self.components[k].inverse_map(z_k, max_gap=max_gap, decay_ratio=decay_ratio)
            results[mask] = x_k
        
        return results
```

### 步骤 3：在 demo 中集成使用

```python
# 训练完成后：

# 单 BF 场景
with torch.no_grad():
    bf.calibrate_latent_empirical_cdf(all_data_norm, n_quantiles=200)
    samples = bf.inverse_map_empirical(n_samples=data_size)
    samples = samples * std + mean

# MultiBF 场景
with torch.no_grad():
    mbf.calibrate_latent_empirical_cdfs(all_data_norm, n_quantiles=200)
    samples = mbf.inverse_map_empirical(n_samples=data_size)
    samples = samples * std + mean
```

### 量化分辨率建议

| `n_quantiles` | 内存开销 | 精度 | 推荐场景 |
|---------------|---------|------|---------|
| 50 | 极小 | 粗糙 | 快速验证 |
| 200 | 小 | 良好 | **推荐默认值** |
| 1000 | 中等 | 精确 | 数据量大时 |

### 可选升级：2D 联合经验 CDF（适合 dim=2）

对于 dim=2 的情况，可以用 2D 直方图捕获维度相关性：
```python
# 构建 2D 直方图
hist, xedges, yedges = np.histogram2d(z[:, 0].numpy(), z[:, 1].numpy(), bins=50)
# 归一化为概率
hist = hist / hist.sum()
# 采样时用 2D 逆 CDF 或直接对直方图 cell 采样
```

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **组件未专一化时效果有限** | 如果组件 k 被 soft-EM 训练成"拟合所有 cluster"，其 latent 分布也会覆盖所有 cluster，ELD-S 效果退化 | 先用 K-Means 预划分（Idea 1）确保组件专一化，再应用 ELD-S |
| **维度独立假设** | 分位数是按维度独立估计的，不捕获维度间的相关性 | 对 dim=2 可升级为 2D 直方图；高维情况下维度独立仍然比 Uniform 有显著改善 |
| **分位数分辨率不足** | n_quantiles 太低时，CDF 近似粗糙，生成样本分布可能有轻微量化 artifact | 推荐 n_quantiles ≥ 200；对小数据集（N < 1000）可能精度有限 |
| **单 BF 场景的多 cluster latent 多峰** | 若单 BF 对多 cluster 数据训练，latent 分布会是多峰的，分位数仍能捕获这些峰 | 实际上这正是 ELD-S 的优势：多峰 latent 分布会在逆 CDF 采样中自然按峰密度加权 |
| **需要校准阶段** | 需要额外的一次正向传播（但无梯度，快） | 使用训练集本身，O(N) 计算一次即可，可缓存 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（升级替代 LZR）**

理由：
1. **无超参数**：不需要设置 percentile_low/percentile_high，自动从数据中学习
2. **数学严格性**：基于概率积分变换，理论保证从正确的经验分布采样
3. **适用范围更广**：同时适用于单 BreezeForest 和 MultiBF
4. **零训练成本**：只需一次正向传播（无梯度计算），10 秒内完成
5. **与 Stimper et al. (2022) 同思路但更简单**：不需要学习 base distribution，直接用数据的经验分布
6. **可与 K-Means 预划分叠加**：组件越专一，ELD-S 效果越好

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a.html  
  （ELD-S 是其无学习的简化版本）
- Coeurdoux, F. et al. (2023/2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 2024*. arXiv:2305.12149.  
  （同样的动机：在 latent 空间中引导采样，无需重训练）
- Bevins, H. & Handley, W. (2023). "Piecewise Normalizing Flows." arXiv:2305.02930.  
  （同文献支持：从 latent 空间角度修复 multi-cluster 生成）
- Devroye, L. (1986). "Non-Uniform Random Variate Generation." Springer.  
  （概率积分变换的经典数学基础）
- Pires, G. & Figueiredo, M. (2020). "Variational Mixture of Normalizing Flows." *ESANN 2020*.  
  （Latent space partitioning 的早期工作）
