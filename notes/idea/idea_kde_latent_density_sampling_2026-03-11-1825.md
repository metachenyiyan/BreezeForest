# Idea: KDE-Based Latent Density Sampling (KLDS)

**创建时间**: 2026-03-11 18:25 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（直接升级 LZR 1235，且无需重训练，即时可验证）

---

## 问题定义

LZR idea（`idea_latent_zone_restriction_2026-03-11-1235.md`）已经识别了核心问题：MultiBF 在生成时从 Uniform([0.01, 0.99]^d) 采样，但每个组件 k 的 cluster k 数据实际上只对应 [0,1]^d 的一个**子区域 Z_k**。现有 LZR 方案将 Z_k 近似为**矩形区域**（各维度独立的百分位数区间），但这个近似存在严重缺陷：

**矩形近似的具体问题：**

1. **忽略维度间相关性**：BreezeForest 是自回归流，dim_0 → dim_1 有 breeze 连接，导致 dim_0 和 dim_1 的 latent 表示 (z_0, z_1) 之间存在强相关性。矩形区域 [a_k^0, b_k^0] × [a_k^1, b_k^1] 会包含大量相关性之外的角落区域，这些角落在 Z_k 的实际分布中密度极低（接近零）。

2. **Inter-cluster z 值混入**：当两个 cluster 在数据空间中投影到某维度有重叠时，矩形区域会同时包含两个 cluster 的 z 值，导致 LZR 的过滤不完全。

3. **对 cluster 形状敏感**：月牙形、环形等非凸 cluster 在 [0,1]^d 中的 latent 表示可能是弯曲的流形，矩形区域会同时包含 cluster 内的点和 cluster 之间的空洞区域。

举例（2D BreezeForest，两个 cluster A 和 B）：
- 矩形 Z_A 的角落 (a_A^0 + small, b_A^1 - small) 可能对应 inter-cluster 区域
- 矩形 Z_A 可能与矩形 Z_B 重叠（特别是当两个 cluster 在某一维度上有重叠投影时）

**需要的是**：不是 Z_k 的矩形边界框，而是 Z_k 的**实际形状**（概率密度分布）。

---

## 从当前项目代码与已有 idea 中得到的背景判断

**代码层面关键观察：**

1. `MultiBF.inverse_map()`（`MultiBF.py` 第 140-171 行）：
   ```python
   z = torch.rand(n_k, self.dim) * 0.98 + 0.01  # 完全均匀采样
   ```
   采样从整个 [0.01, 0.99]^d 进行，完全不考虑 cluster k 的 latent 分布。

2. `BreezeForest.forward()`（`BreezeForest.py` 第 96-108 行）：forward 映射输出范围在 [0,1]^d 内（通过 sigmoid activation 保证），且在每个维度上是单调的。这意味着：
   - z 空间中，两个 cluster 的 latent 表示不会在同一维度内完全分离，但会有密度差异
   - 两个 cluster 的 z 表示在 z 空间中可能是**非线性分布**的（由于自回归结构和 sigmoid 压缩）

3. `BreezeForest.train_forward()` 使用的 `du_dx`（有限差分近似 Jacobian）：Jacobian 的量级直接反映了某个 z 值对应的数据密度。如果 z 处 Jacobian 大，说明对应数据稠密；如果 Jacobian 小，说明对应数据稀疏（即 inter-cluster 区域）。

4. LZR 的矩形近似可以在 Z_k 包含 inter-cluster 区域（Jacobian 小的区域）时产生无效样本——这正是症状描述中的问题。

**已有 idea 的判断：**

- LZR（1235）的**思路正确**，是目前最实用的无需重训练方案
- 但**矩形近似是其主要缺陷**：会遗留一部分 inter-cluster z 值
- LZR 本身提到了可以升级到"convex hull 或 PCA 主成分上的边界"，说明原作者也意识到了矩形的局限性
- **本 idea 是对 LZR 的直接升级**：用 KDE 替换矩形边界，使采样更精准

**外部研究的支持：**

- **Stimper et al. (2022) "Resampling Base Distributions"**：学习一个 rejection sampling 的 base distribution，使 normalizing flow 的采样避免 topology 问题。本 idea 是其**非参数化、无需额外训练**的简化版本。
- **Coeurdoux et al. (2024) "Normalizing Flow Sampling with Langevin Dynamics in the Latent Space"**（Machine Learning, 2024）：在 latent 空间用 Metropolis-Adjusted Langevin Algorithm（MALA）从目标分布采样，再做 inverse_map。本 idea 是其**简化版本**：用 KDE 估计目标分布 p̂_k(z)，用 rejection sampling 替代 MALA（不需要梯度）。

---

## 核心思路

**KDE-Based Latent Density Sampling（KLDS）**：

### 阶段 1：Post-Training Calibration（与 LZR 相同）

对训练后模型：
1. 计算所有训练样本对各组件的 responsibility
2. 对每个组件 k，选取 responsibility > 1/K 的样本（或 top-N% 样本）
3. 通过 f_k 正向传播，得到这些样本的 latent 表示：$\mathcal{Z}_k = \{f_k(x_i) : r_i^k > 1/K\}$

### 阶段 2：KDE 密度估计（替代矩形 Zone）

在 [0,1]^d 空间对 $\mathcal{Z}_k$ 拟合一个**核密度估计（KDE）**：

$$\hat{p}_k(z) = \frac{1}{|\mathcal{Z}_k|} \sum_{z_i \in \mathcal{Z}_k} K_h(z - z_i)$$

其中 $K_h$ 是 Gaussian kernel，带宽 $h$ 通过 Scott's rule 或 Silverman's rule 自动选取：
$$h = \left(\frac{4}{(d+2)|\mathcal{Z}_k|}\right)^{1/(d+4)} \cdot \hat{\sigma}$$

### 阶段 3：KDE-Guided Rejection Sampling（生成时）

对每个组件 k，使用 rejection sampling 从 $\hat{p}_k(z)$ 采样：

1. **Proposal**: z ~ Uniform([0.01, 0.99]^d)（或更窄的矩形范围，先做一次粗过滤）
2. **Acceptance**: u ~ Uniform(0, M_k)，接受 z 当且仅当 $\hat{p}_k(z) \geq u$，其中 $M_k = \max_{z \in \mathcal{Z}_k} \hat{p}_k(z)$
3. 接受的 z 做 inverse_map → x_k

**优化版本（两阶段 rejection sampling）：**

1. 第一阶段：用矩形 zone（LZR 的矩形）做粗过滤（高效）→ 产生候选 z
2. 第二阶段：用 KDE 对候选 z 做精细过滤 → 产生高质量 z
3. 在候选 z 上做 inverse_map

这避免了纯 Uniform → KDE rejection 的效率问题（矩形过滤消除大部分无效样本）。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**精确建模 Z_k 的实际形状**：

- 对于两个分离的 cluster A 和 B，$\mathcal{Z}_A$ 和 $\mathcal{Z}_B$ 是 [0,1]^d 中两个分离的高密度区域（density concentration）
- KDE 在 $\mathcal{Z}_A$ 的高密度区域内有高值，在 inter-cluster 区域（两者之间的低密度空间）接近零
- Rejection sampling 以接近零的概率接受来自 inter-cluster 区域的 z，有效阻断了 inter-cluster 生成路径

**数学直觉（相比矩形 zone 的改进）：**

矩形 zone：$z \in [a_k^0, b_k^0] \times [a_k^1, b_k^1]$ → 包含矩形的所有点，包括角落的低密度区域

KDE zone：$\hat{p}_k(z) \geq \theta$ → 只包含 $\mathcal{Z}_k$ 的"高密度核心"，自然排除低密度区域

对于自回归流（BreezeForest），z_0 和 z_1 之间有相关性，$\mathcal{Z}_k$ 在 (z_0, z_1) 空间中可能是一个斜向的椭圆或弧形——KDE 能捕捉这个形状，矩形不能。

**计算量分析：**

| 方法 | Calibration 时间 | 每次采样时间 | 内存 |
|------|-----------------|-------------|------|
| LZR（矩形）| O(N_train)| O(1)（直接采样）| O(K * d)（边界值）|
| KLDS（KDE）| O(N_train)| O(N_train) per sample（KDE 评估）| O(K * N_k)（存储 Z_k）|
| KLDS（两阶段）| O(N_train)| O(N_train / rejection_rate) << O(N_train)| O(K * N_k) |

KDE 评估可以用 scipy.stats.gaussian_kde 实现，N_train = 3000 时每次评估约 0.1-1ms，对采样 1000 个点完全可接受。

---

## 与历史 idea 的关系

**直接升级 LZR（1235）：**

| 对比维度 | LZR（1235）| KLDS（本 idea）|
|----------|------------|----------------|
| Zone 形状 | 矩形（各维度独立百分位数）| 自由形状（KDE 估计）|
| 维度相关性 | 忽略 | 自动捕捉（多元 KDE）|
| inter-cluster 残留 | 存在（矩形角落）| 极少（KDE 密度门槛过滤）|
| 实现复杂度 | 低 | 中（需要 KDE 库）|
| 采样效率 | O(1)（直接均匀采样）| O(N_train)（KDE 评估），两阶段版本更快 |
| 计算依赖 | 仅 PyTorch | 需要 scipy.stats.gaussian_kde |

**继承 LZR 的优点：**
- 同样是 post-training、无需重训练
- 同样使用 responsibility 确定组件分配
- 同样在 inverse_map 之前做 z 过滤

**与 Stimper 2022（Resampled Base Distributions）的关系：**
- 概念相同：在 latent 空间学一个更好的采样分布
- 本 idea 更简单：非参数 KDE，无需额外神经网络训练
- 本 idea 特化于 BreezeForest 的 [0,1]^d latent 结构

**与 Coeurdoux 2024（Langevin in Latent Space）的关系：**
- 理念相同：在 latent 空间中采样来自目标分布的 z，再做 inverse_map
- 本 idea 用 KDE + rejection sampling（更简单，不需要梯度/MCMC 收敛）
- MALA 可以作为后续升级（当 KDE 在高维 d 下效果变差时）

---

## 具体实现建议

### 步骤 1：扩展 MultiBF 添加 KDE calibration

```python
from scipy.stats import gaussian_kde
import numpy as np

def calibrate_kde_zones(self, x_train, responsibility_threshold=None, bandwidth_method='scott'):
    """
    Compute per-component KDE in latent space from training data.
    
    :param x_train: training data (N, dim)
    :param responsibility_threshold: samples with resp_k > threshold are used for KDE
                                     (default: 1/n_components)
    :param bandwidth_method: KDE bandwidth selection ('scott', 'silverman', or float)
    """
    if responsibility_threshold is None:
        responsibility_threshold = 1.0 / self.n_components
    
    self.kde_zones = []
    self.kde_max_density = []
    self.latent_zones = []  # 同时保留矩形 zone 供两阶段 rejection 使用
    
    with torch.no_grad():
        # 计算 responsibility（与 LZR 相同）
        log_pi = self.get_mixture_log_weights()
        component_log_probs = []
        for k, bf in enumerate(self.components):
            per_sample_ld = self._per_sample_log_det(bf, x_train)
            component_log_probs.append(log_pi[k] + per_sample_ld)
        
        stacked = torch.stack(component_log_probs, dim=0)  # (K, N)
        log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
        responsibilities = torch.exp(log_resp)  # (K, N)
        
        for k, bf in enumerate(self.components):
            resp_k = responsibilities[k]
            mask = resp_k > responsibility_threshold
            
            if mask.sum() < 10:
                topk = max(10, int(0.1 * len(resp_k)))
                _, idx = torch.topk(resp_k, topk)
                mask = torch.zeros_like(resp_k, dtype=torch.bool)
                mask[idx] = True
            
            # 正向映射：x → z（latent space）
            x_k = x_train[mask]
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list)  # (n_k, dim)
            z_k_np = z_k.cpu().numpy()  # shape: (n_k, dim)
            
            # 拟合多元 KDE（dim-dimensional）
            # gaussian_kde 要求 (dim, n_samples) 格式
            kde = gaussian_kde(z_k_np.T, bw_method=bandwidth_method)
            
            # 预计算最大密度（用于 rejection sampling 的 M_k）
            # 在 Z_k 样本点处评估，取最大值的 1.1 倍作为安全上界
            M_k = float(kde(z_k_np.T).max()) * 1.1
            
            self.kde_zones.append(kde)
            self.kde_max_density.append(M_k)
            
            # 同时保存矩形 zone（用于两阶段的粗过滤）
            lo = torch.quantile(z_k, 0.02, dim=0).clamp(min=0.01)
            hi = torch.quantile(z_k, 0.98, dim=0).clamp(max=0.99)
            self.latent_zones.append((lo, hi))
    
    print(f"KDE calibration completed for {len(self.kde_zones)} components")
```

### 步骤 2：KDE-Guided inverse_map

```python
def inverse_map_kde(self, n_samples, max_gap=1e-3, decay_ratio=1.0, max_rejection_rounds=10):
    """
    Generate samples using KDE-based latent sampling.
    Two-stage: rectangular pre-filter + KDE rejection.
    
    :param n_samples: number of samples to generate
    :param max_rejection_rounds: maximum rejection sampling rounds
    :return: generated samples (n_samples, dim)
    """
    assert hasattr(self, 'kde_zones'), "Call calibrate_kde_zones() first"
    
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)
    
    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        kde_k = self.kde_zones[k]
        M_k = self.kde_max_density[k]
        lo_k, hi_k = self.latent_zones[k]  # 矩形粗过滤
        
        accepted_z = []
        remaining = n_k
        
        for _ in range(max_rejection_rounds):
            # 阶段 1：矩形粗过滤（LZR 方法）
            n_proposal = remaining * 5  # 过采样 5x，期望至少 remaining 个通过 KDE
            z_proposal = torch.rand(n_proposal, self.dim) * (hi_k - lo_k) + lo_k
            
            # 阶段 2：KDE 精细过滤
            z_np = z_proposal.cpu().numpy()
            densities = kde_k(z_np.T)  # shape: (n_proposal,)
            u = np.random.uniform(0, M_k, n_proposal)
            accept_mask = densities >= u
            
            z_accepted = z_proposal[torch.tensor(accept_mask)]
            accepted_z.append(z_accepted)
            collected = sum(a.shape[0] for a in accepted_z)
            
            if collected >= remaining:
                break
        
        # 合并并截取所需数量
        if accepted_z:
            z_final = torch.cat(accepted_z, dim=0)[:n_k]
        else:
            # Fallback to rectangular zone if rejection fails
            z_final = torch.rand(n_k, self.dim) * (hi_k - lo_k) + lo_k
        
        x_k = self.components[k].inverse_map(z_final, max_gap=max_gap, decay_ratio=decay_ratio)
        results[mask] = x_k
    
    return results
```

### 步骤 3：在 demo_multi_bf.py 中集成

```python
# 训练完成后：
# 1. KDE 校准（包含矩形 zone 的双阶段校准）
all_batch = get_all_training_data()  # 获取全量训练数据
all_batch = (all_batch - mean) / std
with torch.no_grad():
    mbf.calibrate_kde_zones(all_batch, bandwidth_method='scott')

# 2. KDE 引导生成
mbf.eval()
with torch.no_grad():
    samples = mbf.inverse_map_kde(n_samples=data_size)
    samples = samples * std + mean
```

### 带宽调优建议

| 带宽方法 | 适用场景 | 说明 |
|----------|---------|------|
| `'scott'` | 默认推荐 | Scott's rule，自动适配样本量和维度 |
| `'silverman'` | 样本量较大时 | 略小于 Scott's，更精细 |
| `0.05`（自定义）| cluster 形状复杂 | 较小带宽，捕捉精细结构；可能过拟合 |
| `0.2`（自定义）| cluster 较稀疏 | 较大带宽，更保守 |

推荐从 `'scott'` 开始，如果 inter-cluster 残留仍然可见，尝试减小带宽。

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **高维 KDE 退化** | d > 5 时 KDE 受"维度灾难"影响，密度估计不准 | BreezeForest 通常 d = 2，无此问题；高维时改用 Normalizing Flow 做 KDE |
| **Rejection rate 低** | 当 KDE 与矩形 zone 差异大时，rejection rate 高，采样慢 | 增大 max_rejection_rounds；使用两阶段（矩形粗滤 + KDE 精滤）减少无效评估 |
| **KDE 过拟合** | 带宽过小时，KDE 在训练点周围有尖峰，sampling 偏向训练样本 | 用 cross-validation 选带宽；或用 scott/silverman 自动规则 |
| **组件未专一化时 KDE 失效** | 如果 responsibility 边界模糊，Z_k 包含多个 cluster 的 z → KDE 有多峰 | 先运行 Phase-Locked Init（本轮 idea 1），保证专一化后再做 KDE |
| **scipy 依赖** | 需要 scipy.stats.gaussian_kde | scipy 是常见依赖，应已在环境中 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（与 Phase-Locked Init 并列）**

理由：
1. **无需重训练**：可在当前已训练的 MultiBF 上立即应用，验证速度快
2. **精度显著高于 LZR**：KDE 自动捕捉 latent zone 的实际形状，包括维度间相关性
3. **数学上有理论保证**：等价于 Stimper 2022 的 resampled base distributions 的非参数化版本
4. **可与 Phase-Locked Init 完美叠加**：先用 Phase-Locked Init 训练得到专一化组件，再用 KLDS 采样 → 双重保障
5. **实现已有工具支持**：scipy.stats.gaussian_kde 成熟，项目中 sklearn 已有使用

**建议验证顺序：**
1. 先用 KLDS 在现有已训练模型上验证（快速验证，无需重训练）
2. 如果效果仍不满意，说明组件专一化不足 → 改用 Phase-Locked Init 重训练
3. 重训练后再次应用 KLDS → 理论最优组合

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*, PMLR 151:4915-4936. https://proceedings.mlr.press/v151/stimper22a.html  
  (直接理论来源：在 latent 空间学习更好的 base distribution)
- Coeurdoux, F. et al. (2024). "Normalizing Flow Sampling with Langevin Dynamics in the Latent Space." *Machine Learning*, Springer 2024. https://link.springer.com/article/10.1007/s10994-024-06623-x  
  (MALA 在 latent 空间采样：本 idea 是其 rejection-sampling 简化版)
- Scott, D.W. (1992). *Multivariate Density Estimation: Theory, Practice, and Visualization*. Wiley.  
  (Scott's rule for KDE bandwidth selection)
- idea_latent_zone_restriction_2026-03-11-1235.md (BreezeForest project)  
  (LZR idea that this document upgrades)
