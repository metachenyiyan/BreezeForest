# Idea: CDF-Space Anchor Partition Loss (CAPL)

**创建时间**: 2026-03-11 18:30 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（新架构设计，直接在训练中消除 latent 空间的 cluster 混淆；部分替代 ICDR 1240）

---

## 问题定义

BreezeForest 中 inter-cluster 生成问题的**最深层根源**在于 CDF 空间（latent space [0,1]^d）的结构性缺陷，而当前所有已有 idea 都没有从这个层面解决它：

**根本原因分析（从代码出发）：**

在 `MultiBF.inverse_map()`（`MultiBF.py` 第 140-171 行）中：
```python
z = torch.rand(n_k, self.dim) * 0.98 + 0.01  # z ~ Uniform([0.01, 0.99]^d)
x_k = self.components[k].inverse_map(z, ...)
```

这里假设 Uniform([0.01, 0.99]^d) 是组件 k 的合理 base distribution。**但这个假设只有在组件 k 的 forward 映射将其 cluster k 的数据均匀分散到整个 [0,1]^d 时才成立。**

实际情况是：
- 如果组件 k 用 soft-EM 训练（当前默认），它会尝试拟合所有 cluster 的数据（按 responsibility 加权）
- 组件 k 的 forward 映射会将**多个 cluster 的数据**映射到 [0,1]^d 中的**不同子区域**
- 不同子区域之间的连接区域（inter-cluster 路径在 [0,1]^d 中的映射）也被映射到某些 z 值
- 从 Uniform([0,1]^d) 采样，必然采到这些连接区域的 z 值，产生 inter-cluster 生成

**关键洞察**：如果我们能在**训练时**约束"组件 k 的 cluster k 数据只能映射到 [0,1]^d 的特定子区域 T_k"，那么 T_k 之外的 z 值就不会对应 cluster k 的数据，从 T_k 内均匀采样则完全避免 inter-cluster。

LZR（1235）和 KLDS（本轮）都是**推断时**发现并利用 Z_k，但 Z_k 的形状是由训练过程间接决定的。如果训练时施加约束，**主动将 cluster k 推入指定子区域 T_k**，则 Z_k ≈ T_k，从 T_k 采样是自然而准确的。

**ICDR（1240）的局限**：ICDR 在**数据空间**施加排斥，但不直接控制 [0,1]^d latent 空间的分布。一个组件的数据空间密度降低，不等价于其 latent 表示集中在指定区域。

---

## 从当前项目代码与已有 idea 中得到的背景判断

**代码结构支持：**

1. BreezeForest 的 forward 映射输出在 [0,1]^d（由 `Sigmoid` activation 保证，`TreeLayer.py` 第 67 行），latent 空间边界天然清晰。

2. `BreezeForest.train_forward()`（`BreezeForest.py` 第 130-162 行）在最大化 log|det J|。log|det J| 大的地方是高密度区域——如果我们额外约束"高密度区域必须在 T_k 内"，则可以在训练中实现 latent 分区。

3. `MultiBF.train_forward()` 已经有 per-sample log-det 计算（`_per_sample_log_det`），可以直接利用。

4. `MultiBF.get_mixture_weights()` 的 mixture logits 是可训练参数——在 CAPL 中，partition 与 mixture weights 可以协同训练。

**已有 idea 的判断：**

- Hard-EM（1230）：解决"组件训练谁的样本"问题，但不控制 latent 空间的形状
- LZR（1235）：推断时发现 Z_k，但 Z_k 是训练过程的副产品
- KLDS（本轮）：更好地利用 Z_k，但同样是推断时补救
- ICDR（1240）：数据空间排斥，不直接影响 latent 分布形状

**本 idea 的定位**：这是目前**唯一一个**直接在训练中约束 latent 空间分布形状的方案，填补了所有已有 idea 的空白。

**外部研究的验证：**

- Piecewise Normalizing Flows（Bevins 2023）实际上是此思路的极端版本：完全分离的 K 个流，每个只建模一个 cluster。但 PNF 不允许 end-to-end 联合优化。
- CAPL 是 PNF 的**软约束版本**：不强制分离，而是用一个可调节强度的 anchor loss 鼓励分离。这样可以保留 end-to-end 联合优化的优点（组件间可以共享特征），同时减少 inter-cluster z 值。

---

## 核心思路

**CDF-Space Anchor Partition Loss（CAPL）**：

### 预定义 Partition

将 [0,1]^d 预先划分为 K 个不重叠的子区域（"anchor region"）：

$$T_1, T_2, \ldots, T_K, \quad T_i \cap T_j = \emptyset, \quad \bigcup_k T_k \subseteq [0,1]^d$$

**最简单的划分（推荐）**：沿第一个维度均匀分割：
$$T_k = \left[\frac{k-1}{K}, \frac{k}{K}\right] \times [0,1]^{d-1}, \quad k = 1, \ldots, K$$

或者，在 K-Means 预聚类后，根据 cluster centers 在第一维度的排序来定义 T_k：
$$T_k = [\text{quantile}_{(k-1)/K}(z_0^{\text{all}}), \text{quantile}_{k/K}(z_0^{\text{all}})] \times [0,1]^{d-1}$$

### Anchor Loss

对每个组件 k，计算其 cluster k 数据的 forward 映射，并惩罚落在 T_k 之外的部分：

$$\mathcal{L}_{\text{anchor}} = \lambda_a \cdot \frac{1}{K} \sum_{k=1}^K \mathbb{E}_{x \sim \text{cluster}_k}\left[\text{dist}^2(f_k(x), T_k)\right]$$

其中 $\text{dist}(z, T_k) = \|z - \text{proj}_{T_k}(z)\|_2$（z 到 T_k 的欧氏距离），$\text{proj}_{T_k}(z)$ 是 z 在 T_k 上的投影（即将各维度 clamp 到 T_k 的边界）。

在实现上，"cluster k 的数据"使用 Hard-EM 或 k-means 分配：

$$\mathcal{L}_{\text{anchor}} = \lambda_a \cdot \frac{1}{K} \sum_{k=1}^K \frac{1}{|D_k|} \sum_{x_i \in D_k} \|f_k(x_i) - \text{clamp}(f_k(x_i), T_k)\|_2^2$$

### 总训练损失

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NLL}} + \lambda_a \cdot \mathcal{L}_{\text{anchor}}$$

### 生成时的协同使用

训练后，由于 anchor loss 的作用，组件 k 的 cluster k 数据集中在 T_k 中。生成时：
- 将 `z = torch.rand(n_k, self.dim) * 0.98 + 0.01` 改为从 T_k 中采样
- 或结合 KLDS，在 T_k 内做 KDE 精细采样

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**训练时直接约束 latent 空间形状：**

CAPL 的核心贡献是**主动改变 f_k 的训练目标**，不仅要求"好的密度拟合"（NLL），还要求"cluster k 的 latent 表示要集中在 T_k 内"。

具体机制：

1. 如果 $f_k(x_i) \notin T_k$（即 cluster k 的某个样本映射到了 T_k 之外），anchor loss 产生一个梯度，推动 $f_k$ 使得 $f_k(x_i)$ 向 T_k 内移动。

2. 由于 $f_k$ 是单调递增的 CDF，在 T_k 内积累更多概率质量，等价于：在 T_k 对应的 x 区域（cluster k）处，$f_k$ 的 Jacobian 更大（局部 CDF 斜率更大 = 局部密度更高）。

3. 训练完成后：$f_k$（组件 k 的 forward map）将 cluster k 的数据高密度映射到 T_k，其他 cluster 或 inter-cluster 区域的数据自然落在 $T_k$ 之外（因为 CDF 是单调的，高密度区域在 CDF 空间中占据特定位置）。

4. 生成时从 $T_k$ 均匀采样：$z \sim \text{Uniform}(T_k)$，$x = f_k^{-1}(z)$ 几乎必然落在 cluster k 附近。

**Inter-cluster 区域为什么被消除：**

设 cluster A 和 cluster B 是两个分离的 cluster，$T_A$ 和 $T_B$ 是不重叠的。

- Anchor loss 训练 $f_A$ 使 cluster A 数据 → $T_A$，$f_B$ 使 cluster B 数据 → $T_B$
- cluster A 和 cluster B 之间的 inter-cluster 区域（低密度）在 $f_A$ 下会映射到 $T_A$ 之外（因为 $f_A$ 的密度集中在 cluster A，inter-cluster 区域的 CDF 值变化缓慢，即对应 $T_A$ 中密度低的区域，而 cluster A 本身占据 $T_A$ 的主要部分）
- 从 $\text{Uniform}(T_A)$ 采样，最终的 x 值集中在 cluster A，与 inter-cluster 区域对应的 z 值比例极低

**与 ICDR 的对比：**

| 维度 | ICDR（1240）| CAPL（本 idea）|
|------|------------|----------------|
| 作用空间 | 数据空间（x 空间）| CDF/latent 空间（z 空间）|
| 约束方式 | 排斥：组件 j 密度在 cluster k 区域低 | 吸引：组件 k 的 latent 表示进入 T_k |
| 是否需要生成样本 | V1 版本需要（训练时 bisection），V2 不需要 | 不需要（只要 forward pass + dist 计算）|
| 对 BreezeForest 结构的适配 | 通用（适配所有 NF）| 专用（利用 [0,1]^d 的 CDF 结构）|
| 梯度来源 | log p_j(x_k) 对 θ_j 的梯度 | dist(f_k(x), T_k)^2 对 θ_k 的梯度 |
| 计算开销 | O(K^2) 密度评估 | O(K) forward pass + dist 计算 |

CAPL 计算开销更低（O(K) vs O(K^2)），且梯度信号更直接（告诉组件 k "你的 cluster 应该去哪里"，而不是"你不该在哪里"）。

---

## 与历史 idea 的关系

**部分替代 ICDR（1240）：**

ICDR 和 CAPL 都旨在减少组件在错误区域的密度，但机制不同：
- ICDR：在数据空间推开不同组件的密度区域
- CAPL：在 latent 空间拉入组件到指定分区

CAPL 对 BreezeForest 的 CDF 结构更友好：直接控制 forward 映射的输出位置，而不是间接通过数据空间密度。

**建议**：CAPL 可以单独使用，也可以与 ICDR-V2 叠加（先用 CAPL 约束 latent 分布，再用 ICDR 细化数据空间边界）。在实施成本有限时，优先 CAPL。

**互补 Phase-Locked Init（本轮 idea 1）和 KLDS（本轮 idea 2）：**

- Phase-Locked Init → 训练中通过数据分配保证组件专一化（样本级别）
- CAPL → 训练中通过 anchor loss 保证 latent 空间分区（latent 表示级别）
- KLDS → 推断时通过 KDE 精细采样（采样级别）

三者的组合是理论上最强的方案：
1. Phase-Locked Init 给 CAPL 提供正确的"cluster k 数据"来计算 anchor loss
2. CAPL 使 KLDS 的 Z_k 更接近 T_k（更规则、更易估计）
3. KLDS 在 T_k 内进一步精细采样

---

## 具体实现建议

### 步骤 1：定义 Partition

```python
def compute_partition_regions(n_components, dim, partition_dim=0):
    """
    Define K disjoint anchor regions T_k in [0,1]^d by splitting along partition_dim.
    
    T_k = [(k-1)/K, k/K] x [0,1]^(d-1)
    
    :return: List of (lo_k, hi_k) tensors, each shape (dim,)
    """
    regions = []
    for k in range(n_components):
        lo = torch.zeros(dim)
        hi = torch.ones(dim)
        lo[partition_dim] = k / n_components
        hi[partition_dim] = (k + 1) / n_components
        regions.append((lo, hi))
    return regions


def compute_data_aware_partition(mbf, x_train, assignments, partition_dim=0):
    """
    Define K anchor regions based on the actual latent distribution of each cluster.
    More adaptive: uses quantiles of forward-mapped cluster data to define T_k boundaries.
    
    :param assignments: k-means or hard-EM assignments (N,) int tensor
    """
    regions = []
    all_z0_values = []  # Collect z_0 values across all clusters for global quantile
    
    with torch.no_grad():
        cluster_z0 = []
        for k in range(mbf.n_components):
            mask = (assignments == k)
            if mask.sum() < 2:
                cluster_z0.append(torch.tensor([k / mbf.n_components, (k+1) / mbf.n_components]))
                continue
            x_k = x_train[mask]
            bf = mbf.components[k]
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list)
            z_k0 = z_k[:, partition_dim]
            cluster_z0.append(z_k0)
            all_z0_values.append(z_k0)
        
        all_z0 = torch.cat(all_z0_values)
        # Define boundaries from global quantiles
        for k in range(mbf.n_components):
            lo_val = float(torch.quantile(all_z0, k / mbf.n_components))
            hi_val = float(torch.quantile(all_z0, (k+1) / mbf.n_components))
            lo = torch.zeros(mbf.dim)
            hi = torch.ones(mbf.dim)
            lo[partition_dim] = max(lo_val, 0.01)
            hi[partition_dim] = min(hi_val, 0.99)
            regions.append((lo, hi))
    
    return regions
```

### 步骤 2：计算 Anchor Loss

```python
def compute_anchor_loss(mbf, x, assignments, partition_regions):
    """
    Compute the anchor loss: penalize each component's cluster data for 
    landing outside its assigned T_k in latent space.
    
    L_anchor = (1/K) * sum_k mean_{x in cluster_k} dist^2(f_k(x), T_k)
    
    :param x: training batch (batch_size, dim)
    :param assignments: hard assignments for batch (batch_size,) int tensor
    :param partition_regions: list of (lo_k, hi_k) tuples
    :return: scalar anchor loss
    """
    total_anchor = torch.tensor(0.0, requires_grad=False)
    n_active = 0
    
    for k in range(mbf.n_components):
        mask = (assignments == k)
        if mask.sum() == 0:
            continue
        
        x_k = x[mask]
        bf = mbf.components[k]
        
        # Forward pass for component k on its cluster's data
        breeze_list = []
        z_k = bf.forward(x_k, breeze_list)  # (n_k, dim)
        
        # Compute distance to T_k
        lo_k, hi_k = partition_regions[k]
        z_k_clamped = torch.clamp(z_k, min=lo_k, max=hi_k)
        dist_sq = torch.mean(torch.sum((z_k - z_k_clamped) ** 2, dim=1))
        
        total_anchor = total_anchor + dist_sq
        n_active += 1
    
    return total_anchor / max(n_active, 1)
```

### 步骤 3：集成到训练循环

```python
def train_with_capl(mbf, x, assignments, partition_regions, lambda_anchor=0.5, exact=False):
    """
    Training step with CDF-Space Anchor Partition Loss.
    
    :param assignments: hard assignments from k-means or nearest center
    :param partition_regions: pre-defined T_k regions
    :param lambda_anchor: anchor loss weight (0 = standard NLL, 1 = strong anchoring)
    :return: log_prob (for display), total_loss (for backward)
    """
    # Standard NLL loss
    log_prob = mbf.train_forward(x, exact=exact)
    nll_loss = -log_prob
    
    # Anchor loss (only in CDF space, no bisection needed)
    anchor_loss = compute_anchor_loss(mbf, x, assignments, partition_regions)
    
    total_loss = nll_loss + lambda_anchor * anchor_loss
    return log_prob, total_loss


# Training loop integration
# 1. Pre-compute k-means assignments (once before training)
assignments_all, centers = kmeans_init_assignments(all_data, n_components)
partition_regions = compute_partition_regions(n_components, dim=2)
# or: partition_regions = compute_data_aware_partition(mbf, all_data, assignments_all)

# 2. Training loop with CAPL
for index in range(ttl_iter):
    batch, _ = next(data_iter)
    batch = (batch - mean) / std
    
    # Get batch assignments (nearest center)
    dists = torch.cdist(batch, centers)
    batch_assignments = torch.argmin(dists, dim=1)
    
    # Ramp up lambda_anchor to avoid initial instability
    lambda_a = min(0.5, index / 1000 * 0.5)  # Linear ramp over 1000 steps
    
    log_prob, total_loss = train_with_capl(
        mbf, batch, batch_assignments, partition_regions, lambda_anchor=lambda_a
    )
    loss = total_loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 步骤 4：生成时配合 Partition

```python
def inverse_map_with_partition(mbf, n_samples, partition_regions, max_gap=1e-3):
    """
    Generate samples by sampling from T_k directly (no KDE needed).
    Combined with CAPL training, T_k ≈ Z_k.
    """
    weights = mbf.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, mbf.dim)
    
    for k in range(mbf.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        lo_k, hi_k = partition_regions[k]
        z = torch.rand(n_k, mbf.dim) * (hi_k - lo_k) + lo_k
        x_k = mbf.components[k].inverse_map(z, max_gap=max_gap)
        results[mask] = x_k
    
    return results
```

### 超参数建议

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| `lambda_anchor` | 0.1 – 1.0 | 建议从 0.5 开始；太大会影响 NLL 拟合质量 |
| `partition_dim` | 0 | 沿第一个维度分割通常效果最好（BreezeForest 的 autoregressive 方向）|
| lambda 调度 | 线性增大（0→target）| 前 1000 步线性增加，避免初始不稳定 |
| `compute_data_aware_partition` | 与 Phase-Locked Init 结合 | k-means 初始化后，用 data-aware partition 让边界更贴近实际 cluster 分布 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **NLL 质量下降** | Anchor loss 约束 f_k 的输出范围，可能干扰 NLL 最优化 | 监控 NLL 和 anchor loss；用小 lambda 或线性 ramp |
| **Partition 不匹配 cluster 形状** | 如果 cluster 不是线性可分的（在 latent 空间），均匀 partition 可能强迫错误分组 | 使用 data-aware partition（从 k-means 初始化后计算量化边界）|
| **多 cluster 单组件问题** | 如果某组件需要覆盖多个 cluster（n_components < n_clusters），anchor loss 会使训练冲突 | 确保 n_components ≥ n_clusters；或使用更大的 T_k |
| **第一维度分割的自回归假设** | BreezeForest 的 autoregressive 结构在 dim_0 上更精确，沿 dim_0 分割通常最有效 | 如果 dim_0 方向 cluster 重叠，尝试其他维度或多维度分割 |
| **Anchor Loss 的梯度方向** | `dist(f_k(x), T_k)^2` 的梯度会推动 f_k 的输出向 T_k 边界聚集（而不是中心） | 添加对 T_k 内部均匀分布的正则项（optional）；或增大 T_k 范围到 [(k-1)/K - ε, k/K + ε]（允许轻微重叠）|

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（训练时设计，长期最优解）**

理由：
1. **唯一训练时 latent 空间约束**：现有所有 idea 都在数据空间（Hard-EM, ICDR）或推断时（LZR, KLDS）工作；CAPL 是首个直接在 CDF latent 空间施加训练约束的方案
2. **直接针对根本原因**：通过 anchor loss 确保每个组件的 cluster 数据集中在 T_k，使生成时的 T_k 采样是自然而精确的
3. **与 BreezeForest 架构高度契合**：BreezeForest 的 CDF 输出天然在 [0,1]^d，分区极为自然；forward pass 直接可用，无需特殊修改
4. **计算开销低（O(K)）**：只需要对每个组件做一次 forward pass 加 dist 计算，比 ICDR 的 O(K^2) 更高效
5. **与 Phase-Locked Init 和 KLDS 协同**：三者组合是当前理论最强方案

**建议实施优先级：**
1. **Phase-Locked Init**（立即改善初始化）
2. **CAPL**（训练中固化 latent 分区）
3. **KLDS**（推断时精细采样）

CAPL 是三者中改动最大（需要重训练）但效果最持久的方案。在资源允许时，优先实施 CAPL。

---

## 参考文献

- Bevins, H. et al. (2023). "Piecewise Normalizing Flows." *arXiv:2305.02930*.  
  (CAPL 是 PNF 的"软约束"版本：用 penalty 替代完全分离，保留 end-to-end 联合优化)
- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.  
  (Base distribution 约束的理论来源；CAPL 是其训练时的对偶形式)
- Toth, P. et al. (2020). "Hamiltonian Generative Networks." *ICLR 2020*.  
  (通过 structured latent space 提升生成质量的先例)
- idea_inter_component_density_repulsion_2026-03-11-1240.md (BreezeForest project)  
  (ICDR idea that CAPL partially replaces; CAPL works in z-space vs ICDR's x-space)
- idea_latent_zone_restriction_2026-03-11-1235.md (BreezeForest project)  
  (LZR idea; CAPL makes T_k ≈ Z_k by training, making LZR/KLDS more accurate)
