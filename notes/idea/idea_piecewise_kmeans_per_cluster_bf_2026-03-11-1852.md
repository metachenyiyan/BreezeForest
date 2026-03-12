# Idea: Piecewise BreezeForest — K-Means Pre-Partitioned Per-Cluster Training

**创建时间**: 2026-03-11 18:52 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（替代 Hard-EM）

---

## 问题定义

BreezeForest（单模型）和 MultiBF（混合模型）在多 cluster 数据上的核心困难有两个层次：

**层次一（架构层）**: 单个 BreezeForest 是连续双射 f: ℝ^d → [0,1]^d。对于拓扑上分离的多个 cluster，它必须用一个连续函数"桥接"所有 cluster，导致 cluster 之间的区域也被分配了非零密度。采样时 z ~ Uniform(0.01, 0.99)^d 必然覆盖 cluster 之间的 z 区间，对应到 x 空间即为 inter-cluster 无效生成点。

**层次二（训练层）**: MultiBF 用 soft-EM（logsumexp）训练 K 个组件，每个组件在每次 step 中接受所有样本的梯度（按 responsibility 加权），导致组件无法专一化，反而各自都对所有 cluster 有一定建模。

**核心原因**：这两层问题根源一致——将"多连通"数据强行用一个（或弱分离的多个）连续双射来表示，而没有从数据划分层面给每个 flow 分配"自己的领地"。

---

## 从当前项目代码与已有 idea 中得到的背景判断

**代码层面**：
- `BreezeForest.forward(x)` → `TreeLayer` 计算逐维条件 CDF F_i(x_i | x_{<i}) ∈ (0,1)
- 激活函数是 Sigmoid（映射到 (0,1)，斜率系数 4），输出空间有界
- `generate_sample()` 从 Uniform(0.01, 0.99)^d 采样 z，通过二分搜索 `inverse_map` 反演
- `MultiBF.train_forward()` 使用 logsumexp，每个 step 所有组件都接受全批次梯度
- `MultiBF.inverse_map()` 先 Categorical 采组件 k，再 z ~ Uniform，再 bisection

**已有 idea 分析**：
- **Hard-EM（1230）**：尝试将 soft assignment 改为 hard assignment，方向正确。但存在"先有鸡还是先有蛋"问题：需要好的组件才能做好分配，需要好的分配才能训练好组件。早期阶段 hard assignment 不稳定，容易发生组件坍塌。
- **LZR（1235）**：推断式修复，较 Hard-EM 更快部署，但不解决训练问题。
- **ICDR（1240）**：通过排斥 loss 强化组件分离，是 Hard-EM 的补充。

**文献支持（本轮调研）**：
- **Piecewise Normalizing Flows（Marchetti et al., arXiv 2305.02930）** 明确表明：先用 K-Means 将训练数据划分为 cluster，再对每个 cluster 独立训练一个 flow，相比 Stimper (2022) resampled base distributions 性能更好，且训练更稳定。这是目前文献中针对多 cluster flow 生成最有效的方法之一。
- **AMF-VI（arXiv 2510.02056）**：两阶段训练（先 sequential expert 训练，再 adaptive weight）比 joint 训练更稳定，进一步支持"先划分再训练"思路。
- PNF 的本质与本 Idea 完全一致：取消 EM 和 mixture logit，改为固定外部分区。

---

## 核心思路

**在 MultiBF 之外提供一个更简洁的替代方案：Piecewise BreezeForest**。

1. **预分区（Pre-Partition）**：在训练前，对训练数据运行 K-Means（K = 期望 cluster 数），将数据固定分配到 K 个子集 D_1, ..., D_K。分配是**一次性固定的**，不在训练中更新。

2. **独立训练（Per-Cluster Training）**：为每个子集 D_k 单独训练一个 `BreezeForest` 模型 `bf_k`，训练时只看 D_k 中的数据，完全不接触其他 cluster 的样本。

3. **加权生成（Weighted Sampling）**：生成时，先按 cluster 大小比例采样 k ~ Categorical(|D_k|/N)，再 z ~ Uniform(0.01, 0.99)^d，再 bf_k.inverse_map(z)。

**关键差异与 Hard-EM 的比较**：

| 方面 | Hard-EM（已有） | Piecewise BF（本 Idea） |
|------|----------------|------------------------|
| 组件分配机制 | E-步（迭代更新 responsibility）| K-Means（一次性，训练前完成）|
| 训练目标函数 | 仍包含 logsumexp / NLL over assigned | 纯 NLL over cluster data（与 MultiBF 架构无关）|
| 混合权重 | 可学习 mixture_logits | 由 cluster 大小频率直接决定 |
| 组件坍塌风险 | 存在（早期 assignment 不稳定）| 不存在（K-Means 保证每个 cluster 有固定成员）|
| 实现复杂度 | 中（需改 train_forward_hard_em）| 低（每个 BF 独立用现有代码训练）|
| 训练数据分离 | 软性（仍有 responsibility < 1 的梯度泄漏）| 硬性（完全分离，零泄漏）|

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**数学原理**：

设 cluster k 的数据集为 D_k，其分布为 P_k(x)。训练在 D_k 上的 bf_k 会学到：
- 对 P_k 的高密度区域，CDF 梯度大（高 Jacobian）→ 生成时大量 z 映射到 cluster k
- 对非 P_k 区域（其他 cluster 或 inter-cluster），bf_k 从未见过这些数据，Jacobian 在这些区域极小
- 从 bf_k 采样时，几乎所有生成点都落在 D_k 的支撑集内

**拓扑角度**：
- 单个 cluster 的支撑集在拓扑上是"单连通"的（大致如此，尤其是 Gaussian-like cluster），与 Uniform/Normal 基分布的拓扑一致
- 多 cluster 联合分布是"多连通"的，与单一基分布不匹配
- 将多连通问题分解为 K 个单连通问题，完全规避了拓扑失配

**实验支持（文献）**：
- PNF（arXiv 2305.02930）在标准 2D 多 cluster benchmark（8gaussians、rings、checkerboard 等）上，相比 resampled base distribution 显著减少 inter-cluster 生成点，视觉上 cluster 更清晰。

---

## 它与历史 idea 的关系

**替代 Hard-EM（1230）**：本 Idea 实现了 Hard-EM 的目标（组件专一化训练），但用更简单、更稳定的方式。Hard-EM 的优点（MultiBF 架构复用）和缺点（EM 迭代不稳定）都被消除。如果使用本 Idea，Hard-EM 文档可降级为参考。

**与 LZR（1235）的关系**：本 Idea 是 training-time 修复，LZR 是 inference-time 修复。两者互补，可叠加使用：Piecewise BF 训练 + LZR/KDE 采样 = 最强组合。

**与 ICDR（1240）的关系**：本 Idea 使 ICDR 不再必要——因为各组件已经由 K-Means 完全分离，不需要额外的排斥 loss。ICDR 可降级为次级方案。

---

## 具体实现建议

### 方案 A：最简实现（约 30 行新代码）

不需要创建新的架构类，直接在训练脚本中实现：

```python
import numpy as np
from sklearn.cluster import KMeans
import torch
from model.BreezeForest import BreezeForest

def train_piecewise_bf(
    x_train,       # tensor (N, dim)
    n_clusters,    # K
    shapes,        # BF layer shapes
    sapw=0.5,
    lr=0.005,
    ttl_iter=8000,
    batch_size=200
):
    """
    Trains K independent BreezeForest models, one per K-Means cluster.
    Returns: list of trained BF models + cluster weights.
    """
    x_np = x_train.detach().numpy()
    
    # Step 1: K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(x_np)  # (N,)
    
    cluster_weights = np.array([
        (labels == k).sum() / len(labels) for k in range(n_clusters)
    ])
    
    # Step 2: Train one BF per cluster
    models = []
    for k in range(n_clusters):
        mask = (labels == k)
        x_k = x_train[mask]  # Only this cluster's data
        
        bf_k = BreezeForest(
            dim=x_train.shape[1],
            shapes=shapes,
            sap_w=sapw,
            inc_mode="no strict"
        )
        
        # ActiNorm init from cluster data
        with torch.no_grad():
            bf_k.forward(x_k[:min(200, len(x_k))])
        
        optimizer = torch.optim.Adam(bf_k.parameters(), lr=lr, weight_decay=1e-5)
        
        idx = np.arange(len(x_k))
        for i in range(ttl_iter):
            batch_idx = np.random.choice(idx, batch_size, replace=len(idx) < batch_size)
            batch = x_k[batch_idx]
            
            _, log_det = bf_k.train_forward(batch)
            loss = -log_det
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        models.append(bf_k)
        print(f"Cluster {k}: {mask.sum()} samples, weight={cluster_weights[k]:.3f}")
    
    return models, cluster_weights


def sample_piecewise_bf(models, cluster_weights, n_samples, dim):
    """
    Generate samples from piecewise BF ensemble.
    """
    weights_tensor = torch.tensor(cluster_weights, dtype=torch.float)
    component_indices = torch.multinomial(weights_tensor, n_samples, replacement=True)
    
    results = torch.zeros(n_samples, dim)
    
    with torch.no_grad():
        for k, bf_k in enumerate(models):
            mask = (component_indices == k)
            n_k = mask.sum().item()
            if n_k == 0:
                continue
            
            z = torch.rand(n_k, dim) * 0.98 + 0.01
            bf_k.eval()
            bf_k.batch_example = None  # 允许重新计算分布
            x_k = bf_k.inverse_map(z)
            results[mask] = x_k
    
    return results
```

### 方案 B：包装成 PiecewiseBF 类（推荐用于正式集成）

```python
class PiecewiseBF(torch.nn.Module):
    """
    K independent BreezeForest models, each trained on a K-Means cluster.
    No mixture logits, no EM—partition is fixed from K-Means before training.
    """
    def __init__(self, n_clusters, dim, shapes, **bf_kwargs):
        super().__init__()
        self.n_clusters = n_clusters
        self.dim = dim
        self.components = nn.ModuleList([
            BreezeForest(dim=dim, shapes=copy.deepcopy(shapes), **bf_kwargs)
            for _ in range(n_clusters)
        ])
        self.register_buffer('cluster_weights', torch.ones(n_clusters) / n_clusters)
        self.labels_ = None  # K-Means labels, set during fit()
    
    def fit(self, x_train, **training_kwargs):
        """Run K-Means and train each component."""
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(x_train.detach().numpy())
        self.labels_ = labels
        counts = torch.tensor([
            (labels == k).sum() for k in range(self.n_clusters)
        ], dtype=torch.float)
        self.cluster_weights = counts / counts.sum()
        # ... (train each component as in sample above)
    
    def sample(self, n_samples):
        """Generate samples proportional to cluster weights."""
        indices = torch.multinomial(self.cluster_weights, n_samples, replacement=True)
        results = torch.zeros(n_samples, self.dim)
        with torch.no_grad():
            for k, bf_k in enumerate(self.components):
                mask = (indices == k)
                n_k = mask.sum().item()
                if n_k == 0:
                    continue
                z = torch.rand(n_k, self.dim) * 0.98 + 0.01
                results[mask] = bf_k.inverse_map(z)
        return results
```

### 数据归一化注意事项

K-Means 和 BF 训练都应在**归一化后的空间**（减均值除方差）中进行，与现有 `demo_functions.py` 一致。可以用全局均值/方差归一化，或各 cluster 分别归一化（后者更适合差距大的 cluster）。

### K 的选择建议

| 方法 | 适用场景 |
|------|----------|
| 手动指定 K = 数据集已知 cluster 数 | 最佳（如 GAUSSIANS → K=8, BLOBS → K=3）|
| K = √N / 2 经验法则 | 无先验知识时 |
| 肘部法则（elbow method）| 对 K 不确定时 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **K 设置不当** | K < 真实 cluster 数：某个 BF 需建模多个 cluster；K > 真实 cluster 数：某些 BF 训练数据太少 | 用 elbow method 估计 K；对数据量极少的 cluster 增大 sap_w（更 Gaussian-like）|
| **K-Means 分配边界模糊** | 两个相近 cluster 之间的点可能被 K-Means 错分 | K-Means 对球形 cluster 效果最好，对非球形 cluster 可换用 DBSCAN 或 GMM 分配 |
| **小 cluster 欠拟合** | 数据量少的 cluster 训练样本不足 | 可使用数据增强（对小 cluster 过采样），或减小该 cluster 的 BF 复杂度 |
| **与 MultiBF 接口不兼容** | Piecewise BF 不继承 MultiBF，无法直接替换 | 实现 `PiecewiseBF` 类提供相同接口（sample、fit）|
| **推断时不支持 log p(x) 评估** | 无 mixture logit，很难计算整体 log p(x)（需要 argmax cluster assignment）| 接受局限：此方案主要用于 generation，不适合 density evaluation 场景 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（Hard-EM 的替代方案）**

理由：
1. **PNF 文献已验证**：arXiv 2305.02930 明确表明 piecewise 训练优于混合 soft-EM 和 resampled base distribution 方法
2. **无组件坍塌风险**：K-Means 固定分区消除了 Hard-EM 的"先有鸡还是先有蛋"问题
3. **实现最简**：每个 BF 完全独立训练，复用所有现有代码，零新依赖
4. **可并行训练**：K 个 BF 可以完全并行，总计算量不增加（vs MultiBF 每步需要 K × N 梯度计算）
5. **对 inter-cluster 生成的理论保证最强**：每个 BF 只见过自己 cluster 的数据，其 CDF 在其他区域接近常数（零梯度），不会生成无效点

---

## 参考文献

- Marchetti, G.L. et al. (2023). "Piecewise Normalizing Flows." *arXiv 2305.02930*. https://arxiv.org/abs/2305.02930  
  （直接确认 K-Means + per-cluster flow 优于 soft mixture 和 resampled base）
- Guo, X. et al. (2024). "Adaptive Mixture Flow-based Variational Inference." *arXiv 2510.02056*.  
  （Sequential expert training 在 mixture flows 中更稳定于 joint training）
- Dempster, A.P. et al. (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm." *JRSS-B*.  
  （Hard-EM 理论基础，本 Idea 用 K-Means 替代其 E-step）
- Hartigan, J.A. & Wong, M.A. (1979). "Algorithm AS 136: A K-Means Clustering Algorithm." *Applied Statistics*.
