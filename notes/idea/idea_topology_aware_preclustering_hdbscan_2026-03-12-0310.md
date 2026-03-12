# Idea: Topology-Aware Pre-Clustering via HDBSCAN / Spectral Methods (TAPC)

**创建时间**: 2026-03-12 03:10 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（对 K-Means Pre-Init 的结构性升级，处理非凸 cluster 的关键缺失）

---

## 问题定义

当前 K-Means Pre-Init（2026-03-12 01:51）将 K-Means 作为组件预初始化的聚类算法。这在以下常见场景下会**系统性失败**：

1. **非凸 cluster（BreezeForest 现有数据集中的 MOONS、CIRCLES、SPIRALS）**：K-Means 假设每个 cluster 是球形对称的，对 half-moon 形、环形、螺旋形等非凸数据集，K-Means 会将一个连续 cluster 拆成多个片段，或把两个不同 cluster 合并到一起。
2. **cluster 大小严重不均**：K-Means 倾向于产生大小均等的 cluster，对不均匀数据的分配准确率低。
3. **K 未知**：K-Means 要求预先指定 n_components = K，但实际 cluster 数量可能与模型组件数不同。

**当前 K-Means Pre-Init 的失败模式**（直接影响 multi-cluster 中间点生成问题）：
- 若 K-Means 把 cluster A 分配给了组件 0，但把 cluster B 的一半也分配给了组件 0（因为 cluster B 不是球形），则组件 0 的 warm-start 仍然是多 cluster 数据 → warm-start 无效
- 若 K-Means 过度切分某个非凸 cluster，导致两个组件各自学习同一 cluster 的一半 → 这两个组件仍会在两半之间生成中间点

---

## 从代码与已有 Idea 中得到的背景判断

**代码分析**（`model/distribution2d.py`，`demo_functions.py`，`demo_multi_bf.py`）：

BreezeForest 项目目前使用的数据集：
- `GAUSSIANS`（8 个高斯团，凸集合，K-Means 基本有效）
- `MOONS`（两个月牙形，**高度非凸**，K-Means 会切割月牙）
- `CIRCLE`（两个同心圆，**拓扑不同**，K-Means 完全失效）
- `SPIRALS`（双螺旋，**极度非凸**，K-Means 无用）
- `BLOBS`（sklearn blobs，基本凸集合，K-Means 有效）

**项目内分析**：K-Means Pre-Init（2026-03-12 01:51）在"潜在风险"中明确提到：
> "K-Means 假设球形 cluster，对非凸或大小差异极大的 cluster 效果差 — 尝试 DBSCAN、GMM 聚类"

但该文档没有提出具体的替代方案。本 Idea 将这个"可选建议"发展为完整独立方案。

**外部研究支撑**：
- **PNF（Bevins et al., 2023, arxiv 2305.02930）**：测试了 K-Means vs Mean Shift vs BIRCH，得出 K-Means 在大多数情况下最优——但他们没有测试 HDBSCAN 或 Spectral Clustering，且他们的测试数据集没有包含 non-convex cluster。
- **HDBSCAN（McInnes et al., 2017）**：密度连通聚类，不假设球形，自动确定 cluster 数（将噪声点标记为 -1），在非凸和不规则形状 cluster 上是 SOTA。
- **Spectral Clustering（Von Luxburg, 2007）**：基于相似度图的聚类，通过图拉普拉斯特征向量识别 cluster 形状，可以正确切分 MOONS 和 CIRCLES。
- **HDBSCAN + 流模型（BootSC 2025, arxiv 2508.04200）**：端到端深度谱聚类，已被证明处理非凸 cluster 的效果显著优于 K-Means。

---

## 核心思路

将 K-Means Pre-Init 中的聚类算法从 **K-Means** 升级为**数据拓扑感知的聚类方法**，同时增加**自动 K 选择**能力：

**Phase 0：自适应聚类算法选择**
根据数据特征选择最合适的聚类算法：
- 数据维度 ≤ 5、cluster 形状未知：优先用 **HDBSCAN**（不需要指定 K，自动发现 cluster 结构）
- 数据维度 ≤ 20、K 约等于 n_components：用 **Spectral Clustering**（K 需要指定，但效果好于 K-Means）
- 数据维度高、K 已知：fallback 用 **K-Means**（与原方案一致）

**Phase 1：聚类执行**
- HDBSCAN：自动发现 n_clusters，噪声点单独处理
- Spectral Clustering：指定 n_clusters = n_components，用 RBF kernel
- K-Means：原方案

**Phase 2：Cluster 到 Component 的映射**
- 若 n_clusters ≠ n_components：
  - n_clusters > n_components：将小 cluster 合并到最近的大 cluster（按 cluster centroid 距离）
  - n_clusters < n_components：将大 cluster 切分成多个 component（用 K-Means 在该 cluster 内部做子聚类）
- 若 n_clusters == n_components：直接一一映射

**Phase 3 & 4**：与原 K-Means Pre-Init 一致（per-component ActiNorm init + warm-start）

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**直接因果链**：

1. 如果聚类算法错误地将不同形状的 cluster 切割或合并 → per-component warm-start 的训练数据本身就包含多个 cluster 的点 → warm-start 仍然训练出一个多 cluster 响应的组件 → DAEM 退火也无法完全修复这个初始化错误

2. 如果聚类算法正确识别了非凸 cluster 的真实形状（如月牙、环形）→ 每个组件 warm-start 的数据是单一、干净的 cluster → warm-start 后组件高度专一 → DAEM 退火只需微调，不需要从头建立分工 → 大幅减少 inter-cluster 生成

**量化分析**（基于 K-Means vs HDBSCAN 在 MOONS 数据集上的预期行为）：
- K-Means (K=2)：会将每个月牙形切成两半（因为月牙的两端在欧氏空间中距离相近），导致 4 个 cluster 片段被错误分配给 2 个组件
- HDBSCAN：会正确识别两个完整月牙作为 2 个 cluster，组件分配干净

---

## 与历史 idea 的关系

| 历史 Idea | 关系 | 说明 |
|-----------|------|------|
| **K-Means Pre-Init（2026-03-12 01:51）** | **结构性升级（不是完全替代）** | 本 Idea 是 K-Means Pre-Init 的直接改进版本。K-Means Pre-Init 的"潜在风险"章节已指出 K-Means 对非凸 cluster 无效并建议 DBSCAN；本 Idea 将这个建议实现为完整方案，加入算法选择逻辑、K 不匹配处理、HDBSCAN 自动 K 选择。对于凸形 cluster（GAUSSIANS、BLOBS），K-Means Pre-Init 仍然有效；本 Idea 将 K-Means 保留为 fallback。 |
| **DAEM（2026-03-12 01:51）** | 前置配套 | TAPC 是 DAEM 的前置步骤；更好的聚类 → 更好的 warm-start 初始化 → DAEM 退火从更干净的起点出发，更快收敛，组件坍塌风险更低 |
| **Latent GMM Resampling（2026-03-12 01:51）** | 间接改善 | 更好的组件专一化 → latent space 中每个组件对应的 z_k 分布更集中 → Latent GMM 拟合更准确 |
| **ICDR（2026-03-11 12:40）** | 减少必要性 | 更好的初始化后，组件已经分离，ICDR 的补充作用进一步减小 |

**与 Hard-EM（2026-03-11 12:30）的关系**：Hard-EM 已被 DAEM 替代，本 Idea 不影响该替代关系。

---

## 具体实现建议

### 完整 TAPC 初始化函数

```python
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def topology_aware_preclustering(
    mbf,
    x_train,
    n_warmup_steps=1500,
    warmup_lr=0.005,
    batch_size=64,
    method='auto',          # 'auto', 'hdbscan', 'spectral', 'kmeans'
    min_cluster_size=None,  # HDBSCAN parameter: min samples per cluster
    spectral_affinity='rbf' # spectral clustering kernel
):
    """
    Topology-Aware Pre-Clustering initialization for MultiBF.
    Automatically selects clustering algorithm based on data and method parameter.
    """
    K = mbf.n_components
    x_np = x_train.detach().cpu().numpy()
    n_samples, dim = x_np.shape
    
    if min_cluster_size is None:
        min_cluster_size = max(10, n_samples // (K * 5))
    
    # ===== Phase 0: Algorithm Selection =====
    if method == 'auto':
        if dim <= 10:
            method = 'hdbscan'  # density-connected clustering for low-dim non-convex data
        else:
            method = 'spectral'  # spectral for medium-dim
    
    # ===== Phase 1: Clustering =====
    print(f"[TAPC] Running {method.upper()} clustering on {n_samples} samples...")
    
    if method == 'hdbscan':
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=None,
                metric='euclidean'
            )
            labels = clusterer.fit_predict(x_np)  # -1 = noise
            n_found = len(set(labels)) - (1 if -1 in labels else 0)
            noise_mask = (labels == -1)
            print(f"[TAPC] HDBSCAN found {n_found} clusters, "
                  f"{noise_mask.sum()} noise points")
        except ImportError:
            print("[TAPC] hdbscan not installed, falling back to spectral")
            method = 'spectral'
    
    if method == 'spectral':
        from sklearn.cluster import SpectralClustering
        sc = SpectralClustering(
            n_clusters=K,
            affinity=spectral_affinity,
            n_init=10,
            random_state=42
        )
        labels = sc.fit_predict(x_np)
        n_found = K
        noise_mask = np.zeros(n_samples, dtype=bool)
        print(f"[TAPC] SpectralClustering completed, {K} clusters")
    
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=10, random_state=42)
        labels = km.fit_predict(x_np)
        n_found = K
        noise_mask = np.zeros(n_samples, dtype=bool)
        print(f"[TAPC] KMeans completed, {K} clusters")
    
    # ===== Phase 2: Map n_found clusters -> K components =====
    labels = _remap_clusters_to_components(labels, x_np, K, noise_mask)
    
    cluster_sizes = [(labels == k).sum() for k in range(K)]
    print(f"[TAPC] Component assignment sizes: {cluster_sizes}")
    
    # ===== Phase 3: Per-Component ActiNorm Initialization =====
    for k, bf in enumerate(mbf.components):
        mask = (labels == k)
        x_k = x_train[mask]
        if len(x_k) < 5:
            with torch.no_grad():
                bf.forward(x_train)
        else:
            for layer in bf.treeLayers:
                layer.treeBias = None
                layer.treeScale = None
            with torch.no_grad():
                bf.forward(x_k)
            print(f"  Component {k}: ActiNorm on {len(x_k)} samples")
    
    # Initialize mixture logits proportional to cluster sizes
    with torch.no_grad():
        for k in range(K):
            mbf.mixture_logits.data[k] = torch.log(
                torch.tensor(max(cluster_sizes[k], 1) + 1e-8, dtype=torch.float32)
            )
    
    # ===== Phase 4: Per-Component Warm-Start Training =====
    for k, bf in enumerate(mbf.components):
        mask = (labels == k)
        x_k = x_train[mask]
        if len(x_k) < 10:
            continue
        dataset_k = TensorDataset(x_k)
        loader_k = DataLoader(dataset_k, batch_size=min(batch_size, len(x_k)), shuffle=True)
        iter_k = iter(loader_k)
        optimizer_k = torch.optim.Adam(bf.parameters(), lr=warmup_lr)
        for step in range(n_warmup_steps):
            try:
                (batch_k,) = next(iter_k)
            except StopIteration:
                iter_k = iter(loader_k)
                (batch_k,) = next(iter_k)
            _, log_det = bf.train_forward(batch_k)
            (-log_det).backward()
            optimizer_k.step()
            optimizer_k.zero_grad()
        print(f"  Component {k}: warm-start done")
    
    print("[TAPC] Pre-initialization complete.")
    return labels


def _remap_clusters_to_components(labels, x_np, K, noise_mask):
    """
    Remap variable number of clusters to exactly K components.
    Handles: noise points, n_found > K (merge small), n_found < K (split large).
    """
    # Assign noise points to nearest cluster centroid
    valid_labels = labels[~noise_mask]
    unique_labels = [l for l in set(valid_labels) if l >= 0]
    n_found = len(unique_labels)
    
    if noise_mask.any():
        centroids = {l: x_np[labels == l].mean(axis=0) for l in unique_labels}
        for i in np.where(noise_mask)[0]:
            dists = {l: np.linalg.norm(x_np[i] - c) for l, c in centroids.items()}
            labels[i] = min(dists, key=dists.get)
    
    # Normalize label indices to 0..n_found-1
    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
    labels = np.array([label_map.get(l, 0) for l in labels])
    
    if n_found == K:
        return labels
    
    elif n_found > K:
        # Merge smallest clusters into nearest larger cluster
        from sklearn.cluster import KMeans
        while len(set(labels)) > K:
            unique, counts = np.unique(labels, return_counts=True)
            smallest_label = unique[counts.argmin()]
            other_labels = unique[unique != smallest_label]
            centroids = {l: x_np[labels == l].mean(axis=0) for l in other_labels}
            centroid_small = x_np[labels == smallest_label].mean(axis=0)
            nearest = min(centroids, key=lambda l: np.linalg.norm(centroids[l] - centroid_small))
            labels[labels == smallest_label] = nearest
            # Renormalize
            unique_new = sorted(set(labels))
            remap = {old: new for new, old in enumerate(unique_new)}
            labels = np.array([remap[l] for l in labels])
        return labels
    
    else:  # n_found < K
        # Split largest clusters using KMeans sub-clustering
        from sklearn.cluster import KMeans
        while len(set(labels)) < K:
            unique, counts = np.unique(labels, return_counts=True)
            largest_label = unique[counts.argmax()]
            mask = (labels == largest_label)
            x_split = x_np[mask]
            km = KMeans(n_clusters=2, n_init=5, random_state=42)
            sub_labels = km.fit_predict(x_split)
            new_label = labels.max() + 1
            indices = np.where(mask)[0]
            for idx, sl in zip(indices, sub_labels):
                if sl == 1:
                    labels[idx] = new_label
            # Renormalize
            unique_new = sorted(set(labels))
            remap = {old: new for new, old in enumerate(unique_new)}
            labels = np.array([remap[l] for l in labels])
        return labels
```

### 安装依赖

```bash
pip install hdbscan  # 仅 HDBSCAN 方法需要；spectral 和 kmeans 仅需 sklearn
```

### 使用示例

```python
# 针对不同数据集选择最合适的方法
# GAUSSIANS / BLOBS (凸形):
labels = topology_aware_preclustering(mbf, x_train, method='kmeans')

# MOONS / CIRCLES (非凸):
labels = topology_aware_preclustering(mbf, x_train, method='hdbscan')

# SPIRALS (极度非凸):
labels = topology_aware_preclustering(mbf, x_train, method='hdbscan', 
                                      min_cluster_size=50)

# 自动选择 (推荐):
labels = topology_aware_preclustering(mbf, x_train, method='auto')
```

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **HDBSCAN 找到 n_found ≠ K** | HDBSCAN 自动 K 可能与 n_components 不同，需要 merge/split 步骤 | 已在 `_remap_clusters_to_components` 中处理；检查日志中的 cluster 大小 |
| **HDBSCAN 噪声点过多** | 若 `min_cluster_size` 太大，大量点被标记为噪声 | 减小 `min_cluster_size`；噪声点会被最近邻分配 |
| **Spectral Clustering 计算开销** | 对大数据集（N > 10000），构建相似度矩阵代价高 | 用 `SpectralClustering(n_components=K, n_init=10, n_jobs=-1)` 并行化；或对大数据集用 mini-batch variant |
| **HDBSCAN 依赖安装** | `hdbscan` 不是项目 requirements 中的包 | 已在代码中添加 ImportError 处理，自动 fallback 到 spectral；建议在 requirements.txt 中加入 `hdbscan` |
| **非凸 cluster 的 warm-start 过拟合** | HDBSCAN 正确识别的非凸 cluster（如月牙）在 warm-start 时，BreezeForest 可能拟合形状而非 CDF | 这是期望行为；BreezeForest 的 CDF 会学习月牙形状的条件分布，并不会过拟合（因为 CDF 是全局单调的） |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（K-Means Pre-Init 的必要升级，对非凸数据集来说是 blocking issue）**

理由：
1. **BreezeForest 的测试数据集大量包含非凸 cluster**（MOONS、CIRCLES、SPIRALS），K-Means 在这些数据集上会产生错误的初始化 → warm-start 失效 → DAEM 需要从错误起点退火，效果不稳定
2. **HDBSCAN 的 "auto K" 能力解决了另一个已知问题**：用户不知道 cluster 数量时，HDBSCAN 可以自动发现并据此调整 n_components
3. **对 GAUSSIANS/BLOBS 数据集没有损失**：代码中保留了 K-Means 作为 fallback，对凸形 cluster 效果与原方案相同
4. **不改变模型架构**：纯粹是初始化策略层面的改进
5. **有丰富文献支撑**：HDBSCAN（McInnes 2017）是非凸聚类的 SOTA，Spectral Clustering（Von Luxburg 2007）是经典非凸聚类算法

---

## 参考文献

- McInnes, L. et al. (2017). "hdbscan: Hierarchical density based clustering." *Journal of Open Source Software 2(11)*. https://doi.org/10.21105/joss.00205  
  ← HDBSCAN 的原始论文
- Von Luxburg, U. (2007). "A Tutorial on Spectral Clustering." *Statistics and Computing 17(4)*. https://link.springer.com/article/10.1007/s11222-007-9033-z  
  ← Spectral Clustering 经典教程
- Bevins, H.T. et al. (2023). "Piecewise Normalizing Flows." *arxiv 2305.02930*.  
  ← 测试了 K-Means vs Mean Shift vs BIRCH，但未测试 HDBSCAN；本 Idea 填补这个空白
- McInnes, L. et al. (2018). "UMAP: Uniform Manifold Approximation and Projection." *arxiv 1802.03426*.  
  ← 数据流形学习背景，说明非凸 cluster 的数学结构
