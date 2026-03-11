# Idea: 训练数据密度引导的推断时 KDE 拒绝采样

**创建时间**: 2026-03-11 21:55 UTC  
**推荐优先级**: ⭐⭐ 高优先级（快速验证首选，适用于单 BF 和 MultiBF）

---

## 问题定义

BreezeForest 在生成阶段产生 cluster 间无效点，根本原因是：**模型在 cluster 间的低密度区域仍然具有非零（甚至不可忽视的）概率**。

这个现象可以从两个角度理解：

**角度 1：模型视角**  
训练目标最大化 log |det J_f(x)|（Jacobian 行列式的对数）。由于训练数据不覆盖 cluster 间区域，模型对这些区域的 Jacobian 没有强烈的惩罚（不像 cluster 中心附近有明显的梯度信号）。结果是：inter-cluster 区域的模型密度虽低，但非零。

**角度 2：采样视角**  
从 Uniform([0.01, 0.99]^d) 采样 z，再通过 inverse_map 得到 x。latent space 中某些 z 值对应 cluster 间的 x，而这些 z 被均匀采样到，导致生成出无效点。

**本 Idea 的核心**：在生成阶段用训练数据的核密度估计（KDE）作为"真实数据分布"的代理，过滤掉模型生成的低数据密度候选点。

**与现有 idea 的区别**：
- LZR：latent space 范围限制（per-component box），仅 MultiBF
- GMM Base（idea 1）：替换 base distribution，在 latent space 操作
- **本 Idea：在 data space 操作，用训练数据密度过滤候选点**

三者互补，可叠加使用。

---

## 从当前项目代码与已有 idea 中得到的背景判断

**代码观察**：

1. `generate_sample()` 函数中，生成候选 x 后**没有任何质量过滤**。所有 inverse_map 的输出都直接被接受：
   ```python
   generated = model.inverse_map(seeds)  # 无过滤步骤
   ```

2. BreezeForest 的 `train_forward()` 返回 log-det（批量均值），没有提供 per-sample log-likelihood，这使得直接用模型密度过滤变得复杂（需要 per-sample 计算）。

3. 训练数据本身（x_train）是一个高质量的参考：任何"真实"的生成样本都应该和训练数据有相似的局部密度。

**已有 idea 评估**：
- Hard-EM 和 ICDR 都是训练阶段修复，需要重训练。
- LZR 和 GMM Base 是推断阶段修复，在 latent space 操作。
- **本 Idea 提供第三种互补机制：在 data space 的后处理过滤**，是最直观、最容易 debug 的方案。

**主要问题根源总结**：
- 单 BF：bijection 迫使整个 data space 映射到 [0,1]^d，inter-cluster 区域必然有对应的 latent 值
- MultiBF：soft-EM 训练导致组件不专一，每个组件的 inverse_map 产生各 cluster + inter-cluster 点
- 两种情况的共同特征：**生成的 inter-cluster 点在训练数据分布下概率极低（KDE 值接近 0）**

这正是 KDE 过滤的用武之地。

---

## 核心思路

**三步框架：Calibrate → Oversample → Filter**

### Step 1: 校准（一次性离线步骤）

用训练数据 x_train 训练一个轻量 KDE：
```
KDE_train = KernelDensity(bandwidth=h).fit(x_train)
```
再计算训练数据本身的 KDE 分值，确定接受阈值：
```
train_scores = KDE_train.score_samples(x_train)
τ = percentile(train_scores, p=5)  # 5th percentile 作为 acceptance threshold
```

### Step 2: 过量生成（Oversample）

生成 N_oversample = N_target × oversample_factor 个候选样本（直接用标准 Uniform 或 GMM base 采样）：
```
x_candidates = model.inverse_map(z_samples)  # (N_oversample, dim)
```

### Step 3: 过滤（Filter）

计算每个候选点的 KDE 分值，只保留高于阈值 τ 的：
```
candidate_scores = KDE_train.score_samples(x_candidates)
accepted = x_candidates[candidate_scores >= τ]
```

返回前 N_target 个被接受的样本。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**关键观察**：

训练数据仅来自 K 个 cluster。KDE_train 在 cluster 中心附近得分高（>τ），在 cluster 之间的区域得分低（<<τ，趋向 -∞）。

**因此**：
- 来自 cluster 的有效生成点 → KDE_train 得分高 → 被接受 ✓
- 来自 cluster 间的无效生成点 → KDE_train 得分极低 → 被拒绝 ✗

**KDE 过滤 vs 模型密度过滤的优势**：

模型本身（p_flow(x) = |det J_f(x)|）在 inter-cluster 区域的密度是**模型学到的**，可能存在高估（因为训练没有在那些区域给出负信号）。KDE_train 是**数据驱动的**，直接反映训练数据的真实分布，不受模型估计误差影响。

**数学角度（重要性重采样）**：

完整版本是 importance resampling：
1. 生成候选 {x_i} ~ p_flow(x)
2. 计算权重 w(x_i) = KDE_train(x_i) / p_flow(x_i)（目标密度 / 提案密度）
3. 根据 {w(x_i)} 重采样

简化版（本 Idea）省略 1/p_flow(x) 项，直接用 KDE_train(x_i) 作为阈值。当 p_flow 对所有候选点相对均匀时，这是合理近似。

**外部文献验证**：
- **Stimper et al. (AISTATS 2022)**：使用 learned rejection sampling 修复 normalizing flow 的 topological mismatch，思路与本 Idea 一致。本 Idea 是无需额外训练的简化版本。
- **Importance Corrected Neural JKO Sampling (NeurIPS 2024)**：通过重要性权重修正 flow 模型的 distribution mismatch，与本 Idea 的 importance resampling 思路相同。
- **Annealing Flow (arxiv 2409.20547)**：通过 annealing 引导采样避免 inter-cluster 区域，精神上与 KDE 过滤一致（均为引导采样远离低密度区域）。

---

## 它与历史 idea 的关系

**与 LZR（idea_latent_zone_restriction）**：互补，机制不同
- LZR 在 latent space（z space）限制采样范围
- 本 Idea 在 data space（x space）过滤生成结果
- LZR 依赖 latent zone 估计准确性；本 Idea 直接使用训练数据分布
- **建议**：LZR 作为第一道过滤（latent space 范围），本 Idea 作为第二道过滤（data space 密度）

**与 GMM Base（idea_empirical_latent_gmm_base）**：互补，可叠加
- GMM Base 改变采样策略（换 base distribution）
- 本 Idea 过滤采样结果（rejection filter）
- 叠加使用：先用 GMM Base 减少候选 inter-cluster 样本，再用 KDE 过滤剩余少量无效点

**与 Hard-EM / ICDR（训练阶段 idea）**：正交，可叠加
- 本 Idea 完全不改变训练，只是推断阶段的后处理
- 即使 MultiBF 没有用 Hard-EM 训练，本 Idea 也能有效过滤

**无替代关系**：本 Idea 提供的 data space 过滤机制在历史 idea 中从未出现。

---

## 具体实现建议

### 核心实现

```python
import numpy as np
from sklearn.neighbors import KernelDensity
import torch

class GenerationFilter:
    """
    Post-generation filter using KDE of training data.
    Works for both single BreezeForest and MultiBF.
    """
    
    def __init__(self, x_train, bandwidth='scott', percentile_threshold=5.0):
        """
        :param x_train: training data numpy array (N, dim)
        :param bandwidth: KDE bandwidth ('scott', 'silverman', or float)
        :param percentile_threshold: acceptance threshold percentile
        """
        self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        self.kde.fit(x_train)
        
        # Calibrate threshold from training data
        train_scores = self.kde.score_samples(x_train)  # log-density
        self.threshold = np.percentile(train_scores, percentile_threshold)
        print(f"KDE filter calibrated. log-density threshold: {self.threshold:.3f}")
    
    def filter(self, x_candidates):
        """
        :param x_candidates: numpy array (N_gen, dim)
        :return: filtered numpy array (N_accepted, dim)
        """
        scores = self.kde.score_samples(x_candidates)
        mask = scores >= self.threshold
        n_accepted = mask.sum()
        acceptance_rate = n_accepted / len(x_candidates)
        return x_candidates[mask], acceptance_rate
    
    def filter_and_resample(self, x_candidates, n_target):
        """
        Filter candidates and optionally importance-resample to exact n_target.
        :return: n_target filtered samples
        """
        scores = self.kde.score_samples(x_candidates)
        mask = scores >= self.threshold
        accepted = x_candidates[mask]
        
        if len(accepted) >= n_target:
            return accepted[:n_target]
        else:
            # Not enough accepted: resample with replacement from accepted
            # (rare case, usually avoided by oversampling)
            idx = np.random.choice(len(accepted), size=n_target, replace=True)
            return accepted[idx]


def generate_with_kde_filter(model, kde_filter, n_target, oversample_factor=3, max_gap=1e-3):
    """
    Generate n_target clean samples using KDE-based rejection.
    
    :param model: BreezeForest or MultiBF instance (with .inverse_map)
    :param kde_filter: fitted GenerationFilter instance
    :param n_target: number of final samples
    :param oversample_factor: generate this many times more candidates
    :return: filtered samples tensor (n_target, dim)
    """
    n_gen = n_target * oversample_factor
    
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'n_components'):
            # MultiBF
            x_cand = model.inverse_map(n_gen, max_gap=max_gap)
        else:
            # Single BF
            z = torch.rand(n_gen, model.dim) * 0.98 + 0.01
            x_cand = model.inverse_map(z, max_gap=max_gap)
    
    x_cand_np = x_cand.numpy()
    accepted, rate = kde_filter.filter(x_cand_np)
    
    print(f"KDE filter: acceptance rate = {rate:.1%}, accepted {len(accepted)}/{n_gen}")
    
    return torch.tensor(
        kde_filter.filter_and_resample(x_cand_np, n_target), 
        dtype=torch.float32
    )
```

### 集成到 demo_functions.py

```python
def generate_sample_kde_filtered(model, x_train_raw, std, mean, sample_size):
    """
    Generation pipeline with KDE density filtering.
    x_train_raw: raw (un-normalized) training data, numpy array
    """
    model.eval()
    
    # Step 1: Calibrate KDE filter using raw training data
    kde_filter = GenerationFilter(
        x_train=x_train_raw,
        bandwidth='scott',          # Scott's rule for bandwidth
        percentile_threshold=5.0    # reject bottom 5% by training data density
    )
    
    # Step 2: Generate with filtering (oversample by 3x)
    with torch.no_grad():
        z = torch.rand(sample_size * 3, model.dim) * 0.98 + 0.01
        x_cand = model.inverse_map(z)
        x_cand_raw = (x_cand * std + mean).numpy()   # denormalize for KDE comparison
    
    accepted, rate = kde_filter.filter(x_cand_raw)
    generated = torch.tensor(accepted[:sample_size])
    
    return generated
```

### 带宽选择建议

| 数据类型 | 推荐带宽 | 说明 |
|---------|---------|------|
| 标准化 2D 数据（std≈1） | 'scott' ≈ 0.2–0.4 | 自适应，通常合适 |
| 原始数据尺度不一 | 先标准化再拟合 KDE | 避免大尺度维度主导带宽 |
| Cluster 很紧密 | 手动设小带宽（0.1–0.2） | 避免高斯核"渗漏"到 cluster 间 |
| Cluster 较分散 | 手动设大带宽（0.5–1.0） | 防止 cluster 内部被截断 |

### 阈值百分位调优建议

| 百分位阈值 | 效果 | 适用场景 |
|-----------|------|---------|
| 1% | 宽松过滤，保留 99% 训练数据对应区域 | 轻微 inter-cluster 问题 |
| 5% | 推荐默认值 | 中等 inter-cluster 问题 |
| 10% | 严格过滤，可能截断 cluster 边缘 | 严重 inter-cluster 问题，cluster 间距大 |
| 动态阈值 | 根据最近的 training cluster center 调整 | 进阶方案 |

---

## 进阶变体：完整重要性重采样（Importance Resampling）

若需要理论上更严格的校正，可以使用完整重要性重采样：

```python
def importance_resample(model, kde_filter, x_cand):
    """
    Full importance resampling: weight candidates by KDE(x) / p_flow(x).
    Requires per-sample log p_flow computation.
    """
    # Compute log KDE
    log_kde = kde_filter.kde.score_samples(x_cand.numpy())  # (N,)
    
    # Compute per-sample log p_flow via finite difference
    # For single BF: use _per_sample_log_det equivalent
    x_cand_tensor = torch.tensor(x_cand, dtype=torch.float32)
    epsilon = model.epsilon  # (1, dim)
    
    with torch.no_grad():
        breeze_list = []
        y = model.forward(x_cand_tensor, breeze_list)
        x_deltas = torch.cat([
            (x_cand_tensor - epsilon).unsqueeze(0),
            (x_cand_tensor + epsilon).unsqueeze(0)
        ], dim=0)
        y_deltas = model.breeze_forward(x_deltas, breeze_list)
        du_dx = ((y_deltas[1] - y_deltas[0]) / (2 * epsilon)).abs().clamp(min=0.001)
        log_p_flow = torch.sum(torch.log(du_dx), dim=1).numpy()  # per-sample
    
    # Importance weights: w ∝ exp(log_kde - log_p_flow)
    log_weights = log_kde - log_p_flow
    log_weights -= log_weights.max()  # numerical stability
    weights = np.exp(log_weights)
    weights /= weights.sum()
    
    # Resample
    n_target = len(x_cand) // 3
    idx = np.random.choice(len(x_cand), size=n_target, p=weights, replace=False)
    return x_cand[idx]
```

重要性重采样的优势：严格消除 bias（使输出分布趋向 KDE_train）；缺点：需要 per-sample log p_flow，计算量更大。推荐先用简单阈值版本验证效果，再考虑升级到完整重采样。

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **KDE 维度灾难** | 高维数据（>10D）中 KDE 可靠性急剧下降 | BreezeForest 目前演示为 2D，暂无此问题；高维时改用 GMM 代替 KDE |
| **过拒（Over-rejection）** | 阈值过严导致 cluster 边缘也被拒绝，生成样本过于集中在 cluster 中心 | 调低百分位阈值（从 5% 降到 1%）；可视化 cluster 边缘分布 |
| **KDE 带宽敏感** | 带宽太小→ KDE 过于局部（高斯峰窄）→ 大量样本被拒绝；带宽太大→过滤失效 | 使用 'scott' 自适应带宽；对不同 cluster 尺度的数据先标准化 |
| **Oversample 倍率不足** | 若 inter-cluster 生成比例很高，3× oversample 可能不够 | 自适应 oversample：观察 acceptance rate，若 < 30% 则继续生成更多候选 |
| **KDE 与真实密度偏差** | 训练数据有限时，KDE 对 cluster 边缘的估计可能不准 | 增大训练集（已有 3000-5000 样本足够 2D KDE）；调宽 KDE 带宽 |

---

## 推荐优先级

**⭐⭐ 高优先级（最容易实施和调试，适合快速验证）**

理由：
1. **无需改变模型或训练**：纯推断阶段后处理，对已有训练结果直接可用
2. **直观可解释**：逻辑简单——不要生成训练数据密度低的点
3. **互补于现有 idea**：与 LZR（latent 限制）和 GMM Base（改变采样策略）形成三层防御
4. **适用于单 BF**：是当前 idea 中唯一对单 BreezeForest（非 MultiBF）也完全适用的 data space 过滤方案
5. **外部验证**：Stimper 2022（rejection sampling base）和 importance resampling 文献均有理论支撑

**建议使用顺序**：
1. 先用本 Idea（KDE 过滤）快速验证 inter-cluster 问题是否可解，无需任何模型改动
2. 再用 GMM Base 替换 base distribution，减少所需 oversample 倍率
3. 最后用 Hard-EM 重训练，从根本上解决问题，KDE 过滤作为最终保险层

---

## 参考文献

- **Stimper, V. et al. (AISTATS 2022). "Resampling Base Distributions of Normalizing Flows."**  
  使用 learned rejection sampling 作为 normalizing flow base，本 Idea 是其推断侧简化版。

- **Importance Corrected Neural JKO Sampling (NeurIPS 2024, OpenReview).**  
  通过重要性权重修正 flow model 的 distribution mismatch，与本 Idea 的 importance resampling 变体一致。

- **Coeurdoux, F. et al. (Machine Learning 2024). "Normalizing flow sampling with Langevin dynamics in the latent space."**  
  在 latent space 用 MCMC 引导采样远离低密度区域，与本 Idea 思路相似但机制不同（本 Idea 更简单）。

- **Annealing Flow (arxiv 2409.20547, ICML 2025).**  
  通过 annealing 引导 CNF 采样避免 inter-modal 区域，验证了"引导采样远离低密度区域"的一般有效性。
