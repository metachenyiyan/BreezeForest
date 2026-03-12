# Idea: LGMM — Latent Gaussian Mixture Calibration（替代盒形 LZR 的推理时基分布修正）

**创建时间**: 2026-03-11 23:05 UTC  
**推荐优先级**: ⭐⭐ 高优先级（推理阶段独立可部署，无需重训练）

---

## 问题定义

MultiBF 在生成阶段，对每个组件 k 采样：
```
z ~ Uniform(0.01, 0.99)^d
x = f_k^{-1}(z)
```

这里的 `Uniform(0.01, 0.99)^d` 是一个**与数据完全无关的固定先验**。BreezeForest 训练的目标是将数据 X 映射到 Uniform，但真实数据在 latent 空间中并不均匀分布——多 cluster 数据的 latent 表示会聚集在 [0,1]^d 内的特定子区域，剩余区域对应低密度空间或 inter-cluster 区域。

从全 Uniform 采样 z 时，一部分 z 会落在低密度 inter-cluster 区域，反演后产生无效的 cluster 间样本。

### 已有 LZR 方案的局限

现有 `idea_latent_zone_restriction_2026-03-11-1235.md`（LZR）通过统计 latent 表示的 **百分位数区间 [a_k, b_k]^d** 来限制采样范围。这种方法存在三个结构性弱点：

1. **盒形区域假设（Box-shaped zone）**：对 [a_k^1, b_k^1] × ... × [a_k^d, b_k^d] 独立采样，忽略了 latent 维度间的相关性。实际上 latent 表示的 cluster 通常不是轴对齐的矩形框。
2. **均匀采样忽略密度变化**：即使在 Z_k 内，密度也是非均匀的（cluster 中心密度高，边缘密度低）。盒形区域的均匀采样仍然会产生部分低密度样本。
3. **软 EM 导致 Z_k 估计不准**：如果用 soft-EM 训练的 MultiBF，每个组件的 latent 表示中包含来自多个 cluster 的样本（混淆），盒形边界包含了大量非目标 cluster 的 latent 值。

**本 Idea 提出用 Gaussian Mixture Model（GMM）替代盒形区域作为 latent 采样分布**，直接修复以上三个问题。

---

## 从当前项目代码与已有 idea 中得到的背景判断

**代码层面**：
- `BreezeForest.forward(x, breeze_list)` 返回 latent 表示 z = BF(x)（形状为 (N, dim)）
- `BreezeForest.inverse_map(z)` 接受任意 z 做反演，当前 z 来自 `torch.rand(n_k, self.dim) * 0.98 + 0.01`
- `MultiBF.calibrate_latent_zones()`（LZR 建议添加）只计算 percentile bounds，丢失了分布形状信息
- `model/tools.py` 的 `bisection()` 函数使用 `normal.Normal` 作为初始搜索范围的 reference distribution —— 这说明非均匀 z 分布在技术上是完全支持的

**已有 idea 层面**：
- LZR (1235) 是盒形区域方案，本方案是其**理论上更严格的替代**
- Hard-EM 和 PNF-style 改进了训练阶段，本方案改进推理阶段，两者正交，可叠加
- ICDR (1240) 试图在训练中推动组件分离，本方案在推理中直接用数据驱动的分布来避免低密度区域，效果更直接

**外部研究验证**：
- Stimper et al. (2022) AISTATS：用 learned rejection sampling 作为 base distribution 直接改善多模态 flow 的生成质量
- 2023-2024 GMM 作为 flow base distribution 的多篇论文（arxiv 2504.05304, openreview 等）：GMM 初始化/prior 对 multi-modal 密度估计有显著改善
- Stick-Breaking Mixture Flows (ICLR 2025)：混合基分布在多模态后验估计中的有效性
- LZR 方案本身的设计来源（Stimper 2022）比盒形更严格——**latent 空间中用 GMM 拟合本质上是 Stimper 方法的无需重训练近似版**

---

## 核心思路

**训练后校准（Post-Training Calibration）**，无需修改训练：

### 步骤 1：获取 latent 表示
- 对训练数据做 responsibility 加权采样，提取"属于"组件 k 的样本子集
- 通过 `bf.forward(x_k, [])` 获得 latent 表示 z_k = BF_k(x_k)

### 步骤 2：在 logit(z) 空间拟合 GMM
- BreezeForest 的输出在 [0,1]^d（Sigmoid 激活），直接对 z 用 GMM 不合适（有界区域）
- 在 logit 空间：`w = logit(z) = log(z/(1-z))`，这是无界的，GMM 更自然
- 用 sklearn 的 `GaussianMixture` 对 w_k 拟合（建议 1-3 个分量，因为一个 cluster 的 latent 表示通常是单峰的）

### 步骤 3：生成时从 GMM 采样
- 替代 `z = torch.rand(n_k, dim) * 0.98 + 0.01`
- 改为：从 GMM 采样 w，再 `z = sigmoid(w)`，并 clamp 到 [0.01, 0.99]
- 用 `bf.inverse_map(z)` 生成 x

**核心优势**：GMM 采样直接遵循了 cluster k 数据在 latent 空间的实际分布，从高密度区域采样的概率高，从低密度（inter-cluster）区域采样的概率低。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**理论推导**：

设 z_k = {BF_k(x_i) : x_i ∈ cluster k} 是 cluster k 在 latent 空间的表示集合。由于 BF_k 是 cluster k 数据到 [0,1]^d 的 CDF 映射，z_k 在 latent 空间的分布 **近似于 Uniform[0,1]^d 的高密度子区域**。

更重要的是：cluster j（j≠k）的样本 x_j 在 BF_k 的 latent 空间中映射到 {z_k : z_k = BF_k(x_j)}，这些值倾向于聚集在 z_k 的补集区域（因为 BF_k 没有为 cluster j 设计 CDF 梯度）。

通过对 z_k 拟合 GMM，我们：
1. 直接捕获了 cluster k 数据在 BF_k latent 空间中的真实分布
2. 从这个 GMM 采样，生成的 z 天然落在 cluster k 的 latent 高密度区域
3. BF_k^{-1}(z) 将这些 z 映射回 cluster k 的数据空间 → 生成有效样本

**与 LZR 的比较**：

| 方面 | LZR（盒形）| LGMM（本方案）|
|------|-----------|--------------|
| 区域形状 | 轴对齐矩形框 | 椭圆形 GMM（捕获协方差） |
| 维度相关性 | 忽略 | 完整建模 |
| 采样密度 | 均匀（无密度权重） | 按 GMM 密度加权（高密度区域采样多） |
| 低密度区域采样概率 | 有（盒内均匀） | 极低（GMM 尾部概率小） |
| 实现复杂度 | 低（百分位数计算） | 中（GMM 拟合，但 sklearn 一行代码） |
| 对软 EM 模型的鲁棒性 | 差（z_k 包含多 cluster，盒形失真） | 好（GMM 的单峰成分会过滤掉多 cluster 混淆） |

---

## 与历史 idea 的关系

**关系类型：替代 + 升级（LZR Idea 2）**

| 方面 | LZR (1235) | LGMM (本方案 2305) |
|------|-----------|------------------|
| 核心机制 | percentile box | Gaussian Mixture in logit space |
| 维度相关性 | 无 | 有（GMM 协方差矩阵） |
| 密度加权 | 无（均匀） | 有（GMM 密度比例采样） |
| 软 EM 鲁棒性 | 差 | 好（GMM 会自然排除离群 latent 值） |
| 理论依据 | Stimper (2022) 的简化 | 更接近 Stimper (2022) 的精神 |
| 适用场景 | 快速 demo（无 sklearn GMM 拟合） | 正式部署（GMM 更准确） |

**与 PNF-Style (2300) 和温度退火 EM (2255) 的关系**：
- 三者互补：训练阶段 (2255/2300) 改善组件专一性，推理阶段 (本方案) 进一步限制 latent 采样区域
- 组合效果：专一化训练 + LGMM 采样 = 最强的 inter-cluster 生成防护

**与 ICDR (1240) 的关系**：
- ICDR 需要在训练中增加额外计算（cross-component forward passes），实现复杂
- 本方案在推理时实现类似效果（通过不采样 inter-cluster latent 区域），且无额外训练成本
- 建议：如果已有本方案 + 训练时改进，ICDR 的优先级可以降低

---

## 具体实现建议

### 步骤 1：添加 calibrate_latent_gmm() 到 MultiBF

```python
from sklearn.mixture import GaussianMixture
import numpy as np

def calibrate_latent_gmm(self, x_train, n_components_per_cluster=1, resp_threshold=None):
    """
    Fit a GMM in logit-space latent representations for each mixture component.
    
    :param x_train: training data (N, dim), already normalized
    :param n_components_per_cluster: GMM components per BF (1-3, typically 1)
    :param resp_threshold: min responsibility to include a sample (default: 1/K)
    :return: None (stores GMMs in self.latent_gmms)
    """
    if resp_threshold is None:
        resp_threshold = 1.0 / self.n_components
    
    self.latent_gmms = []
    
    with torch.no_grad():
        # Compute responsibilities
        log_pi = self.get_mixture_log_weights()
        component_log_probs = []
        for k, bf in enumerate(self.components):
            ld = self._per_sample_log_det(bf, x_train)
            component_log_probs.append(log_pi[k] + ld)
        
        stacked = torch.stack(component_log_probs, dim=0)  # (K, N)
        log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
        responsibilities = torch.exp(log_resp)  # (K, N)
        
        for k, bf in enumerate(self.components):
            resp_k = responsibilities[k]
            mask = resp_k > resp_threshold
            if mask.sum() < max(10, n_components_per_cluster * 3):
                topk = max(20, int(0.2 * len(resp_k)))
                _, idx = torch.topk(resp_k, topk)
                mask = torch.zeros_like(resp_k, dtype=torch.bool)
                mask[idx] = True
            
            x_k = x_train[mask]
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list)  # (n_k, dim) in [0, 1]^d
            
            # Transform to logit space for unconstrained GMM
            z_np = z_k.cpu().numpy()
            # Safe logit: clamp away from 0 and 1
            z_np = np.clip(z_np, 1e-4, 1 - 1e-4)
            w_np = np.log(z_np / (1 - z_np))  # logit: unbounded
            
            # Fit GMM in logit space
            gmm = GaussianMixture(
                n_components=n_components_per_cluster,
                covariance_type='full',
                n_init=3,
                random_state=42
            )
            gmm.fit(w_np)
            self.latent_gmms.append(gmm)
    
    print(f"Calibrated latent GMMs for {len(self.latent_gmms)} components.")
```

### 步骤 2：修改 inverse_map 使用 GMM 采样

```python
def inverse_map_with_gmm(self, n_samples, max_gap=1e-3, decay_ratio=1.0):
    """
    Generate samples using per-component latent GMM sampling.
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
        
        # Sample from GMM in logit space
        w_samples, _ = self.latent_gmms[k].sample(n_k)  # (n_k, dim) numpy
        # Map back to [0, 1]: sigmoid
        z_np = 1.0 / (1.0 + np.exp(-w_samples))
        # Clamp to valid range
        z_np = np.clip(z_np, 0.01, 0.99)
        z = torch.tensor(z_np, dtype=torch.float)
        
        x_k = self.components[k].inverse_map(
            z, max_gap=max_gap, decay_ratio=decay_ratio
        )
        results[mask] = x_k
    
    return results
```

### 步骤 3：在 demo_multi_bf.py 中集成

```python
# 训练完成后（无需重训练）：
all_batch = get_full_batch(distribution, data_size=3000, mean=mean, std=std)
with torch.no_grad():
    mbf.calibrate_latent_gmm(all_batch, n_components_per_cluster=1)

# 生成时使用 GMM 采样
with torch.no_grad():
    samples = mbf.inverse_map_with_gmm(n_samples=data_size)
    samples = samples * std + mean
```

### 超参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `n_components_per_cluster` | 1 | 单个 cluster 的 latent 表示通常是单峰的；若 cluster 本身有子结构可尝试 2-3 |
| `resp_threshold` | 1/K | 排除被多组件共享的样本，使 GMM 更纯净 |
| `covariance_type` | 'full' | 捕获完整协方差；数据量少时可改为 'diag' |
| logit clamp | [1e-4, 1-1e-4] | 避免 logit 无穷大；z 值接近 0 或 1 的样本通常是 outlier |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **GMM 过拟合** | 如果 n_k 很小，GMM 拟合不稳定 | 确保 n_k ≥ 20；用 `n_components_per_cluster=1` |
| **logit 空间假设** | GMM 在 logit 空间做无界假设，若 z 极端（接近 0/1）可能有数值问题 | clamp z 到 [1e-4, 1-1e-4] 再做 logit |
| **软 EM 导致 GMM 污染** | 软 EM 训练的模型中，z_k 包含多 cluster 的样本，GMM 可能建模出混合分布 | 与 PNF-style 或温度退火 EM 结合使用，或提高 resp_threshold |
| **numpy/sklearn 依赖** | 需要 sklearn.mixture.GaussianMixture | sklearn 已在项目中使用（distribution2d.py），无额外依赖 |
| **GMM 采样在边界区域** | GMM 的高斯尾部可能采样到 logit 空间极端值，clamp 后集中在边界 | 在 GMM 上做 rejection sampling：拒绝 \|w\| > 3σ 的样本 |

---

## 推荐优先级

**⭐⭐ 高优先级（推理阶段的最优改进，无需重训练）**

理由：
1. **比 LZR 明显更强**：GMM 比盒形区域更准确地捕获 cluster 的 latent 形状和密度
2. **无需重训练**：在任何已训练的 MultiBF 模型上均可即时部署
3. **实现成本低**：核心代码约 30 行，sklearn GMM 拟合极快（< 1 秒）
4. **理论基础更扎实**：本质上是 Stimper (2022) learned base distribution 的无梯度近似版本
5. **与训练改进正交**：可与 PNF-style (2300)、温度退火 EM (2255) 组合使用，形成最强防护

**建议与训练改进组合**：
- 最优策略 A：PNF-style 独立训练 (2300) + LGMM 采样（本方案）
- 最优策略 B：温度退火 EM (2255) + LGMM 采样（本方案）
- 快速验证（无需重训练）：现有 MultiBF + LGMM 采样

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a/stimper22a.pdf  
  *(核心思路来源：learned base distribution 替代 uniform prior)*
- arxiv 2504.05304 (2025). "Gaussian Mixture Flow Matching Models."  
  *(GMM 作为 flow/diffusion 模型的分布参数的有效性验证)*
- ICLR 2025 Submission. "Stick-Breaking Mixture Flows with Component-Wise Tail Adaptation."  
  *(混合基分布在多模态后验估计中的理论支持)*
- Reynolds, D. (2009). "Gaussian Mixture Models." *Encyclopedia of Biometrics*.  
  *(GMM 经典参考)*
- Pires, G. & Figueiredo, M. (2020). "Variational Mixture of Normalizing Flows." *ESANN 2020*.  
  *(mixture of flows 的 latent space 分割思想)*
