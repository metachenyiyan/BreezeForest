# Idea: GMM 拟合隐变量基础分布（升级 LZR）

**创建时间**: 2026-03-11 13:41 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（对 Idea 2: LZR 的原则性升级）

---

## 问题定义

`notes/idea/idea_latent_zone_restriction_2026-03-11-1235.md`（下称"Idea 2 / LZR"）已正确识别了 MultiBF 生成阶段的核心问题：每个组件 f_k 的逆映射 f_k^{-1} 将整个 z ∈ [0.01, 0.99]^d 空间都映射回数据空间，而训练数据 cluster k 实际只对应 [0.01, 0.99]^d 内的某个**子区域**。从整个 [0.01, 0.99]^d 采样会触发映射到非 cluster k 区域甚至 cluster 之间。

LZR 的修复思路是"为每个组件估计轴对齐的边界框（percentile box）"，但这个方案存在以下根本性局限：

1. **忽略维度间相关性**：z ∈ [0.01, 0.99]^d 内 cluster k 的分布不是轴对齐矩形，而是有相关结构的曲面。用矩形边界框采样会在角落（box 内但非真实分布范围内）采出映射到 inter-cluster 区域的 z。
2. **过于保守或过于激进**：矩形框要么太紧（截断合法样本）要么太宽（包含 inter-cluster z 区域）。
3. **多 cluster 映射到同一组件时失效**：如果一个组件负责多个 cluster（soft-EM 训练的常见情况），其 z 分布是多峰的，单个矩形框无法准确描述。

**根本解决方案**：用**高斯混合模型（GMM）**拟合每个组件在隐变量空间内训练数据的实际分布，在生成时从这个学到的分布采样，而非从均匀矩形采样。这是 Stimper et al. (2022) 的"Resampled Base Distribution"思路的简化数据驱动版本。

---

## 从当前项目代码与已有 Idea 中得到的背景判断

**代码结构关键点**：

1. **BreezeForest 的 forward 输出空间**（`model/BreezeForest.py`）：  
   TreeLayer 的激活函数是 `Sigmoid`（`model/tools.py:class Sigmoid`），其输出范围为 (0, 1)。因此 `bf.forward(x)` 输出 z ∈ (0, 1)^d，与 [0.01, 0.99]^d 采样范围对应。z 是真实的 CDF 值，其分布反映了数据的真实 CDF 结构。

2. **`MultiBF.inverse_map`**（`model/MultiBF.py:140-171`）：
   ```python
   z = torch.rand(n_k, self.dim) * 0.98 + 0.01  # 均匀采样
   x_k = self.components[k].inverse_map(z, ...)
   ```
   **这是修改的精确位置**：只需将 `torch.rand(...) * 0.98 + 0.01` 替换为从 GMM 采样，并 clamp 到 [0.01, 0.99] 即可。

3. **LZR（Idea 2）的实现方案**已提供了 `calibrate_latent_zones()` 和 `inverse_map_with_zones()`，本 Idea 可以用同样的框架，替换 zone 采样方式为 GMM 采样。

4. **sklearn 已是依赖**（`distribution2d.py` 使用了 `sklearn.datasets`），可以直接用 `sklearn.mixture.GaussianMixture`。

**外部调研关键发现**：

- **Stimper et al. (2022) "Resampling Base Distributions of Normalizing Flows"（AISTATS 2022）**：  
  使用学习的 rejection sampling 来修正 base distribution，直接解决了 flow 的 topology 问题。本 Idea 是其数据驱动的轻量替代版：不需要学习额外的模型，直接从训练数据的 latent 表示中拟合 GMM。

- **与矩形 LZR 的数学差异**：  
  设 z_k = {f_k(x_i) : x_i ∈ cluster k 的样本}，则：  
  - LZR 用 [q_{p,d}^lo, q_{p,d}^hi] 矩形逼近 z_k 的分布支撑  
  - GMM 拟合 z_k 的实际概率分布 p_k^*(z)，在 z ∈ z_k 时概率高，在 z ∉ z_k 时概率低  
  GMM 采样的样本按 p_k^*(z) 加权，自然避开 z_k 的低密度区域（即 inter-cluster 对应的 z）。

---

## 核心思路

**训练后校准（与 LZR 相同框架，但用 GMM 替换矩形）**：

1. 对训练数据，计算每个样本在各组件下的 responsibility（同 LZR）
2. 对组件 k 被高度 responsible 的样本，正向传播得到其 latent 表示 z_i^k = f_k(x_i)
3. 对 {z_i^k} 拟合一个 K_z 分量的 GMM（K_z 通常为 1-3，不需要太复杂）
4. 生成时：从 GMM 采样 z，clamp 到 [0.01, 0.99]^d，再用 f_k^{-1}(z) 反演

**关键公式**：

```
传统生成：z ~ Uniform([0.01, 0.99]^d) ← 均匀，不区分 cluster
LZR：     z ~ Uniform([a_k, b_k]^d)   ← 矩形框，忽略相关性
GMM 基分布：z ~ GMM_k(μ, Σ) ∩ [0.01, 0.99]^d ← 拟合实际分布，尊重相关结构
```

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**数学论证**：

设训练数据 cluster k 的样本为 {x_i^k}，其 latent 表示为 {z_i^k = f_k(x_i^k)}。

- 由 f_k 的双射性：**f_k^{-1}({z_i^k}) = {x_i^k}**（cluster k 的数据点）
- inter-cluster 的点 x* 有 z* = f_k(x*) **不在** {z_i^k} 的高密度区域
- 从 GMM_k 采样 z：高概率落在 {z_i^k} 的密集区域 → f_k^{-1}(z) 高概率接近 {x_i^k}（cluster k）
- 从 GMM_k 采样 z：低概率落在 {z_i^k} 的低密度区域（即 inter-cluster 对应的 z）

**与 LZR 的优势对比**：

| 场景 | LZR（矩形框）| GMM 基分布 |
|------|------------|-----------|
| z_k 是椭圆形分布 | 矩形框包含四个"角落"（低密度 z），这些 z 可能映射到 inter-cluster 区域 | GMM 自然跳过低密度角落 |
| z_k 是 L 形分布 | 矩形框严重高估有效区域 | GMM 用 2 个分量准确描述 |
| 一个组件对应 2 个 cluster | 单矩形框：中间区域（两 cluster 之间）被包含在内 | GMM 的 2 个分量分别对应两个 cluster，中间区域概率低 |
| 高维（dim > 5）| "角落灾难"：矩形框的大部分体积在角落 | GMM 体积与数据分布对齐 |

**直觉解释**：

LZR 是在 z 空间画一个矩形，然后从矩形内均匀采样。GMM 是在 z 空间学习实际数据的密度图，然后按密度采样。前者会从低密度的"荒地"采样（这些荒地对应 inter-cluster 区域），后者只从"有数据的地方"采样。

---

## 与历史 idea 的关系

| 历史 Idea | 关系 |
|----------|------|
| **Idea 2（LZR，12:35）** | **直接升级，替代矩形框方案**。LZR 的整体框架（post-training calibration、`calibrate_latent_zones`、`inverse_map_with_zones`）完全继承，只需将矩形边界采样替换为 GMM 采样。若已实现 LZR，迁移到本 Idea 约需 10 行修改。|
| **Idea 1（Hard-EM，12:30）** | **互补，但本 Idea 对 Idea 1 的依赖更小**。即使组件没有经过 Hard-EM 专一化，GMM 拟合的是已训练模型下"高 responsibility"样本的 latent 分布，可以适应多 cluster/组件的情况（GMM 自动用多个分量对应多个 cluster）。|
| **Idea 3（ICDR，12:40）** | 前置增益：ICDR 使各组件密度区域分离，GMM 拟合的 z_k 分布会更紧凑（variance 更小），生成质量进一步提升。|
| **Stimper et al. (2022) Resampled Base Distributions** | 本 Idea 是 Stimper 方法的**简化数据驱动版**。Stimper 方法需要训练一个额外的学习模型，本 Idea 只需拟合一个 GMM（非学习，直接用 sklearn，无梯度）。适配 BreezeForest，成本更低。|

---

## 具体实现建议

### 步骤 1：添加 `calibrate_gmm_base()` 方法到 MultiBF

```python
from sklearn.mixture import GaussianMixture
import numpy as np

def calibrate_gmm_base(self, x_train, n_latent_components=2, 
                        resp_threshold=None, top_k_percent=0.5):
    """
    为每个 MultiBF 组件在 latent z 空间拟合 GMM，用于精确基础分布采样。
    
    :param x_train: 训练数据 tensor (N, dim)，归一化后
    :param n_latent_components: 每个组件的 latent GMM 分量数（1-3）
    :param resp_threshold: responsibility 阈值（默认 1/K）
    :param top_k_percent: 若阈值样本不足，取 top k% 样本
    """
    self.latent_gmms = []
    if resp_threshold is None:
        resp_threshold = 1.0 / self.n_components

    with torch.no_grad():
        # 计算 responsibility
        log_pi = self.get_mixture_log_weights()
        comp_log_probs = []
        for k, bf in enumerate(self.components):
            ld = self._per_sample_log_det(bf, x_train)
            comp_log_probs.append(log_pi[k] + ld)
        
        stacked = torch.stack(comp_log_probs, dim=0)  # (K, N)
        log_resp = stacked - torch.logsumexp(stacked, dim=0, keepdim=True)
        resp = torch.exp(log_resp)  # (K, N)
        
        for k, bf in enumerate(self.components):
            resp_k = resp[k].cpu()
            
            # 选取高 responsibility 的样本
            mask = resp_k > resp_threshold
            if mask.sum() < max(20, int(0.05 * len(resp_k))):
                # 后备：取 top_k_percent 样本
                n_top = max(20, int(top_k_percent * len(resp_k)))
                _, idx = torch.topk(resp_k, n_top)
                mask = torch.zeros_like(resp_k, dtype=torch.bool)
                mask[idx] = True
            
            x_k = x_train[mask]
            
            # 正向传播到 latent 空间
            breeze_list = []
            z_k = bf.forward(x_k.to(next(bf.parameters()).device), 
                             breeze_list)  # (n_k, dim)
            z_k_np = z_k.cpu().numpy()
            
            # 拟合 GMM
            n_components_gmm = min(n_latent_components, len(z_k_np) // 10)
            n_components_gmm = max(1, n_components_gmm)
            
            gmm = GaussianMixture(
                n_components=n_components_gmm,
                covariance_type='full',  # 捕捉维度间相关性
                n_init=5,
                random_state=42
            )
            gmm.fit(z_k_np)
            self.latent_gmms.append(gmm)
            
            print(f"组件 {k}: 使用 {mask.sum().item()} 个高 resp 样本，"
                  f"GMM log-likelihood: {gmm.score(z_k_np):.3f}")

def sample_from_gmm_clamped(gmm, n_samples, dim, lo=0.01, hi=0.99, 
                              max_trials=5):
    """从 GMM 采样，并 clamp 到 [lo, hi]^dim，拒绝越界样本（简单版本）"""
    all_samples = []
    remaining = n_samples
    for _ in range(max_trials):
        z_raw, _ = gmm.sample(remaining * 2)  # 多采一倍，过滤越界
        z_tensor = torch.tensor(z_raw, dtype=torch.float32)
        valid = ((z_tensor >= lo) & (z_tensor <= hi)).all(dim=1)
        z_valid = z_tensor[valid]
        if len(z_valid) > 0:
            all_samples.append(z_valid[:remaining])
            remaining -= len(z_valid[:remaining])
        if remaining <= 0:
            break
    
    if remaining > 0:
        # 后备：用 clamp 强制合法
        z_raw, _ = gmm.sample(remaining)
        z_fallback = torch.tensor(z_raw, dtype=torch.float32).clamp(lo, hi)
        all_samples.append(z_fallback)
    
    return torch.cat(all_samples, dim=0)[:n_samples]
```

### 步骤 2：修改 `inverse_map()` 使用 GMM 基分布

```python
def inverse_map_with_gmm_base(self, n_samples, max_gap=1e-3, 
                               decay_ratio=1.0):
    """
    使用 GMM 基分布生成样本，替代 Uniform([0.01, 0.99]^dim)。
    需先调用 calibrate_gmm_base()。
    """
    assert hasattr(self, 'latent_gmms'), "先调用 calibrate_gmm_base()"
    
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)
    
    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        # 从组件 k 的 GMM 基分布采样
        z = sample_from_gmm_clamped(
            self.latent_gmms[k], n_k, self.dim
        )
        
        x_k = self.components[k].inverse_map(
            z, max_gap=max_gap, decay_ratio=decay_ratio
        )
        results[mask] = x_k
    
    return results
```

### 步骤 3：在训练完成后调用（demo 文件修改）

```python
# 训练完成后：
print("正在校准 GMM 基础分布...")
all_data = next(iter(DataLoader(distribution, batch_size=3000, shuffle=True)))[0]
all_data_norm = (all_data - mean) / std

with torch.no_grad():
    mbf.calibrate_gmm_base(
        all_data_norm,
        n_latent_components=2,  # 每个组件的 latent GMM 分量数
    )

# 使用 GMM 基分布生成
with torch.no_grad():
    samples = mbf.inverse_map_with_gmm_base(n_samples=data_size)
    samples = samples * std + mean
```

### 超参数调优建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `n_latent_components` | 1（单 cluster/组件）或 2（soft-EM 组件覆盖多 cluster）| 用 BIC 自动选择效果最好（`gmm.bic(z_k_np)`）|
| `resp_threshold` | 1/K（均匀阈值）| 若 soft-EM，可调高到 0.5 以选更"纯净"的样本 |
| `covariance_type` | `'full'` | 捕捉维度间相关性；维度高时用 `'diag'` 节省计算 |
| `rejection 策略` | 先拒绝后 clamp | 先尝试拒绝法（保真度高），不足时用 clamp 后备 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **GMM 拟合不准（样本少）** | 某个组件的 high-resp 样本极少，GMM 拟合不稳定 | 放宽 `top_k_percent` 到 0.7；或用 `n_latent_components=1` |
| **拒绝率过高** | GMM 分布在 [0.01, 0.99]^d 范围外的部分太多 | 用截断正态分布代替 GMM（scipy.stats.truncnorm）；或增大 `max_trials` |
| **组件不专一时 GMM 多峰** | soft-EM 组件的 z_k 分布多峰，GMM 需多分量 | 用 BIC 自动选 n_components；或先用 Hard-EM/K-Means init 使组件专一化 |
| **维度高时 full covariance 不稳定** | dim > 10 时 full 协方差矩阵可能奇异 | 改用 `covariance_type='diag'` 或 `'tied'` |
| **不参与反向传播** | GMM 是后处理，无法用于端到端梯度优化 | 这是设计选择；若需端到端，可改用可学习的 normalizing flow 作为基分布（Stimper et al. 2022）|

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（对 LZR 的原则性升级）**

理由：
1. **LZR 的直接升级**：在 LZR 框架内替换采样方式，约 30 行新代码，代价极低
2. **理论更加严格**：GMM 采样按照 latent 空间的真实数据密度采样，而非均匀矩形采样
3. **无需重训练**：与 LZR 一样，是推断阶段的 post-training 修改，可立即验证
4. **处理 soft-EM 组件**：即使组件没有完全专一化（包含多个 cluster），GMM 的多分量结构可以自然适应
5. **理论来源**：与 Stimper et al. (2022) AISTATS 同一理论根源，有成熟文献支持
6. **与其他 Idea 正交**：可与 Hard-EM、K-Means init、ICDR 任意组合，且每种组合都比单独使用 LZR 更好

**与 LZR（Idea 2）的选择建议**：
- **即时验证场景**：先试 LZR（矩形框），成本更低，如果效果不够好再升级到 GMM
- **正式实验场景**：直接使用 GMM，跳过 LZR，理论更严格，代码量差异不大

---

## 参考文献

- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a.html  
  (学习 rejection sampling base distribution 解决 topology 问题，本 Idea 的理论来源)
- Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*, Chapter 9: Mixture Models and EM.  
  (GMM 拟合理论基础)
- Rezende, D. & Mohamed, S. (2015). "Variational Inference with Normalizing Flows." *ICML 2015*.  
  (Normalizing flows base distribution 的重要性分析)
- sklearn.mixture.GaussianMixture documentation: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 2024*. https://arxiv.org/abs/2305.12149  
  (同类 latent-space 采样改进方案)
