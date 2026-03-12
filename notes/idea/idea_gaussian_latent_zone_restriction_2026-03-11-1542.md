# Idea: Gaussian Latent Zone Restriction (G-LZR)

**创建时间**: 2026-03-11 15:42 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（生成阶段最强修复方案，可无需重训练）

---

## 问题定义

MultiBF 生成阶段的根本缺陷：采样 z ~ Uniform(0.01, 0.99)^d 并通过 f_k^{-1}(z) 生成，但全 latent cube 中只有一部分 z 值（Z_k）对应 cluster k 的有效数据区域。采样包含 Z_k^c（Z_k 的补集），导致 f_k^{-1}(Z_k^c) 产生其他 cluster 的点或 inter-cluster 的无效点。

历史 Idea 1235（LZR）已提出"限制 z 的采样范围"的核心方向，并给出了**矩形边界框（axis-aligned bounding box）**估计 Z_k 的方法。这个方向是正确的，但矩形边界框有以下已知不足：

1. **忽略维度间相关性**：若 Z_k 在 latent 空间是一个斜向的椭球，矩形框会包含大量 Z_k^c 中的点，仍然产生无效样本
2. **过度保守**：矩形框可能过大（包含其他 cluster 的 z 值）或过小（截断合法 z 值）
3. **多个 cluster 的 latent zone 可能在轴对齐方向重叠但在斜向方向分离**：矩形无法利用这种分离

---

## 从代码与已有 Idea 得到的背景判断

### 代码分析

- `BreezeForest.forward()` 将 x 映射到 [0,1]^d（sigmoid 输出），对应 CDF 值
- `BreezeForest.inverse_map()` 中，bisection 接受任意 z ∈ [0,1]^d 并找到对应的 x
- 当前 `MultiBF.inverse_map()` 对每个组件 k 采样 z ~ Uniform(0.01, 0.99)^dim——整个有效 latent cube

- `BreezeForest.compute_dis()` 已有对数据均值/方差的计算（用于 bisection 的分布初始化），此机制可被扩展为 zone 估计

### 已有 Idea 1235（LZR）的局限

LZR 的步骤：
1. 计算各训练样本的 responsibility，选出高 responsibility 样本
2. 通过 forward pass 得到 z_i^k = f_k(x_i)
3. 对各维度独立计算百分位数：lo_k[d] = percentile(z_k[:, d], 5%), hi_k[d] = percentile(z_k[:, d], 95%)
4. 生成时从 Uniform([lo_k, hi_k]) 采样

**问题**：轴对齐 bounding box 是"维度独立"假设，无法捕获维度间协方差结构。

### 外部调研关键发现

**Langevin Dynamics in Latent Space（Coeurdoux et al., 2024, Machine Learning）**：
- Normalizing flow 在低概率区域（inter-cluster）存在 Jacobian 范数爆炸问题
- 在 latent space 做 Metropolis adjusted Langevin Algorithm（MALA）可规避这一问题
- Latent space 的 MALA 利用 Jacobian 变换，自然回避低密度区域
- 核心：MALA 采样到的 z 值天然集中在高 Jacobian 区域（即 cluster 中心周围）

**Enhanced Importance Sampling in Latent Space（2025，arXiv:2501.03394）**：
- 在 latent space 做 importance sampling 而非 data space，可有效避免 inter-mode 低密度区域
- 提议分布（proposal distribution）在 latent space 更易构造

**Stimper et al. (2022) "Resampling Base Distributions"**：
- 使用 rejection sampling 构建改进的 base distribution，解决 topological 不匹配
- G-LZR 是其简化版：不需要额外学习步骤，直接从数据估计有效区域

---

## 核心思路

用**高斯/椭球拟合**替换矩形 bounding box，更精确地描述每个组件 k 在 latent space 中的有效 z 分布区域：

### 方案 A：单变量高斯（快速版）

对 cluster k 的 latent 表示 Z_k = {z_i^k = f_k(x_i) : x_i ∈ D_k}，拟合一个多元高斯：

```
q_k(z) = N(z | μ_k, Σ_k)
```

其中 μ_k = mean(Z_k)，Σ_k = cov(Z_k)（dim × dim 协方差矩阵）。

生成时：从 q_k 截断采样（只取落在 [0.01, 0.99]^d 范围内的样本）：
```
z ~ TruncatedNormal(μ_k, Σ_k, lower=0.01, upper=0.99)
```

### 方案 B：PCA 椭球（轻量级结构捕获）

用 PCA 对 Z_k 做降维，识别主成分方向：
1. PCA 主成分 V_k（top-d 向量），解释 Z_k 的主要方差方向
2. 在 PCA 空间中用轴对齐椭球：主成分 i 的范围 = [μ_k,i - 3σ_k,i, μ_k,i + 3σ_k,i]
3. 从椭球内均匀采样，再映射回原始 latent 空间

### 方案 C：Truncated Gaussian Sampling（最简版）

不用协方差，只用对角 Gaussian 截断采样，保证样本在 [0.01, 0.99]^d 内：
```python
z = torch.randn(n_samples, dim) * std_k.unsqueeze(0) + mu_k.unsqueeze(0)
z = z.clamp(0.01, 0.99)  # 截断到有效范围
```

这比 LZR 的矩形均匀采样更集中于 cluster 中心，自然减少边界处的无效 z 值。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**核心数学逻辑**：

f_k 是 bijective 映射。对于两个分离的 cluster A（组件 k 负责）和 cluster B：
- cluster A 的点 {x_A} → 通过 f_k → 集中在 latent 子区域 Z_k^A
- cluster B 的点 {x_B} → 通过 f_k → 集中在 latent 子区域 Z_k^B
- inter-cluster 的点 → 通过 f_k → 位于 Z_k^A 和 Z_k^B 之间的低密度区域

**G-LZR 的优势**：
- 高斯拟合 Z_k^A 后，样本 z ~ N(μ_k, Σ_k) 的概率质量高度集中于 Z_k^A 中心
- Z_k^B 和 inter-cluster 区域的 z 值由于远离 μ_k，被高斯分布自然下压（概率极小）
- 与 LZR 的矩形框相比，高斯分布尊重了 Z_k^A 的实际形状（包括斜向分布）
- 即使 Z_k^A 和 Z_k^B 在某些轴上重叠，协方差矩阵能捕捉它们在斜向的分离

**与 Coeurdoux (2024) MALA 的比较**：
- MALA 在 latent space 迭代采样，精度高但计算开销大（每步需要 gradient 计算）
- G-LZR 用预计算的高斯参数做直接采样，计算开销极低，效果接近

---

## 与历史 Idea 的关系

| 历史 Idea | 关系 |
|-----------|------|
| **Idea 1235（LZR）** | **直接升级**。G-LZR 保留 LZR 的核心思路（限制 latent 采样区域），但将 zone 表示从"轴对齐矩形框"升级为"多元高斯/PCA 椭球"。更精确捕获 latent 分布形状，减少误包含的 inter-cluster z 值。 |
| **Idea 1541（K-Means + Hard-EM）** | **协同增强**。K-Means Dedicated Training 后，各组件的 latent zone 自然分离，G-LZR 的高斯估计更准确（zone 不重叠）。两者是最强组合。 |
| **Idea 1240（ICDR）** | **不冲突**。ICDR 是 training-time 修复，G-LZR 是 inference-time 修复，可并行使用，但 ICDR 优先级低于本方案和 K-Means 方案。 |

---

## 具体实现建议

### 步骤 1：扩展 `calibrate_latent_zones()` 以支持高斯估计

```python
def calibrate_gaussian_latent_zones(
    self, x_train, 
    percentile_low=1.0, percentile_high=99.0,
    mode='gaussian'  # 'gaussian', 'pca', 'box'
):
    """
    Fit per-component Gaussian/ellipsoidal latent zones.
    
    :param x_train: training data (N, dim)
    :param mode: 'gaussian' (full covariance), 'pca' (PCA ellipsoid), 'box' (original LZR)
    """
    self.gaussian_latent_zones = []
    
    with torch.no_grad():
        # Compute responsibilities (same as LZR)
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
            threshold = 1.0 / self.n_components
            mask = resp_k > threshold
            if mask.sum() < 10:
                topk = max(10, int(0.2 * len(resp_k)))
                _, idx = torch.topk(resp_k, topk)
                mask = torch.zeros_like(resp_k, dtype=torch.bool)
                mask[idx] = True
            
            x_k = x_train[mask]
            breeze_list = []
            z_k = bf.forward(x_k, breeze_list)  # (n_k, dim)
            
            if mode == 'gaussian':
                # Fit full multivariate Gaussian
                mu_k = z_k.mean(dim=0)          # (dim,)
                z_centered = z_k - mu_k
                cov_k = (z_centered.T @ z_centered) / (len(z_k) - 1)  # (dim, dim)
                # Add small regularization for numerical stability
                cov_k = cov_k + 1e-4 * torch.eye(self.dim)
                self.gaussian_latent_zones.append({
                    'mode': 'gaussian',
                    'mu': mu_k,
                    'cov': cov_k,
                    'L': torch.linalg.cholesky(cov_k)  # for sampling
                })
            
            elif mode == 'pca':
                # PCA-based ellipsoid
                mu_k = z_k.mean(dim=0)
                z_centered = z_k - mu_k
                U, S, Vt = torch.linalg.svd(z_centered, full_matrices=False)
                # Each principal component i has std = S[i] / sqrt(n_k - 1)
                std_pca = S / (len(z_k) - 1) ** 0.5
                self.gaussian_latent_zones.append({
                    'mode': 'pca',
                    'mu': mu_k,
                    'V': Vt.T,   # (dim, dim) rotation matrix
                    'std': std_pca  # (dim,) std along each PC
                })
            
            print(f"Component {k}: fitted {mode} zone, "
                  f"mu={mu_k.numpy().round(3)}, n_k={mask.sum()}")
    
    return self
```

### 步骤 2：支持高斯 latent zone 的生成函数

```python
def inverse_map_gaussian_zones(self, n_samples, max_gap=1e-3, n_sigma=2.0):
    """
    Generate samples using Gaussian latent zone sampling.
    :param n_sigma: number of std devs to sample within (truncation radius)
    """
    assert hasattr(self, 'gaussian_latent_zones'), \
        "Call calibrate_gaussian_latent_zones() first"
    
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)
    
    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        zone = self.gaussian_latent_zones[k]
        
        if zone['mode'] == 'gaussian':
            # Sample from multivariate Gaussian, reject outside [0.01, 0.99]^d
            mu, L = zone['mu'], zone['L']
            z_samples = []
            n_needed = n_k
            while len(z_samples) < n_needed:
                # Oversample and filter
                n_raw = int(n_needed * 3)
                eps = torch.randn(n_raw, self.dim)
                z_raw = mu + (L @ eps.T).T
                
                # Keep samples within valid range AND within n_sigma std devs
                in_range = (z_raw >= 0.01).all(dim=1) & (z_raw <= 0.99).all(dim=1)
                # Mahalanobis distance check
                z_c = z_raw - mu
                L_inv = torch.linalg.solve_triangular(L, z_c.T, upper=False).T
                mahal = (L_inv ** 2).sum(dim=1)
                in_ellipsoid = mahal < n_sigma ** 2 * self.dim
                
                valid = in_range & in_ellipsoid
                z_samples.append(z_raw[valid])
            
            z_k = torch.cat(z_samples, dim=0)[:n_k]
        
        elif zone['mode'] == 'pca':
            mu, V, std = zone['mu'], zone['V'], zone['std']
            eps = torch.randn(int(n_k * 3), self.dim) * std.unsqueeze(0)
            z_pca = eps @ V.T + mu
            in_range = (z_pca >= 0.01).all(dim=1) & (z_pca <= 0.99).all(dim=1)
            # Truncate to n_sigma std devs in PCA space
            in_ellipsoid = ((eps / std.unsqueeze(0)) ** 2).sum(dim=1) < n_sigma ** 2 * self.dim
            z_k = z_pca[in_range & in_ellipsoid][:n_k]
            if len(z_k) < n_k:  # fallback
                z_k = (z_pca[in_range])[:n_k]
        
        x_k = self.components[k].inverse_map(z_k, max_gap=max_gap)
        results[mask] = x_k
    
    return results
```

### 步骤 3：最简版（方案 C - 对角高斯截断）

若不想引入协方差矩阵，可用最简版本（仍比 LZR 好，因为高斯中心权重更高）：

```python
def calibrate_diagonal_gaussian_zones(self, x_train):
    """Lightweight: fit diagonal Gaussian per component latent zone."""
    self.diag_zones = []
    with torch.no_grad():
        # ... responsibility computation same as above ...
        for k, bf in enumerate(self.components):
            # ... mask selection same as above ...
            z_k = bf.forward(x_k, [])
            self.diag_zones.append({
                'mu': z_k.mean(0),
                'std': z_k.std(0).clamp(min=0.01)
            })

def inverse_map_diag_gaussian(self, n_samples, n_sigma=2.5, max_gap=1e-3):
    """Sample from diagonal truncated Gaussian per component."""
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)
    
    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        mu, std = self.diag_zones[k]['mu'], self.diag_zones[k]['std']
        
        # Truncated Gaussian sampling (rejection)
        z_acc = []
        while sum(len(z) for z in z_acc) < n_k:
            eps = torch.randn(n_k * 5, self.dim)
            z_raw = mu + eps * std
            valid = (z_raw >= 0.01).all(1) & (z_raw <= 0.99).all(1)
            valid &= (eps.abs() < n_sigma).all(1)  # within n_sigma std devs
            z_acc.append(z_raw[valid])
        
        z_k = torch.cat(z_acc, 0)[:n_k]
        results[mask] = self.components[k].inverse_map(z_k, max_gap=max_gap)
    
    return results
```

### 超参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `mode` | `'pca'`（首选）或 `'diagonal'`（轻量版） | 完整协方差矩阵在高维时可能不稳定 |
| `n_sigma` | 2.0 – 2.5 | 控制截断半径。2σ ≈ 95% 的概率质量，2.5σ ≈ 98.8% |
| `percentile_low/high` | 2% / 98% | 用于 responsibility 筛选的置信区间 |
| 最优 pipeline | K-Means Dedicated + G-LZR | 训练后组件专一 → zone 估计准确 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **协方差矩阵不稳定** | 若 n_k 过小（< 20），协方差估计不可靠 | 使用对角高斯（方案 C），或添加正则化 `+ 1e-4 * I` |
| **拒绝率过高** | 若有效 z 区域很小，高斯采样的拒绝率高 | 增大 `n_sigma`，或切换到方案 C（diagonal，拒绝率更低）|
| **Zone 重叠（soft-EM 训练后）** | 若组件未充分专一化，各 zone 的高斯仍会重叠 | 先做 K-Means Dedicated Training（Idea 1541），再做 G-LZR |
| **高维 PCA 不稳定** | 在高维数据（dim > 10）中，低阶 PCA 可能丢失重要方向 | 使用全秩 PCA（保留所有 dim 个主成分），只是旋转了坐标系 |
| **bisection 数值问题** | 从 Gaussian zone 采样到的 z 可能使 bisection 数值不稳定（若 z 接近 0 或 1） | 截断时强制 z ∈ [0.02, 0.98]，比 LZR 默认的 [0.01, 0.99] 更保守 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（与 Idea 1541 并列，且无需重训练）**

理由：
1. **直接升级 LZR（1235）**：解决矩形 bounding box 的核心缺陷，可在现有已训练模型上立即应用
2. **无需重训练**：和 LZR 一样，只需在 inference 阶段做一次 calibration（几分钟内完成）
3. **理论支撑**：Coeurdoux (2024) 证明 latent space 采样可规避 inter-cluster 低密度区域的 Jacobian 爆炸；高斯拟合是最自然的参数化方案
4. **与 K-Means Dedicated Training 协同**：Dedicated Training 后，latent zone 自然分离，高斯拟合的精度远高于 soft-EM 训练后的估计
5. **渐进降级**：PCA 椭球 → 对角高斯 → 矩形框，提供灵活的复杂度-精度权衡

---

## 参考文献

- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning*. https://arxiv.org/abs/2305.12149
- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*. https://proceedings.mlr.press/v151/stimper22a.html
- arXiv:2501.03394 (2025). "Enhanced Importance Sampling Through Latent Space Exploration in Normalizing Flows."
- Pires, G. & Figueiredo, M. (2020). "Variational Mixture of Normalizing Flows." *ESANN 2020*.
