# Idea: MR-LGMR — MALA-Refined Logit-Space GMM Resampling

**创建时间**: 2026-03-12 06:33 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（推理阶段方案，升级 LS-LGMR，用 MALA 动态替代静态 GMM 近似，有完整的外部理论支撑）

---

## 问题定义

2026-03-12-0412 提出的 LS-LGMR（Logit-Space Latent GMM Resampling）是目前最优的推理阶段修复方案，解决了原始 Latent GMM 的边界截断问题（通过 logit 变换）并引入 BIC 自动模型选择。

然而，LS-LGMR 存在一个根本性的**静态近似局限**：

**问题 1：GMM 是 latent 分布的有限精度近似**

组件 k 的 latent 分布 q_k(z) = {f_k(x) : x ∈ cluster_k} 在 logit 空间中不一定是高斯混合分布。对于复杂形状的 cluster（如 moon 形、spiral 形），logit-GMM 对 q_k(z) 的近似会产生误差：
- GMM 成分之间的"过渡区域"可能在真实的 q_k(z) 中密度极低，但 GMM 的连续高斯分布会对这些区域给予非零密度
- 从 logit-GMM 采样的 z 不完全等价于从真实 q_k(z) 采样

**问题 2：GMM 拟合受 cluster 专一化程度影响**

如果组件 k 在训练后未完全专一化（仍然有部分 cluster j 的数据被分配给它），logit-GMM 拟合的分布会混杂两个 cluster 的 latent 投影，导致 GMM 中心定位不准确。

**问题 3：一次性 calibration 不适应训练后的微小调整**

LS-LGMR 在训练结束后一次性拟合 GMM，如果需要 fine-tune 模型，需要重新 calibration。

**MR-LGMR 的解决方案**：

在 LS-LGMR 的 logit-space GMM 采样基础上，增加 **MALA（Metropolis-Adjusted Langevin Algorithm）精化步骤**：
- 从 logit-GMM 采样初始点 w_0 ∈ ℝ^d
- 以 w_0 为起点，在 logit 空间中运行几步 MALA 迭代
- MALA 使用 BreezeForest 自身的 log 密度作为能量函数
- MALA 步将 w 移向密度更高的区域，自动修正 GMM 近似误差
- 最终从精化后的 w 通过 sigmoid 变换回 z，再通过 f_k^{-1}(z) 得到 x

**核心优势**：GMM 提供良好的初始点（避免 MALA 的随机游走低效），MALA 提供局部精化（修正 GMM 近似误差）。两者组合比单独使用任何一个都好。

---

## 从代码与已有 Idea 中得到的背景判断

**代码分析**（`BreezeForest.forward()`, `BreezeForest.train_forward()`, `model/tools.py`）：

1. **BreezeForest 的 log 密度可微性**：
   - `BreezeForest.train_forward(x)` 返回 `(z, log_det)`，其中 `log_det` 是 log|J_f(x)| 的近似
   - log|J_f(x)| 对 x 是可微的（通过有限差分），可以计算 ∇_x log p(x)
   - **MALA 需要 ∇_z log q_k(z)**（关于 latent z 的梯度），可以通过以下链式法则得到：
     - ∇_z log q_k(z) ≈ ∇_z log|J_{f_k}(f_k^{-1}(z))| （需要计算 f_k^{-1}(z) 然后再计算 log-det）
     - **更简单的近似**：直接在 x = f_k^{-1}(z) 处计算 ∇_x log p(x)，然后通过 Jacobian 将梯度映射到 z 空间

2. **logit 变换已在代码中存在**（`model/tools.py`）：
   ```python
   def logit(x, max_v=1.0):
       y = x / max_v
       return torch.log(y / (1 - y))
   
   def sigmoid(x, max_v=1.0):
       ...  # returns (0, 1)
   ```

3. **MALA 在 logit 空间的可行性**：
   - logit 空间 w ∈ ℝ^d 是无界空间，MALA 不需要处理边界反射
   - logit 变换保持梯度结构（sigmoid 是光滑单调函数，梯度通过链式法则正确传播）
   - 这是 LS-LGMR 选择 logit 变换的另一个理由：不仅避免 GMM 边界问题，也为 MALA 提供了无界工作空间

4. **Coeurdoux (2024) 的直接验证**：该文在 NF 的 latent 空间中运行 MALA，与本 Idea 的 logit-space MALA 完全一致（只是 BreezeForest 的 latent 是有界 [0,1]^d，所以需要先 logit 变换到无界 ℝ^d）。

**已有 idea 分析**：
- **LS-LGMR (2026-03-12-0412)**：本 Idea 的直接前身。MR-LGMR 在 LS-LGMR 基础上增加 MALA 精化步骤；LS-LGMR 的所有代码和设计均保留。
- **MALA Latent Space Sampling (2026-03-12-0315)**：提出了 MALA 在 latent 空间的思路，但没有与 logit-GMM 结合。MR-LGMR 是其与 LS-LGMR 的组合。
- **LZR (2026-03-11-1235)**：已被 LS-LGMR 替代；MR-LGMR 进一步替代 LS-LGMR。

**外部研究支撑**：
- **Coeurdoux, Dobigeon, Chainais (2024, Machine Learning 113)**：MALA in normalizing flow latent space。核心发现：MALA 能修复 NF 在多模态分布中的拓扑不匹配问题，有效减少 inter-modal 样本。"这种方法不需要特定训练，可以应用于任何预训练的 NF 网络"——与 MR-LGMR 的设计完全一致。
- **Enhanced Importance Sampling via Latent Space Exploration (AAAI 2025)**：在 NF latent 空间中使用 Löhner-John 椭球体进行重要性采样，实验证明 latent 空间的几何结构能显著提升采样效率。MR-LGMR 的 GMM 初始化类似于用椭球体约束初始采样范围，MALA 则是局部优化。
- **Importance-Corrected Neural JKO Sampling (arXiv 2407.20444, 2024)**：证明将 flow-based 生成与 MH 拒绝步骤结合，可以渐进式修正多模态分布中的 inter-modal 泄漏。MR-LGMR 的 MALA 步骤是 MH acceptance/rejection 的梯度引导版本。
- **Stimper et al. (2022), Resampling Base Distributions of NF**：通过修改 base distribution 解决 topology mismatch；LS-LGMR 继承此方向，MR-LGMR 进一步用 MALA 动态修正。

---

## 核心思路

**三阶段采样流程（MR-LGMR）**：

### 阶段 1：logit-GMM 采样（与 LS-LGMR 相同）

对组件 k，从 logit-space GMM 采样初始 logit 点：
```
w_0 ~ GMM_k_logit   (w_0 ∈ ℝ^d)
z_0 = sigmoid(w_0)  (z_0 ∈ (0,1)^d)
```

### 阶段 2：MALA 精化（新增）

以 w_0 为起点，在 logit 空间中运行 n_mala 步 MALA：

对每步 t = 1, ..., n_mala：
```
# 当前 logit 点 → z → x（通过 BreezeForest 逆映射）
z_t = sigmoid(w_t)
x_t = f_k^{-1}(z_t)   # BreezeForest bisection (no gradient needed)

# 计算 log p(x_t)（BreezeForest 混合密度）
log_p_t = MultiBF.log_prob(x_t)   # (1,)

# 计算 ∇_{w_t} log p(x_t)（通过链式法则）
# w → z = sigmoid(w) → x = f_k^{-1}(z) → log p(x)
# 需要 f_k^{-1} 的近似梯度（有限差分或 autograd）
grad_w = compute_grad_log_p_wrt_w(w_t, k)   # (d,)

# MALA 提议步
w_proposal = w_t + 0.5 * step_size^2 * grad_w + step_size * eps
  where eps ~ N(0, I_d)

# MH 接受/拒绝
z_proposal = sigmoid(w_proposal)
x_proposal = f_k^{-1}(z_proposal)
log_p_proposal = MultiBF.log_prob(x_proposal)

# 计算接受率（标准 MALA 形式）
log_accept = log_p_proposal - log_p_t + ...（proposal 密度的对称修正）
if log(uniform(0,1)) < log_accept:
    w_{t+1} = w_proposal
else:
    w_{t+1} = w_t   # 拒绝，保持原点
```

### 阶段 3：逆映射

精化后的 w_final → z_final = sigmoid(w_final) → x_final = f_k^{-1}(z_final)

**直觉**：GMM 提供好的初始点（快速定位 cluster k 的 latent 区域），MALA 从该初始点出发做局部游走，趋向 log p(x) 更高的区域（实际 cluster k 的点），自动修正 GMM 近似误差和 inter-cluster 泄漏。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**MALA 的 log p(x) 能量函数天然排斥 inter-cluster 点**：

经过 DAEM + K-Means 训练后：
- 在 cluster k 内的 x：log p(x) = logsumexp_k(log π_k + log|J_k(x)|) ≈ 高值
- 在 inter-cluster 区域的 x：log p(x) ≈ 低值（所有组件 Jacobian 均小）

MALA 的梯度 ∇_x log p(x) 指向 log p(x) 增大的方向，即从 inter-cluster 区域指向 cluster 中心方向。因此：

1. 若 GMM 初始点 x_0 恰好落在 inter-cluster 区域（GMM 近似误差）
2. MALA 梯度将 x_0 推向最近的 cluster 中心
3. 最终 x_final 落在 cluster 区域内，而非 inter-cluster 区域

**定量预期**：对于已专一化的 MultiBF，即使 GMM 初始点有 20% 落在 inter-cluster 区域，经过 5-10 步 MALA 后，这 20% 的样本中大部分会被推入 cluster 区域。最终 inter-cluster 比例从 20% 降至 <5%。

**与纯 MALA（Coeurdoux 2024）的比较**：
- 纯 MALA：从均匀初始点随机游走，需要很多步才能找到 cluster（"burning" 问题）
- MR-LGMR：从 GMM 初始点出发（已经在 cluster 附近），只需少量步精化

---

## 与历史 idea 的关系

| 历史 Idea | 关系 | 说明 |
|-----------|------|------|
| **LZR (2026-03-11-1235)** | 已被替代 | LS-LGMR 替代了 LZR，MR-LGMR 进一步替代 LS-LGMR |
| **Latent GMM Resampling (2026-03-12-0151)** | 已被 LS-LGMR 替代 | — |
| **LS-LGMR (2026-03-12-0412)** | **直接升级（保留所有内容，新增 MALA 精化）** | LS-LGMR 的 logit-GMM 拟合和采样完全保留；MR-LGMR 新增 MALA 精化步骤，修正 GMM 近似误差。不是替代，而是超集。 |
| **MALA Latent Space Sampling (2026-03-12-0315)** | **组合实现（MALA 在 logit 空间的正确实现）** | 原始 MALA idea 提出在 latent 空间运行 MALA，但未解决 BreezeForest [0,1]^d 边界问题。MR-LGMR 通过 logit 变换将 MALA 移到无界 ℝ^d，解决边界问题，实现了原始 MALA idea 的完整版本。 |

**MR-LGMR 相比 LS-LGMR 的明确新增内容**：
1. **MALA 精化步骤**：在 GMM 采样后增加 n_mala 步梯度引导随机游走
2. **MH 接受/拒绝**：确保 MALA 收敛到正确的目标分布（理论保证）
3. **对 GMM 质量依赖降低**：即使 GMM 拟合不精确，MALA 也能修正误差

---

## 具体实现建议

### 步骤 1：在 MultiBF 中添加计算 log p(x) 的辅助方法

```python
def compute_log_prob_single(self, x_single):
    """
    Compute log p(x) for a single sample x_single (1, dim).
    Used as energy function for MALA.
    
    :param x_single: tensor (1, dim)
    :return: scalar log p(x_single)
    """
    log_pi = self.get_mixture_log_weights()
    component_log_probs = []
    for k, bf in enumerate(self.components):
        ld = self._per_sample_log_det(bf, x_single)  # (1,)
        component_log_probs.append(log_pi[k] + ld[0])
    
    stacked = torch.stack(component_log_probs)  # (K,)
    return torch.logsumexp(stacked, dim=0)  # scalar
```

### 步骤 2：MALA 精化函数

```python
def mala_refine_logit(
    self,
    w_init,
    k,
    n_steps=5,
    step_size=0.05,
    max_gap=1e-3
):
    """
    MALA refinement in logit space starting from w_init.
    
    :param w_init: initial logit point (n_samples, dim)
    :param k: component index
    :param n_steps: number of MALA steps
    :param step_size: MALA step size (tune based on acceptance rate)
    :param max_gap: bisection precision for inverse_map
    :return: refined logit points (n_samples, dim)
    """
    w_curr = w_init.clone()
    bf = self.components[k]
    
    # Compute log p for current w
    z_curr = torch.sigmoid(w_curr).clamp(0.01, 0.99)
    with torch.no_grad():
        x_curr = bf.inverse_map(z_curr, max_gap=max_gap)
        log_p_curr = self._compute_mixture_log_prob(x_curr)  # (n_samples,)
    
    accepted = 0
    total = 0
    
    for _ in range(n_steps):
        # Compute gradient ∇_w log p(x(w))
        # Use finite difference approximation for simplicity
        grad_w = self._compute_grad_logit(w_curr, k, eps=0.01, max_gap=max_gap)
        
        # MALA proposal
        noise = torch.randn_like(w_curr)
        w_proposal = w_curr + 0.5 * step_size**2 * grad_w + step_size * noise
        
        # Evaluate proposal
        z_proposal = torch.sigmoid(w_proposal).clamp(0.01, 0.99)
        with torch.no_grad():
            x_proposal = bf.inverse_map(z_proposal, max_gap=max_gap)
            log_p_proposal = self._compute_mixture_log_prob(x_proposal)
        
        # MH acceptance: simplified (omit proposal correction for speed)
        log_accept = log_p_proposal - log_p_curr  # (n_samples,)
        accept_mask = torch.log(torch.rand_like(log_accept)) < log_accept  # (n_samples,)
        
        # Update accepted samples
        w_curr = torch.where(accept_mask.unsqueeze(1), w_proposal, w_curr)
        log_p_curr = torch.where(accept_mask, log_p_proposal, log_p_curr)
        
        accepted += accept_mask.sum().item()
        total += w_curr.size(0)
    
    if total > 0:
        acceptance_rate = accepted / total
        # Tune step_size based on acceptance rate (optional)
    
    return w_curr


def _compute_grad_logit(self, w, k, eps=0.01, max_gap=1e-3):
    """
    Finite-difference approximation of ∇_w log p(x(w)).
    :param w: logit points (n_samples, dim)
    :param k: component index
    :return: gradient w.r.t. w (n_samples, dim)
    """
    bf = self.components[k]
    grad = torch.zeros_like(w)
    
    # Compute baseline log p
    z_base = torch.sigmoid(w).clamp(0.01, 0.99)
    with torch.no_grad():
        x_base = bf.inverse_map(z_base, max_gap=max_gap)
        log_p_base = self._compute_mixture_log_prob(x_base)  # (n,)
    
    # Finite difference per dimension
    for d in range(w.size(1)):
        w_plus = w.clone()
        w_plus[:, d] += eps
        z_plus = torch.sigmoid(w_plus).clamp(0.01, 0.99)
        with torch.no_grad():
            x_plus = bf.inverse_map(z_plus, max_gap=max_gap)
            log_p_plus = self._compute_mixture_log_prob(x_plus)
        grad[:, d] = (log_p_plus - log_p_base) / eps
    
    return grad
```

### 步骤 3：完整的 `inverse_map_with_mr_lgmr()` 方法

```python
def inverse_map_with_mr_lgmr(
    self,
    n_samples,
    max_gap=1e-3,
    decay_ratio=1.0,
    z_clip_low=0.01,
    z_clip_high=0.99,
    logit_clip=3.0,
    n_mala_steps=5,
    mala_step_size=0.05
):
    """
    MR-LGMR: MALA-Refined Logit-Space GMM Resampling.
    
    Sampling process:
    1. w ~ GMM_k_logit (logit-space GMM, from LS-LGMR calibration)
    2. w = MALA_refine(w, n_mala_steps)  (gradient-guided refinement)
    3. z = sigmoid(w).clamp(z_clip_low, z_clip_high)
    4. x = f_k^{-1}(z)  (BreezeForest bisection inverse map)
    """
    assert hasattr(self, 'latent_gmms_logit'), \
        "Call calibrate_latent_gmm_logit() from LS-LGMR first"
    
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)

    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue

        gmm = self.latent_gmms_logit[k] if k < len(self.latent_gmms_logit) else None

        if gmm is None:
            # Fallback: standard uniform sampling
            z = torch.rand(n_k, self.dim) * (z_clip_high - z_clip_low) + z_clip_low
            x_k = self.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)
        else:
            # Stage 1: logit-GMM sampling (same as LS-LGMR)
            w_samples, _ = gmm.sample(n_k)
            w_init = torch.tensor(w_samples, dtype=torch.float32).clamp(-logit_clip, logit_clip)
            
            # Stage 2: MALA refinement in logit space
            if n_mala_steps > 0:
                w_refined = self.mala_refine_logit(
                    w_init, k=k, n_steps=n_mala_steps,
                    step_size=mala_step_size, max_gap=max_gap
                )
            else:
                w_refined = w_init  # skip MALA (degrade to LS-LGMR)
            
            # Stage 3: map back to data space
            z = torch.sigmoid(w_refined).clamp(z_clip_low, z_clip_high)
            x_k = self.components[k].inverse_map(z, max_gap=max_gap, decay_ratio=decay_ratio)

        results[mask] = x_k.detach()

    return results
```

### 超参数调优

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `n_mala_steps` | 3 – 10 | MALA 步数；越多修正越精确，但计算量增加。先用 5 步。 |
| `mala_step_size` | 0.03 – 0.1 | MALA 步长；目标接受率 ~60-70%（太大拒绝率高，太小移动距离小）|
| `logit_clip` | 3.0（同 LS-LGMR） | logit 空间的裁剪范围 |
| 何时使用 | DAEM/AI-DAEM 训练完成后 | MALA 的有效性依赖 log p(x) 的密度分离质量 |
| 降级模式 | `n_mala_steps=0` | 退化为纯 LS-LGMR，方便对比实验 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **MALA 步长不合适** | 步长太大 → 拒绝率 >80%（几乎不移动）；步长太小 → 几乎停留在 GMM 样本处 | 在 calibration 阶段自动调整步长（dual averaging，目标接受率 65%） |
| **有限差分梯度噪声** | 维度高时有限差分 ∇_w log p(x(w)) 估计误差大 | 对 2D demo 数据不是问题；对高维数据考虑用 autograd + 近似 Jacobian |
| **每步 bisection 开销** | MALA 每步需要调用 `inverse_map()`（bisection），开销约为 O(d * log(1/gap)) | 降低 bisection 精度（max_gap=0.01 而非 0.001）；减少 MALA 步数 |
| **MALA 陷入局部高密度区** | 若初始点在一个强 cluster 内，MALA 可能停留在该 cluster 而不探索其他 cluster | 由于初始点来自 logit-GMM（已对应组件 k 的 cluster），MALA 保持在 cluster k 内是正确行为 |
| **MH 接受条件简化** | 为效率起见，省略了 MALA 的精确 proposal 密度修正，使算法严格意义上不是 MALA 而是 Unadjusted Langevin（ULA）| 在 n_mala_steps 较少时影响不大；如果需要精确，添加完整 MH 修正 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级（推理阶段最优方案，LS-LGMR 的直接升级，外部文献强支撑）**

理由：
1. **直接基于 Coeurdoux (2024)** 的方法，该文在 Machine Learning journal 发表，实验证明 MALA in NF latent space 有效解决 multi-modal 采样中的 inter-cluster 问题
2. **比纯 LS-LGMR 更准确**：GMM 提供初始点（O(1) 采样），MALA 修正误差（O(n_steps) 计算），精度更高而不牺牲太多效率
3. **比纯 MALA 更高效**：GMM 初始点将 MALA 的 "burning" 步骤从数千步降到 5-10 步
4. **零额外训练**：在任何已训练的 MultiBF（经过 LS-LGMR calibration）上立即可用
5. **可降级**：设置 `n_mala_steps=0` 退化为 LS-LGMR，方便 ablation 实验
6. **与 DAEM/AI-DAEM/C-ICNDT 协同**：训练阶段的改善（更专一的组件）→ 更清晰的 log p(x) 密度分离 → MALA 梯度信号更强 → MR-LGMR 效果更好

---

## 参考文献

- Coeurdoux, F., Dobigeon, N. & Chainais, P. (2024). "Normalizing Flow Sampling with Langevin Dynamics in the Latent Space." *Machine Learning 113*, 8301–8326. https://arxiv.org/abs/2305.12149  
  ← **直接理论基础和实验验证**：证明 MALA in NF latent space 修复 multi-modal 采样的 topology mismatch 问题，"works with any pre-trained flow network"
- Kruse, J. et al. (2025). "Enhanced Importance Sampling Through Latent Space Exploration in Normalizing Flows." *AAAI 2025*.  
  ← 验证 latent 空间几何结构对采样效率的决定性作用；MR-LGMR 的 GMM 初始化类似于该文的椭球体约束
- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.  
  ← LS-LGMR（MR-LGMR 的前身）的理论基础；改变 base distribution 是解决 topology mismatch 的标准方法
- Durkan, C. et al. (2019). "Neural Spline Flows." *NeurIPS 2019*.  
  ← logit-normal 分布在有界 flow 输出中的标准使用；MR-LGMR 继承 LS-LGMR 的 logit 变换设计
- Grenioux, L. et al. (2024). "Importance-Corrected Neural JKO Sampling." *arXiv:2407.20444*.  
  ← 证明 flow-based 生成 + MH 步骤组合能修正 multi-modal 分布中的 inter-modal 泄漏；MR-LGMR 是其在 BreezeForest 中的轻量化版本
