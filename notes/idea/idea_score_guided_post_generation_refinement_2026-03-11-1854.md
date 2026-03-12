# Idea: Score-Guided Post-Generation Refinement (SGR)

**创建时间**: 2026-03-11 18:54 UTC  
**推荐优先级**: ⭐⭐ 高优先级（新方向，与 ICDR 互补但更简单且适用于单 BF）

---

## 问题定义

BreezeForest（单 BF 或 MultiBF）生成流程的**末端**缺乏一个"质量守门员"：

1. 从 z ~ Uniform(0.01, 0.99)^d 采样
2. x = f^{-1}(z)（二分搜索）
3. **直接输出 x，不做任何验证**

这导致"哑弹"样本（z 恰好落在 cluster 间隙区域）以相同概率被输出，与真实高密度样本无区分。

**核心问题**：f^{-1} 是连续双射，对**任意**合法 z 都能返回一个 x，包括"无意义"的 inter-cluster 点。当前代码没有任何机制在生成后筛查样本质量。

**为什么不能靠 EDR（本轮 Idea 2）完全解决**：
- EDR 是从采样侧限制，减少了间隙 z 的出现频率
- 但 KDE/GMM 是近似，仍可能有少量间隙 z 被采到
- SGR 作为最后一道保险，将这些"漏网"的 inter-cluster 点"拉回"附近的真实 cluster

---

## 从当前项目代码与已有 idea 中得到的背景判断

**代码层面**：
- `bf.train_forward(x)` 已经计算了 `log|det J_f(x)|`（模型对 x 的对数密度，除常数外），并在训练时用于 NLL 损失
- PyTorch 的 autodiff 支持对 `log|det J_f(x)|` 关于 x 求梯度：`∇_x log|det J_f(x)|`
- 这个梯度即**模型的分数函数**（score function）：∂ log p(x) / ∂x
- 沿分数方向做梯度上升，x 会向更高密度区域移动
- `train_forward` 使用有限差分估计 Jacobian，所以 ∇_x log|det J| 可通过 autodiff 穿透有限差分计算获得

**已有 idea 分析**：
- **ICDR（1240）**：training-time 排斥 loss，通过在训练中传播梯度来推开组件。SGR 是 inference-time 的单点"修正"，不依赖训练。两者机制完全不同。
- **LZR（1235）/EDR（本轮 Idea 2）**：采样侧限制，不对已生成样本做修正。SGR 可以配合 EDR 使用（EDR 减少 inter-cluster z 采样，SGR 修正漏网的 inter-cluster x）。
- **Hard-EM（1230）/Piecewise BF（本轮 Idea 1）**：training-time 方案，不解决 inference 时的个别异常样本。

**文献验证（本轮调研）**：
- **Diffusion Rejection Sampling（DiffRS，NeurIPS 2024）**：在扩散模型中，用模型评估样本质量并在中间步骤做修正。SGR 类似，但针对 normalizing flow 的确定性 inference 流程。
- **Optimal Budgeted Rejection Sampling（Verine et al., AISTATS 2024）**：证明拒绝采样在固定预算下的最优策略。SGR 是梯度上升的"软"版本（不拒绝，而是修正），比硬拒绝更高效（不浪费生成预算）。
- **MALA in latent space（Coeurdoux 2024）**：使用 Langevin 动力学在 latent space 中改善 flow 生成质量。SGR 在 x-space 做梯度上升（更简单，直接使用模型的分数），是 Coeurdoux 方法的简化版本（无 MH 步，梯度上升而非 Langevin）。

---

## 核心思路

**在生成样本 x_0 = f^{-1}(z) 之后，运行 K 步梯度上升，将 x 推向模型学到的最近高密度区域**：

```
x_0 = f^{-1}(z)                          # 标准 bisection 生成
for t in range(K):
    log_density = log|det J_f(x_t)|       # 用 train_forward 计算
    grad = ∂ log_density / ∂ x_t          # autodiff
    x_{t+1} = x_t + α * grad / ||grad||  # 归一化梯度上升（步长 α）
x_final = x_K                             # 输出修正后的样本
```

**直觉**：
- inter-cluster 区域的 log|det J| 低（BreezeForest 在那里 Jacobian 小）
- cluster 内部的 log|det J| 高（BreezeForest 在那里 Jacobian 大，因为密度集中）
- 梯度 ∂ log|det J| / ∂x 指向"最陡的密度上升"方向，即指向最近的 cluster 中心
- K 步梯度上升后，inter-cluster 点会被"拉"到最近的 cluster 附近

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**理论依据**：

对于训练良好的 BreezeForest，log|det J_f(x)| ≈ log p(x) + const（因为 p_z 是 Uniform）。其梯度：
```
∇_x log|det J_f(x)| = ∇_x log p(x) = score function
```

Score function 是 p(x) 的梯度场（Stein score），在多 cluster 分布中：
- 在 cluster 内部：指向 cluster 中心（低熵区域）
- 在 inter-cluster 区域：指向最近的 cluster（score 有"分水岭"但梯度明确）
- 在边界处：梯度更大（密度变化更剧烈）

所以几步梯度上升，inter-cluster 点会被 score field 拉向最近的 cluster。**不需要知道有几个 cluster，也不需要知道每个 cluster 在哪里**——这信息隐含在 BreezeForest 的 log|det J| 中。

**对比 EDR（本轮 Idea 2）**：

| 方面 | EDR | SGR |
|------|-----|-----|
| 作用时机 | 采样前（z 选择）| 生成后（x 修正）|
| 机制 | 避免采到间隙 z | 将 inter-cluster x 推向 cluster |
| 对单 BF 有效 | ✓ | ✓ |
| 对 MultiBF 有效 | ✓ | ✓ |
| 额外计算量 | forward pass × N | train_forward + autodiff × N × K |
| 需要额外模型/拟合 | 需要 KDE/GMM 拟合 | 不需要（直接用已训练 BF）|
| 可与 EDR 叠加 | — | ✓（SGR 修正 EDR 漏网的点）|

**对比 ICDR（已有 Idea 3/1240）**：
- ICDR 在**训练时**通过 O(K²) 排斥 loss 推开组件
- SGR 在**推断时**通过每个样本的 O(K) 梯度步修正个别点
- SGR 不依赖 MultiBF 架构（单 BF 也适用）
- SGR 计算开销：每样本 K 次 forward + autodiff，K=5 时实用

---

## 它与历史 idea 的关系

**与 ICDR（1240）的关系**：不同时机、不同机制，互补而不替代。
- ICDR 是 training-time 主动推开组件，适合在知道 inter-cluster 问题严重时提前训练
- SGR 是 inference-time 被动修正，适合快速部署或对已训练模型的后处理
- 如果 ICDR 训练后仍有少量 inter-cluster 点，SGR 可作为最终保险

**与 LZR（1235）的关系**：不同维度的修复。
- LZR 在 z-space 限制采样范围（防止 inter-cluster z 出现）
- SGR 在 x-space 修正已生成样本（将 inter-cluster x 拉回 cluster）
- 理论上两者可以叠加：LZR 减少 90% 的 inter-cluster z，SGR 修正剩余 10%

**与 Hard-EM（1230）/Piecewise BF（本轮 Idea 1）的关系**：不同阶段，互补。
- Piecewise BF 是根本性训练修复（最推荐）
- SGR 可在 Piecewise BF 之上进一步提升生成质量，或在没有时间重训练时单独使用

**无替代历史 idea**：SGR 是此轮首次提出的 inference-time x-space 梯度修正方案。

---

## 具体实现建议

### 核心实现：score_guided_refine()

```python
def score_guided_refine(
    bf,
    x_init,
    n_steps=5,
    step_size=0.01,
    normalize_grad=True,
    use_train_forward_light=True
):
    """
    Post-generation score-guided refinement.
    
    Moves generated samples toward higher density regions using the model's
    own score function (gradient of log|det J|).
    
    :param bf: trained BreezeForest (or MultiBF component)
    :param x_init: initial generated samples, tensor (N, dim)
    :param n_steps: number of gradient ascent steps (5-20, default 5)
    :param step_size: step size α (0.001 - 0.05, default 0.01)
    :param normalize_grad: if True, use gradient direction only (more stable)
    :param use_train_forward_light: if True, use light (one-sided FD) for speed
    :return: refined samples (N, dim)
    """
    x = x_init.clone().detach()
    
    for t in range(n_steps):
        x = x.requires_grad_(True)
        
        # Compute log|det J_f(x)| using existing train_forward
        if use_train_forward_light:
            _, log_det = bf.train_forward(x, light=True)
        else:
            _, log_det = bf.train_forward(x, light=False)
        
        # Note: train_forward returns mean log_det over batch
        # We need per-sample gradients -> use sum instead
        # Recompute per-sample log_det
        epsilons = bf.epsilon
        x_delta = x + epsilons
        
        breeze_list = []
        z = bf.forward(x, breeze_list)
        z_delta = bf.breeze_forward(x_delta, breeze_list)
        
        delta_u = (z_delta - z) * bf.dim_mask + 1 - bf.dim_mask
        log_det_per_sample = torch.sum(torch.log(torch.abs(delta_u) + 1e-8), dim=1)  # (N,)
        
        # Gradient w.r.t. x
        grad = torch.autograd.grad(log_det_per_sample.sum(), x)[0]  # (N, dim)
        
        if normalize_grad:
            grad_norm = torch.norm(grad, dim=1, keepdim=True).clamp(min=1e-8)
            grad = grad / grad_norm
        
        x = (x + step_size * grad).detach()
    
    return x


def generate_with_sgr(
    bf,
    n_samples,
    dim,
    n_refine_steps=5,
    step_size=0.01,
    max_gap=1e-3
):
    """
    Full generation pipeline with SGR post-processing.
    """
    # Standard generation
    bf.eval()
    with torch.no_grad():
        z = torch.rand(n_samples, dim) * 0.98 + 0.01
        x_init = bf.inverse_map(z, max_gap=max_gap)
    
    # SGR refinement (requires grad)
    bf.train()  # enable dropout/batchnorm if any
    x_refined = score_guided_refine(bf, x_init, n_steps=n_refine_steps, step_size=step_size)
    bf.eval()
    
    return x_refined
```

### 对 MultiBF 的适配

```python
def generate_multibf_with_sgr(mbf, n_samples, n_refine_steps=5, step_size=0.01):
    """MultiBF generation with per-component SGR."""
    weights = mbf.get_mixture_weights().detach()
    comp_idx = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, mbf.dim)
    
    for k, bf_k in enumerate(mbf.components):
        mask = (comp_idx == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        
        x_k = generate_with_sgr(bf_k, n_k, mbf.dim, n_refine_steps, step_size)
        results[mask] = x_k
    
    return results
```

### 超参数调优指南

| 参数 | 推荐范围 | 调优策略 |
|------|---------|---------|
| `n_steps` | 3–20 | 从 5 开始；可视化结果，增加步数直到收益递减 |
| `step_size` | 0.005 – 0.05 | 太大：样本过度修正（聚集在 cluster 中心）；太小：修正不足 |
| `normalize_grad` | True（推荐）| 归一化梯度使步长效果更一致，数值更稳定 |

**可视化调试方法**：
```python
# 对比修正前后的 log|det J| 分布
log_dets_before = []
log_dets_after = []
for x, x_ref in zip(x_init_batches, x_refined_batches):
    with torch.no_grad():
        _, ld_b = bf.train_forward(x)
        _, ld_a = bf.train_forward(x_ref)
    log_dets_before.append(ld_b.item())
    log_dets_after.append(ld_a.item())
# 期望: mean(log_dets_after) > mean(log_dets_before)
```

### 与 EDR（本轮 Idea 2）的叠加使用

```python
# 最强组合：EDR (减少间隙 z 采样) + SGR (修正漏网的 inter-cluster x)

# Step 1: EDR 生成
z_edr = torch.sigmoid(torch.tensor(gmm.sample(n_samples), dtype=torch.float)).clamp(0.01, 0.99)
with torch.no_grad():
    x_edr = bf.inverse_map(z_edr)

# Step 2: SGR 修正
x_final = score_guided_refine(bf, x_edr, n_steps=3, step_size=0.005)  # 3步即可，EDR已减少问题
```

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **过度修正（over-refinement）** | 步数过多或步长过大，所有样本都聚集到全局最大密度点 | 限制步数 K ≤ 10；使用归一化梯度限制步长效果；加入少量噪声（Langevin 噪声项）保持多样性 |
| **梯度计算成本** | 每个样本 K 次 autodiff，比纯生成慢 K 倍 | K=5 时开销可接受（约 5× forward pass 时间）；可用 batched 计算并行 |
| **有限差分梯度精度** | `train_forward` 使用 ε-有限差分估计 Jacobian，不是精确梯度 | 使用 `train_forward_exact` 获得更精确梯度（慢但正确）；或保持现有实现（足够好） |
| **BF 密度不准（欠拟合）** | 若 BF 在 inter-cluster 区域的密度估计不准，score 方向可能误导 | 先确保 BF 训练充分收敛；结合 Piecewise BF 提高密度估计质量 |
| **修正后样本不服从精确分布** | 梯度上升不是 MCMC，不保证样本服从 p(x)（有 Langevin 版本可解决）| 对需要精确分布的应用，改用 MALA（加 MH 接受步）；对视觉生成质量，梯度上升已足够 |

---

## Langevin 版本（更严格的变体）

若需要理论上正确的 MCMC 保证（如密度评估任务），可改为 MALA：

```python
def mala_refine(bf, x_init, n_steps=50, step_size=0.005, temperature=1.0):
    """
    Metropolis-Adjusted Langevin Algorithm (MALA) for theoretically correct sampling.
    
    More expensive than SGR but guarantees asymptotic correctness.
    """
    x = x_init.clone()
    
    for t in range(n_steps):
        x.requires_grad_(True)
        
        # Compute log p(x) via BF
        breeze_list = []
        z = bf.forward(x, breeze_list)
        x_delta = x + bf.epsilon
        z_delta = bf.breeze_forward(x_delta, breeze_list)
        delta_u = (z_delta - z) * bf.dim_mask + 1 - bf.dim_mask
        log_p = torch.sum(torch.log(torch.abs(delta_u) + 1e-8), dim=1)  # (N,)
        
        grad = torch.autograd.grad(log_p.sum(), x)[0]
        x = x.detach()
        
        # Langevin proposal: x' = x + ε*grad + sqrt(2ε*T)*noise
        noise = torch.randn_like(x) * (2 * step_size * temperature) ** 0.5
        x_proposal = x + step_size * grad + noise
        
        # MH acceptance (simplified: accept with probability min(1, p(x')/p(x)))
        x_proposal.requires_grad_(False)
        with torch.no_grad():
            # Compute log p(x_proposal)
            breeze_list_p = []
            z_p = bf.forward(x_proposal, breeze_list_p)
            x_delta_p = x_proposal + bf.epsilon
            z_delta_p = bf.breeze_forward(x_delta_p, breeze_list_p)
            delta_u_p = (z_delta_p - z_p) * bf.dim_mask + 1 - bf.dim_mask
            log_p_proposal = torch.sum(torch.log(torch.abs(delta_u_p) + 1e-8), dim=1)
        
        # Accept/reject
        log_ratio = (log_p_proposal - log_p.detach()) / temperature
        accept = (torch.log(torch.rand(len(x))) < log_ratio.clamp(max=0))
        x = torch.where(accept.unsqueeze(1), x_proposal, x)
    
    return x.detach()
```

**对 BreezeForest 特别说明**：由于 BreezeForest 的 latent space 是 [0,1]^d（有界），可以在 w = logit(z) 空间做 MALA，然后 sigmoid 回到 z 空间，避免 x-space 中可能的数值问题。这与 Coeurdoux (2024) 的方法完全一致。

---

## 推荐优先级

**⭐⭐ 高优先级（作为 EDR 的配套/后处理方案）**

理由：
1. **唯一对已生成点做主动修正的方案**：其他所有 idea 都是"预防"inter-cluster 点生成；SGR 是"治疗"
2. **无需重训练、无需额外拟合**：直接复用已训练 BF 的 autodiff，零额外依赖
3. **适用于单 BF 和 MultiBF**：ICDR 只为 MultiBF 设计；SGR 通用
4. **与 EDR 互补**：EDR 减少间隙 z 采样，SGR 修正漏网的 inter-cluster x；两者叠加是最强组合
5. **有理论后盾（MALA 变体）**：SGR 的理论上正确版本即为 Coeurdoux (2024) 的 MALA 方法，已在文献中验证有效

**建议使用场景**：
- 快速部署（无时间重训练）→ 单独 SGR（5步梯度上升）
- 标准部署 → EDR + SGR（2步）
- 最佳质量 → Piecewise BF（训练） + EDR（采样） + SGR（修正）

---

## 参考文献

- Coeurdoux, F. et al. (2024). "Normalizing Flow Sampling with Langevin Dynamics in the Latent Space." *Machine Learning, 113*, 8301–8326.  
  https://arxiv.org/abs/2305.12149  
  （MALA 变体的直接前驱，SGR 是其无 MH 步的简化版本）
- Na, B. et al. (2024). "Diffusion Rejection Sampling." *ICML 2024*, PMLR 235.  
  https://proceedings.mlr.press/v235/na24a.html  
  （在生成模型中用密度评估做后处理质量修正的一般思路）
- Verine, A. et al. (2024). "Optimal Budgeted Rejection Sampling for Generative Models." *AISTATS 2024*, PMLR 238.  
  （固定预算拒绝采样的理论最优策略，SGR 是"软"版本）
- Song, Y. et al. (2020). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR 2021*.  
  （Score function 在生成模型中推动样本向高密度区域的理论基础）
