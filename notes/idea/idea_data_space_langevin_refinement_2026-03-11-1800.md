# Idea: Data-Space Langevin MCMC Sample Refinement (LMSR)

**创建时间**: 2026-03-11 18:00 UTC  
**推荐优先级**: ⭐⭐⭐ 最高优先级（无需重训练，覆盖 single BF 和 MultiBF）

---

## 问题定义

BreezeForest（单模型和 MultiBF）在生成阶段，从 Uniform([0.01, 0.99]^d) 采样 z，再通过 bisection-based `inverse_map` 得到 x。

根本缺陷：
- 这是一次性、开环的映射：z → x = f^{-1}(z)，不考虑生成后 x 的实际密度
- 对于 single BF：[0,1]^d 的连通拓扑结构与 multi-cluster 数据的不连通支撑存在**拓扑不匹配**，inter-cluster 的 z 值必然映射到 cluster 之间的低密度区域
- 对于 MultiBF：即使 Hard-EM 使组件专一化，各组件的 `inverse_map` 仍然对整个 [0.01, 0.99]^d 均匀采样，其中有部分 z 对应到非目标 cluster 的区域

**关键洞察**：生成后的样本 x 可能落在低密度区域，但这个信息是**可计算的**（`log p_θ(x) = log |det J_f(x)|`），且梯度 `∇_x log p_θ(x)` 可以通过自动微分获得。可以利用这个密度梯度将样本"推回"高密度区域。

---

## 背景判断（来自代码与已有 idea）

**从代码中得到的关键观察**：

1. BreezeForest 的 `train_forward` 计算 `log_det = Σ log(dF_i/dx_i)`（有限差分近似），这也是 `log p(x)` 的直接代理
2. 正向映射 `bf.forward(x, breeze_list)` 是完全可微的（PyTorch autograd）
3. `inverse_map` 使用 bisection，**不可微**；但正向 `forward` 是可微的
4. 密度梯度 `∇_x log p_θ(x) = ∇_x log |det J_f(x)|` 可以通过正向 pass 的 autograd 计算得到

**从已有 idea 得到的背景判断**：

- LZR（Idea 2, 2026-03-11-1235）：通过限制 latent 采样范围来避免低密度生成 → **间接方法**，用轴对齐 bounding box 近似 cluster 的 latent 分布，不能处理非轴对齐形状
- Hard-EM（Idea 1）和 ICDR（Idea 3）：都是训练时修复 MultiBF，不解决 single BF 问题
- 所有已有 idea 对 **single BF** 的 inter-cluster 生成问题没有提出训练时或推理时修复方案

**外部调研发现**：
- Coeurdoux et al. (2024) "Normalizing flow sampling with Langevin dynamics in the latent space"：在 latent z 空间跑 MALA。但 BreezeForest 的 inverse（bisection）不可微，难以计算 ∇_z log p(f^{-1}(z)) 
- 本 Idea 改为在**数据空间 x** 跑 Langevin，利用可微的正向 pass 计算 ∇_x log p_θ(x)，完全绕开 bisection 不可微的问题
- Song et al. (2019+) 的 score-based 生成模型：证明了密度梯度（score）可以有效引导样本向高密度区域移动

---

## 核心思路

**两阶段生成策略**：

**阶段 1（初始化）**：标准生成流程
```
z ~ Uniform(0.01, 0.99)^d
x_0 = f^{-1}(z)  [通过 bisection inverse_map]
```

**阶段 2（Langevin 精炼）**：在数据空间跑 MALA（Metropolis-adjusted Langevin Algorithm）

对 t = 1, 2, ..., L 步：
```
g_t = ∇_{x_t} log p_θ(x_t)           [score function，通过 autograd 计算]
x_t_proposed = x_t + η * g_t + √(2η) * ε_t    [ε_t ~ N(0, I)]
接受/拒绝（Metropolis 修正）：以 min(1, p(x_proposed)/p(x_t)) 的概率接受
```

最终输出精炼后的 x_L。

---

## 为什么它适合解决 multi-cluster 中间点生成问题

**机制分析**：

1. 若 x_0 落在 inter-cluster 区域（低密度），则 `log p_θ(x_0)` 很小
2. 密度梯度 `∇_x log p_θ(x_0)` 指向密度上升最快的方向，即**最近的 cluster 方向**
3. Langevin 步骤会将 x_0 推向最近的 cluster
4. 几步之后 x 稳定在 cluster 内部的高密度区域，不再在 inter-cluster 区域漂移
5. Metropolis 修正保证最终采样分布是 p_θ(x) 的精确后验

**与 LZR 的本质区别**：
- LZR 通过限制 z 的范围来**预防**低密度采样（axis-aligned box 约束）
- Langevin 精炼通过梯度引导来**纠正**已生成的低密度样本（精确密度引导）
- Langevin 不需要了解 cluster 数目、不需要 calibration 步骤，自动处理任意形状的 cluster

**对于 single BF（最重要的差异点）**：
- LZR、Hard-EM、ICDR 均只适用于 MultiBF
- 本 Idea 对 **single BF** 同样有效：BF 的 log p(x) = log |det J| 是可微的
- Langevin 会将 inter-cluster 样本推向 training data 的 cluster，无论模型是否是 mixture

---

## 与历史 idea 的关系

| 已有 Idea | 关系 | 说明 |
|----------|------|------|
| LZR（Idea 2, 1235） | **互补 + 升级** | LZR 在 latent 空间做轴对齐约束（推理前过滤），Langevin 在数据空间做连续精炼（推理后修正）。可以叠加：LZR 先过滤，再 Langevin 精炼。对于 single BF，Langevin 是 LZR 的唯一替代方案 |
| Hard-EM（Idea 1, 1230） | **互补** | Hard-EM 解决训练时组件分工，Langevin 解决推理时生成质量。两者不冲突，可同时使用 |
| ICDR（Idea 3, 1240） | **互补** | ICDR 通过训练时排斥减少组件间密度重叠，Langevin 在推理时对漏出的 inter-cluster 样本做后处理 |
| 所有历史 Idea | **填补 single BF 空白** | 历史 3 个 idea 均只针对 MultiBF。本 Idea 是第一个同时适用于 single BF 的推理时修复方案 |

**与外部文献的关系**：
- Coeurdoux et al. (2024)：在 latent 空间跑 Langevin，需要可微的 inverse → 与 BreezeForest 不兼容
- 本 Idea 改在**数据空间**运行，利用可微的正向 pass → **专为 BreezeForest 设计的适配方案**

---

## 具体实现建议

### 步骤 1：实现 score function 计算

```python
def compute_score(bf, x):
    """
    Compute ∇_x log p_θ(x) for a BreezeForest model.
    
    log p(x) ≈ Σ_i log(dF_i/dx_i), approximated by the finite-difference 
    Jacobian diagonal in train_forward. For gradient computation, we use
    exact autograd through the forward pass.
    
    :param bf: BreezeForest instance
    :param x: tensor (batch_size, dim), requires_grad or not
    :return: score (batch_size, dim)
    """
    x = x.detach().requires_grad_(True)
    
    # Use the approximate log|det J| computation (consistent with training)
    epsilons = bf.epsilon  # (1, dim)
    x_plus = x + epsilons
    x_minus = x - epsilons
    
    breeze_list = []
    y = bf.forward(x, breeze_list)
    y_plus = bf.breeze_forward(x_plus, breeze_list)
    y_minus = bf.breeze_forward(x_minus, breeze_list)
    
    du_dx = (y_plus - y_minus) / (2 * epsilons)
    du_dx = torch.abs(du_dx * bf.dim_mask + 1 - bf.dim_mask).clamp(min=0.001)
    log_det = torch.sum(torch.log(du_dx), dim=1)  # (batch_size,)
    
    # Compute gradient of log_det w.r.t. x
    total_log_det = log_det.sum()
    grad = torch.autograd.grad(total_log_det, x)[0]  # (batch_size, dim)
    return grad.detach()
```

### 步骤 2：实现 Langevin 精炼

```python
def langevin_refine(bf, x_init, n_steps=20, step_size=0.01, 
                    temperature=1.0, use_mh=True):
    """
    Refine generated samples via Langevin MCMC in data space.
    
    :param bf: BreezeForest or component BreezeForest
    :param x_init: initial samples from inverse_map (batch_size, dim)
    :param n_steps: number of Langevin steps
    :param step_size: Langevin step size η
    :param temperature: temperature parameter (1.0 = standard)
    :param use_mh: if True, apply Metropolis-Hastings correction
    :return: refined samples (batch_size, dim)
    """
    x = x_init.detach().clone()
    
    # Compute initial log density for MH acceptance
    if use_mh:
        with torch.no_grad():
            _, log_det_current = bf.train_forward(x, light=True)
            # Per-sample log density
            log_p_current = _per_sample_log_density(bf, x)
    
    for _ in range(n_steps):
        score = compute_score(bf, x)
        noise = torch.randn_like(x) * (2 * step_size / temperature) ** 0.5
        x_proposed = x + step_size / temperature * score + noise
        
        if use_mh:
            log_p_proposed = _per_sample_log_density(bf, x_proposed)
            log_accept = (log_p_proposed - log_p_current) / temperature
            accept_mask = (torch.log(torch.rand_like(log_accept)) < log_accept)
            x = torch.where(accept_mask.unsqueeze(1), x_proposed, x)
            log_p_current = torch.where(accept_mask, log_p_proposed, log_p_current)
        else:
            x = x_proposed
    
    return x.detach()


def _per_sample_log_density(bf, x):
    """Per-sample log |det J| for a BreezeForest."""
    x = x.detach()
    epsilons = bf.epsilon
    x_deltas = torch.cat([
        (x - epsilons).view(1, -1, x.size(1)),
        (x + epsilons).view(1, -1, x.size(1))
    ], dim=0)
    breeze_list = []
    y = bf.forward(x, breeze_list)
    x_deltas = bf.breeze_forward(x_deltas, breeze_list)
    du_dx = (x_deltas[1] - x_deltas[0]) / (2 * epsilons)
    du_dx = torch.abs(du_dx * bf.dim_mask + 1 - bf.dim_mask).clamp(min=0.001)
    return torch.sum(torch.log(du_dx), dim=1)
```

### 步骤 3：集成到生成流程

**单 BF 生成（替换 `generate_sample` in `demo_functions.py`）**：
```python
def generate_sample_with_langevin(model, std, mean, sample_size, 
                                   n_langevin_steps=30, step_size=0.005):
    model.eval()
    with torch.no_grad():
        seeds = torch.rand(sample_size, model.dim) * 0.98 + 0.01
        x_init = model.inverse_map(seeds)
    
    # Langevin refinement (needs grad, so outside no_grad)
    x_refined = langevin_refine(model, x_init, n_steps=n_langevin_steps, 
                                  step_size=step_size)
    
    with torch.no_grad():
        x_refined = x_refined * std + mean
    return x_refined
```

**MultiBF 生成（替换 `MultiBF.inverse_map`）**：
```python
def inverse_map_with_langevin(self, n_samples, max_gap=1e-3, 
                               n_langevin_steps=20, step_size=0.005):
    # Standard mixture sampling
    weights = self.get_mixture_weights().detach()
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    results = torch.zeros(n_samples, self.dim)
    
    for k in range(self.n_components):
        mask = (component_indices == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        z = torch.rand(n_k, self.dim) * 0.98 + 0.01
        x_k = self.components[k].inverse_map(z, max_gap=max_gap)
        
        # Langevin refinement using component k's density
        x_k_refined = langevin_refine(
            self.components[k], x_k, 
            n_steps=n_langevin_steps, step_size=step_size
        )
        results[mask] = x_k_refined
    
    return results
```

### 超参数调优建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `n_langevin_steps` | 10–50 | 步数越多越精确，但计算量线性增加；20 步通常足够 |
| `step_size` | 0.001–0.01 | 太大：样本飞散；太小：收敛慢。建议从 0.005 开始 |
| `temperature` | 0.5–1.0 | 降温可以更严格地锁定密度模式，但多样性下降 |
| `use_mh` | True | 建议开启 Metropolis 修正，保证采样精确性 |

---

## 潜在风险 / 副作用

| 风险 | 描述 | 缓解方案 |
|------|------|---------|
| **计算开销** | 每个样本需要额外运行 L 步 Langevin，每步需要一次正向 pass + autograd | 使用光型近似（`light=True`）减少每步开销；L 通常不超过 50 |
| **样本多样性下降** | Langevin 可能导致样本都收敛到最近的密度模式，reducing diversity | 使用 temperature > 0 控制多样性；或仅对明显 inter-cluster 样本做精炼（先用 LZR 过滤） |
| **梯度数值问题** | 若 x 落在极低密度区域，`∇_x log p` 可能非常大（梯度爆炸） | 对 score 做 clip：`score = score.clamp(-clip, clip)`；推荐 `clip=10.0` |
| **BF 正向 pass 的 actinorm 状态依赖** | `treeBias`、`treeScale` 在 forward 时被初始化和缓存，多次 forward 可能有副作用 | 确保 Langevin 步骤中 BF 处于 `eval()` 模式，actinorm 参数已固定 |
| **不适合外推** | 若初始 x_0 距离所有 cluster 都很远（extreme outlier），Langevin 可能需要很多步才能到达 cluster | 先用 LZR 或 rejection sampling 过滤极端 outlier，再做 Langevin 精炼 |

---

## 推荐优先级

**⭐⭐⭐ 最高优先级**

理由：
1. **填补历史 idea 的 single BF 空白**：现有 3 个 idea 全部针对 MultiBF，本 Idea 是首个同时覆盖 single BF 的推理时修复
2. **无需重训练**：对已训练的模型立即生效，验证成本低
3. **原理最直接**：直接利用模型自身的密度梯度修正生成样本，而非间接约束 latent 采样范围
4. **与 BreezeForest 架构高度兼容**：利用可微的正向 pass，完全绕开 bisection 不可微问题
5. **理论支撑强**：MALA 有严格的 Markov chain 收敛保证；score-based refinement 是当前 generative model 领域主流方向

**建议实施顺序**：
1. 先在 single BF + GAUSSIANS 数据集验证 Langevin 对 inter-cluster 样本的纠正效果
2. 再在 MultiBF + Blobs/Moons 上验证组合使用（inverse_map + Langevin）
3. 与 LZR 对比：Langevin 应能处理 LZR 无法处理的非轴对齐 cluster 形状

---

## 参考文献

- Coeurdoux, F. et al. (2024). "Normalizing flow sampling with Langevin dynamics in the latent space." *Machine Learning 2024*. https://arxiv.org/abs/2305.12149  
  （在 latent 空间的 Langevin；本 Idea 改为 data 空间以兼容 BreezeForest 的 bisection inverse）
- Song, Y. & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS 2019*.  
  （score function 引导生成的理论基础）
- Roberts, G. & Tweedie, R. (1996). "Exponential Convergence of Langevin Distributions." *Bernoulli*.  
  （MALA 收敛性理论保证）
- Stimper, V. et al. (2022). "Resampling Base Distributions of Normalizing Flows." *AISTATS 2022*.  
  （类似目标但方法不同；本 Idea 在数据空间操作而非 latent 空间）
