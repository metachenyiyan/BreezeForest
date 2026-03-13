# BreezeForest 历史 idea 清理报告（最老 10 篇）

- 日期：2026-03-13
- 范围：`notes/idea/` 中按文件名时间排序最老的 10 篇
- 评估依据：
  1. 当前代码实现（`BreezeForest` / `MultiBF` 的 flow、loss、inverse_map、mixture 机制）
  2. 10 篇文档与当前架构的适配性、重复度、可落地性
  3. 外部检索（Piecewise Normalizing Flows、Resampled Base Flows、DAEM 等）

## 纳入评估的最老 10 篇

1. `idea_responsibility_rejection_sampling_2026-03-11-1610.md`
2. `idea_gmm_latent_prior_sampling_2026-03-11-2250.md`
3. `idea_piecewise_breeze_forest_2026-03-11-2255.md`
4. `idea_pnf_style_preclustering_2026-03-11-2300.md`
5. `idea_lgmm_latent_gmm_calibration_2026-03-11-2305.md`
6. `idea_daem_deterministic_annealing_em_2026-03-11-2312.md`
7. `idea_kmeans_warm_start_init_2026-03-11-2314.md`
8. `idea_per_component_latent_kde_sampling_2026-03-11-2316.md`
9. `idea_deterministic_annealing_em_2026-03-11-2335.md`
10. `idea_per_component_latent_kde_sampling_2026-03-11-2340.md`

## 最终保留（3）

1. `idea_responsibility_rejection_sampling_2026-03-11-1610.md`  
   - 原因：可直接复用现有 `MultiBF` 概率结构做 posterior 过滤，无需重训练；对 cluster-between 无效点是最直接的推理阶段阻断。

2. `idea_pnf_style_preclustering_2026-03-11-2300.md`  
   - 原因：针对组件不专一这一训练期根因提出结构性方案（预聚类 + 独立训练）；与外部 PNF 证据一致，长期参考价值高。

3. `idea_lgmm_latent_gmm_calibration_2026-03-11-2305.md`  
   - 原因：直击当前 `inverse_map` 的 uniform latent 先验失配问题；属于低侵入后训练校准，且比盒形 zone 思路更精细。

## 删除（7）

1. `idea_gmm_latent_prior_sampling_2026-03-11-2250.md`  
   - 原因：与 `idea_lgmm_latent_gmm_calibration_2026-03-11-2305.md` 高度重复，后者描述更完整。

2. `idea_piecewise_breeze_forest_2026-03-11-2255.md`  
   - 原因：与 `idea_pnf_style_preclustering_2026-03-11-2300.md` 方向重叠，保留后者作为该方向代表即可。

3. `idea_daem_deterministic_annealing_em_2026-03-11-2312.md`  
   - 原因：与 `idea_deterministic_annealing_em_2026-03-11-2335.md` 同题重复；本轮优先保留训练结构性方案（PNF）与推理采样方案。

4. `idea_kmeans_warm_start_init_2026-03-11-2314.md`  
   - 原因：主要作为 DAEM/EM 路线配套初始化技巧；在仅保留 3 篇限制下，优先级低于根因方案与推理修复方案。

5. `idea_per_component_latent_kde_sampling_2026-03-11-2316.md`  
   - 原因：与 `...2340.md` 为近重复版本，且与 LGMM 方案功能重叠。

6. `idea_deterministic_annealing_em_2026-03-11-2335.md`  
   - 原因：与 2312 版本重复，且整体优先级低于已保留的 3 个方向组合。

7. `idea_per_component_latent_kde_sampling_2026-03-11-2340.md`  
   - 原因：与 2316 版本近重复；为减少噪声本轮整体移除 KDE 双版本。

