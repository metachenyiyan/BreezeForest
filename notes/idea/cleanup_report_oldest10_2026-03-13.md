# BreezeForest 历史 idea 清理报告（最老 10 篇）

- 日期：2026-03-13
- 范围：`notes/idea/` 按文件名时间排序后的最老 10 篇
- 评估依据：
  1. 当前实现：`BreezeForest` / `MultiBF` 的 flow、mixture、loss、inverse_map
  2. 文档与当前 multi-cluster 核心问题（组件不专一 + 低密度区域采样）的匹配度
  3. 外部检索：PNF（arXiv:2305.02930）、Resampled Base Flows（AISTATS 2022）、latent-space MCMC 等

## 本轮纳入评估（最老 10 篇）

1. `idea_responsibility_rejection_sampling_2026-03-11-1610.md`
2. `idea_pnf_style_preclustering_2026-03-11-2300.md`
3. `idea_lgmm_latent_gmm_calibration_2026-03-11-2305.md`
4. `idea_kmeans_sequential_component_pretraining_2026-03-11-2345.md`
5. `idea_precluster_hardEM_2026-03-11-2351.md`
6. `idea_gmm_latent_base_distribution_2026-03-11-2352.md`
7. `idea_icdr_v2_confirmed_2026-03-11-2353.md`
8. `idea_kmeans_preclustering_dedicated_training_2026-03-12-0015.md`
9. `idea_empirical_latent_distribution_sampling_2026-03-12-0020.md`
10. `idea_icdr_v2_with_annealing_schedule_2026-03-12-0025.md`

## 最终保留（3）

1. `idea_responsibility_rejection_sampling_2026-03-11-1610.md`  
   - 原因：无需重训练，直接基于 `P(k|x)` 后验过滤生成样本，和现有 `MultiBF` API 高度匹配。

2. `idea_pnf_style_preclustering_2026-03-11-2300.md`  
   - 原因：针对“组件不专一”根因，且与 PNF 外部证据一致，是训练阶段最有代表性的结构方案。

3. `idea_lgmm_latent_gmm_calibration_2026-03-11-2305.md`  
   - 原因：针对 `inverse_map` 的 uniform latent 失配，提供低侵入后训练校准，且较盒形 zone 方案更精细。

## 删除（7）

1. `idea_kmeans_sequential_component_pretraining_2026-03-11-2345.md`
2. `idea_precluster_hardEM_2026-03-11-2351.md`
3. `idea_gmm_latent_base_distribution_2026-03-11-2352.md`
4. `idea_icdr_v2_confirmed_2026-03-11-2353.md`
5. `idea_kmeans_preclustering_dedicated_training_2026-03-12-0015.md`
6. `idea_empirical_latent_distribution_sampling_2026-03-12-0020.md`
7. `idea_icdr_v2_with_annealing_schedule_2026-03-12-0025.md`

删除共性依据：与保留方案高度重复、为同一路线的派生/改写版本，或对当前代码路径侵入更大但增量价值不明显，保留后会增加历史噪声。

