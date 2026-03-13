# BreezeForest 历史 idea 清理报告（最老 10 篇）

- 日期：2026-03-13
- 范围：`notes/idea/` 按文件名时间排序后的最老 10 篇
- 评估依据：
  1. 当前代码实现：`BreezeForest` / `MultiBF` 的结构、loss、采样与 inverse_map 流程
  2. multi-cluster 核心矛盾匹配度：组件专一化不足 + latent 全域采样导致 cluster-between 无效生成
  3. 外部调研：PNF（arXiv:2305.02930）、Resampled Base Flows（AISTATS 2022）、latent Langevin/MALA（2024）等

## 本轮纳入评估（最老 10 篇）

1. `idea_responsibility_rejection_sampling_2026-03-11-1610.md`
2. `idea_pnf_style_preclustering_2026-03-11-2300.md`
3. `idea_lgmm_latent_gmm_calibration_2026-03-11-2305.md`
4. `idea_kmeans_init_hard_em_2026-03-12-0030.md`
5. `idea_annealed_responsibility_temperature_2026-03-12-0032.md`
6. `idea_lzr_kmeans_purified_zone_estimation_2026-03-12-0034.md`
7. `idea_kmeans_preassign_dedicated_training_2026-03-12-0055.md`
8. `idea_mvn_latent_base_distribution_2026-03-12-0100.md`
9. `idea_mala_latent_space_sampling_2026-03-12-0105.md`
10. `idea_gmm_latent_base_distribution_2026-03-12-0119.md`

## 最终保留（3）

1. `idea_responsibility_rejection_sampling_2026-03-11-1610.md`
   - 原因：无需重训练，直接基于 `P(k|x)` 的后验过滤生成样本，和当前 `MultiBF` API 高度兼容，能直接抑制 cluster-between 无效点。

2. `idea_lgmm_latent_gmm_calibration_2026-03-11-2305.md`
   - 原因：针对 `inverse_map` 里 uniform latent 采样失配问题，提供低侵入后训练校准（logit-space GMM），比矩形 zone / 单高斯更能刻画 latent 结构。

3. `idea_kmeans_preassign_dedicated_training_2026-03-12-0055.md`
   - 原因：直接针对“组件不专一”根因（预聚类 + 专属训练），与 PNF 外部证据一致，且作为训练阶段方案具有长期参考价值。

## 删除（7）

1. `idea_pnf_style_preclustering_2026-03-11-2300.md`
2. `idea_kmeans_init_hard_em_2026-03-12-0030.md`
3. `idea_annealed_responsibility_temperature_2026-03-12-0032.md`
4. `idea_lzr_kmeans_purified_zone_estimation_2026-03-12-0034.md`
5. `idea_mvn_latent_base_distribution_2026-03-12-0100.md`
6. `idea_mala_latent_space_sampling_2026-03-12-0105.md`
7. `idea_gmm_latent_base_distribution_2026-03-12-0119.md`

删除共性依据：与保留文档方向高度重复、属于过渡版本（被更完整版本覆盖），或在当前代码路径下实现复杂度明显更高而增量价值有限，保留会继续放大历史噪声。

