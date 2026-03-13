# BreezeForest 历史 idea 清理报告（最老 10 篇）

- 日期：2026-03-13
- 范围：`notes/idea/` 按文件名时间排序后的最老 10 篇
- 评估依据：
  1. 当前实现：`BreezeForest` / `MultiBF` 的结构、loss、采样与 inverse_map 逻辑
  2. 当前核心矛盾：组件专一化不足 + latent 全域均匀采样导致 cluster-between 无效生成
  3. 外部调研：PNF（arXiv:2305.02930）、Resampled Base Flows（AISTATS 2022）、latent-space MCMC（arXiv:2305.12149）与 DAEM 文献

## 本轮纳入评估（最老 10 篇）

1. `idea_responsibility_rejection_sampling_2026-03-11-1610.md`
2. `idea_piecewise_kmeans_independent_training_2026-03-12-0130.md`
3. `idea_gmm_latent_base_distribution_2026-03-12-0135.md`
4. `idea_latent_gmm_resampling_2026-03-12-0151.md`
5. `idea_latent_gmm_resampling_2026-03-12-0211.md`
6. `idea_piecewise_breeze_forest_kmeans_2026-03-12-0211.md`
7. `idea_responsibility_entropy_annealing_2026-03-12-0211.md`
8. `idea_kmeans_precluster_pnf_aligned_2026-03-12-0230.md`
9. `idea_gmm_latent_base_distribution_2026-03-12-0235.md`
10. `idea_temperature_annealed_assignment_2026-03-12-0240.md`

## 最终保留（3）

1. `idea_responsibility_rejection_sampling_2026-03-11-1610.md`
   - 原因：与当前 `MultiBF` 代码直接兼容（基于 `P(k|x)` 的后验过滤），无需改训练流程即可抑制明显的 cluster-between 误生成。

2. `idea_gmm_latent_base_distribution_2026-03-12-0135.md`
   - 原因：直击当前 uniform latent 采样失配问题；属于低侵入后训练校准方案，且有 Resampled Base Flows 思路支持。

3. `idea_kmeans_precluster_pnf_aligned_2026-03-12-0230.md`
   - 原因：与 PNF 路线一致，直接针对组件不专一根因（预聚类 + 专属训练），对当前 multi-cluster 任务具有长期参考价值。

## 删除（7）

1. `idea_piecewise_kmeans_independent_training_2026-03-12-0130.md`
2. `idea_latent_gmm_resampling_2026-03-12-0151.md`
3. `idea_latent_gmm_resampling_2026-03-12-0211.md`
4. `idea_piecewise_breeze_forest_kmeans_2026-03-12-0211.md`
5. `idea_responsibility_entropy_annealing_2026-03-12-0211.md`
6. `idea_gmm_latent_base_distribution_2026-03-12-0235.md`
7. `idea_temperature_annealed_assignment_2026-03-12-0240.md`

删除共性依据：
- 与保留文档高度重复（同方向多稿并存）
- 作为过渡版本，被更清晰版本覆盖
- 对当前代码路径的增量价值弱于保留稿件（或需要额外调参复杂度）

