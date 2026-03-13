# BreezeForest 历史 idea 清理报告（最老 10 篇）

- 日期：2026-03-13
- 目录：`notes/idea/`
- 排序规则：按文件名中的时间信息升序；同时间戳按文件名字典序
- 评估依据：
  1. 当前代码实现：`BreezeForest` / `MultiBF` 的模型结构、loss、inverse_map 与采样流程
  2. 当前核心矛盾：组件专一化不足 + latent 全域均匀采样，导致 cluster-between 无效生成
  3. 外部调研：PNF（arXiv:2305.02930）、Resampled Base Flows（AISTATS 2022）、latent-space MCMC（Coeurdoux 2024）、DAEM/annealing 相关文献

## 本轮纳入评估（最老 10 篇）

1. `idea_responsibility_rejection_sampling_2026-03-11-1610.md`
2. `idea_gmm_latent_base_distribution_2026-03-12-0135.md`
3. `idea_kmeans_precluster_pnf_aligned_2026-03-12-0230.md`
4. `idea_interpolated_boundary_negative_density_loss_2026-03-12-0310.md`
5. `idea_self_density_rejection_sampling_2026-03-12-0310.md`
6. `idea_topology_aware_preclustering_hdbscan_2026-03-12-0310.md`
7. `idea_ess_adaptive_daem_2026-03-12-0315.md`
8. `idea_inter_cluster_anti_density_training_2026-03-12-0315.md`
9. `idea_latent_mh_sampling_2026-03-12-0315.md`
10. `idea_ea_daem_entropy_augmented_2026-03-12-0332.md`

## 最终保留（3）

1. `idea_responsibility_rejection_sampling_2026-03-11-1610.md`
   - 原因：与当前 `MultiBF` 现有 API（`_per_sample_log_det`、`get_mixture_weights`）高度贴合，无需改训练即可落地后验过滤；对 inter-cluster 错误生成有直接抑制作用。

2. `idea_gmm_latent_base_distribution_2026-03-12-0135.md`
   - 原因：直接针对当前 `z ~ Uniform([0.01,0.99]^d)` 的采样失配问题，属于低侵入后训练校准路线；与 Resampled Base Distribution 文献方向一致，工程可行性高。

3. `idea_kmeans_precluster_pnf_aligned_2026-03-12-0230.md`
   - 原因：直接命中组件不专一这一训练根因；与 PNF 的“先分簇再分片训练”思路一致，且对当前 `MultiBF` 架构最有长期参考价值。

## 删除（7）

1. `idea_interpolated_boundary_negative_density_loss_2026-03-12-0310.md`
2. `idea_self_density_rejection_sampling_2026-03-12-0310.md`
3. `idea_topology_aware_preclustering_hdbscan_2026-03-12-0310.md`
4. `idea_ess_adaptive_daem_2026-03-12-0315.md`
5. `idea_inter_cluster_anti_density_training_2026-03-12-0315.md`
6. `idea_latent_mh_sampling_2026-03-12-0315.md`
7. `idea_ea_daem_entropy_augmented_2026-03-12-0332.md`

删除共性依据（摘要）：
- 与保留方向重复或可被更简洁方案覆盖（例如两份插值反密度训练文档高度重叠）
- 依赖当前仓库尚未落地的训练分支（DAEM 系列升级文档）
- 工程成本/推理代价明显偏高（如 latent MCMC）且短期落地价值低于保留候选
- 对当前核心矛盾的“投入产出比”不如保留的 3 篇

