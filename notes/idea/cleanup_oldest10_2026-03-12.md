# Idea Cleanup Record (Oldest 10) — 2026-03-12

本记录仅用于说明本轮历史 idea 清理依据。

## 评估范围（按文件名时间排序最老 10 个）

1. `idea_hard_em_component_specialization_2026-03-11-1230.md`
2. `idea_latent_zone_restriction_2026-03-11-1235.md`
3. `idea_inter_component_density_repulsion_2026-03-11-1240.md`
4. `idea_da_em_kmeans_init_2026-03-11-1250.md`
5. `idea_generative_consistency_filtering_2026-03-11-1251.md`
6. `idea_assignment_entropy_regularization_2026-03-11-1252.md`
7. `idea_kmeans_presplit_dedicated_training_2026-03-11-1300.md`
8. `idea_gmm_latent_prior_2026-03-11-1310.md`
9. `idea_kmeans_piecewise_training_2026-03-11-1320.md`
10. `idea_temperature_annealed_responsibility_2026-03-11-1320.md`

## 保留（3 个）

- `idea_da_em_kmeans_init_2026-03-11-1250.md`  
  保留原因：与当前 `MultiBF` soft-assignment 训练机制高度匹配，提供从 soft 到近 hard 的连续过渡，且有确定性退火 EM 文献支撑。

- `idea_gmm_latent_prior_2026-03-11-1310.md`  
  保留原因：直接作用于当前生成流程（`inverse_map` 之前的 latent 采样），针对多簇低密度区误采样问题，和 flow base-distribution 改进方向一致。

- `idea_kmeans_piecewise_training_2026-03-11-1320.md`  
  保留原因：对应“多簇先分片再独立训练 flow”的强基线，和当前代码结构兼容（无需重构模型层），并有 piecewise flow 方向外部经验支持。

## 删除（7 个）

- `idea_hard_em_component_specialization_2026-03-11-1230.md`
- `idea_latent_zone_restriction_2026-03-11-1235.md`
- `idea_inter_component_density_repulsion_2026-03-11-1240.md`
- `idea_generative_consistency_filtering_2026-03-11-1251.md`
- `idea_assignment_entropy_regularization_2026-03-11-1252.md`
- `idea_kmeans_presplit_dedicated_training_2026-03-11-1300.md`
- `idea_temperature_annealed_responsibility_2026-03-11-1320.md`

## 删除核心依据（汇总）

- 与保留文档存在明显方向重叠，且被更完整或更可落地版本覆盖（如 Hard-EM / LZR / 早期 KMeans pre-split 版本）。
- 对当前架构的增量价值较低，或实现复杂度与预期收益比不如保留方案。
- 部分方案依赖额外超参数/流程，实践稳定性与可复现性不如保留方案。
