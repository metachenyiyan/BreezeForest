# Idea Cleanup (Oldest 10) — 2026-03-12

## Scope
- Directory: `notes/idea/`
- Total idea docs at cleanup time: **144**
- Rule: evaluate the **oldest 10** by filename timestamp, keep 3, delete 7.

## Evaluated (oldest 10)
1. `idea_hard_em_component_specialization_2026-03-11-1230.md`
2. `idea_latent_zone_restriction_2026-03-11-1235.md`
3. `idea_inter_component_density_repulsion_2026-03-11-1240.md`
4. `2026-03-11_1241_hard_em_cluster_conditional_training.md`
5. `2026-03-11_1242_empirical_latent_resampling.md`
6. `2026-03-11_1243_contrastive_void_penalty.md`
7. `idea_da_em_kmeans_init_2026-03-11-1250.md`
8. `idea_generative_consistency_filtering_2026-03-11-1251.md`
9. `idea_assignment_entropy_regularization_2026-03-11-1252.md`
10. `idea_kmeans_presplit_dedicated_training_2026-03-11-1300.md`

## Kept (3)
- `idea_da_em_kmeans_init_2026-03-11-1250.md`
  - Most complete training-side specialization strategy for current `MultiBF` (temperature annealing + pre-init), and aligns with known EM/annealing practice.
- `idea_assignment_entropy_regularization_2026-03-11-1252.md`
  - Low-overhead objective-side complement (responsibility sharpening + anti-collapse) with solid information-theoretic basis.
- `2026-03-11_1242_empirical_latent_resampling.md`
  - Practical inference-side mitigation (no retraining required) and aligned with resampled-base direction for reducing off-manifold/inter-cluster sampling.

## Deleted (7)
- `idea_hard_em_component_specialization_2026-03-11-1230.md`
  - Superseded by later, more stable DA-EM variant.
- `2026-03-11_1241_hard_em_cluster_conditional_training.md`
  - Same direction as 1230; redundant once DA-EM version is kept.
- `idea_latent_zone_restriction_2026-03-11-1235.md`
  - Axis-aligned latent box restriction is brittle and less robust than kept alternatives.
- `idea_inter_component_density_repulsion_2026-03-11-1240.md`
  - Extra training complexity and potential optimization conflict; weaker cost/benefit than AER.
- `2026-03-11_1243_contrastive_void_penalty.md`
  - Negative midpoint construction is data-geometry sensitive; less generally reliable.
- `idea_generative_consistency_filtering_2026-03-11-1251.md`
  - Useful but sampling-time cost is high and calibration assumptions are stronger than empirical latent resampling.
- `idea_kmeans_presplit_dedicated_training_2026-03-11-1300.md`
  - Heavy workflow shift and overlaps with many newer pre-cluster dedicated-training notes in the repo.
