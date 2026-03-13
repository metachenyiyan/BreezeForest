## Historical idea cleanup (2026-03-13)

### Scope
- Directory: `notes/idea/`
- Total idea docs before cleanup: 141
- Evaluated set: the oldest 10 files by filename timestamp

### Oldest 10 evaluated files
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

### Kept (3)
1. `idea_da_em_kmeans_init_2026-03-11-1250.md`
2. `idea_kmeans_presplit_dedicated_training_2026-03-11-1300.md`
3. `idea_gmm_latent_prior_2026-03-11-1310.md`

### Deleted (7)
1. `idea_hard_em_component_specialization_2026-03-11-1230.md`
2. `idea_latent_zone_restriction_2026-03-11-1235.md`
3. `idea_inter_component_density_repulsion_2026-03-11-1240.md`
4. `idea_generative_consistency_filtering_2026-03-11-1251.md`
5. `idea_assignment_entropy_regularization_2026-03-11-1252.md`
6. `idea_kmeans_piecewise_training_2026-03-11-1320.md`
7. `idea_temperature_annealed_responsibility_2026-03-11-1320.md`

### Core rationale
- Priority was given to proposals with the highest current value for BreezeForest's multi-cluster invalid-generation issue, strongest fit with the current MultiBF architecture, and clearest external support.
- Kept set covers:
  - stable training-time component specialization (`DA-EM + KMeans init`),
  - topology-aware cluster-wise modeling (`KMeans pre-split dedicated training`),
  - low-density sampling reduction without retraining (`latent GMM prior`).
- Deleted items were mainly superseded variants, highly overlapping alternatives, or less robust/less direct paths relative to the kept set.
