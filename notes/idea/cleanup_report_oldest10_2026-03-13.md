# BreezeForest 历史 idea 清理报告（最老 10 篇）

- 日期：2026-03-13
- 范围：`notes/idea/` 中按文件名时间排序最老的 10 篇
- 评估依据：
  1. 当前代码实现（`BreezeForest` / `MultiBF` 的 flow、loss、inverse_map、mixture 责任度机制）
  2. 文档内容与当前架构适配性
  3. 外部检索（PNF、Resampled Base Flows、DAEM、FlowGMM/多模态 base、latent-MCMC 等）

## 纳入评估的最老 10 篇

1. `idea_responsibility_rejection_sampling_2026-03-11-1610.md`
2. `idea_cels_cluster_empirical_latent_sampling_2026-03-11-2009.md`
3. `idea_kmeans_independent_pretrain_2026-03-11-2011.md`
4. `idea_empirical_latent_histogram_sampling_2026-03-11-2128.md`
5. `idea_kmeans_epoch_hard_em_2026-03-11-2131.md`
6. `idea_latent_gmm_sampling_2026-03-11-2132.md`
7. `idea_responsibility_confidence_filter_2026-03-11-2133.md`
8. `idea_empirical_latent_gmm_base_distribution_2026-03-11-2153.md`
9. `idea_kde_training_density_rejection_sampling_2026-03-11-2155.md`
10. `idea_kmeans_warmstart_annealing_hardem_2026-03-11-2157.md`

## 最终保留（3）

1. `idea_kmeans_independent_pretrain_2026-03-11-2011.md`  
   - 理由：对当前核心矛盾（soft mixture 导致组件不专一、跨簇桥接）命中最直接；与 `MultiBF` 架构高度匹配；与 PNF 的外部证据一致；训练路径清晰、可落地。

2. `idea_cels_cluster_empirical_latent_sampling_2026-03-11-2009.md`  
   - 理由：针对当前 `inverse_map` 中 uniform latent 采样问题，给出后训练校准的低侵入方案；与 Resampled Base Distribution / 多模态 latent base 思路一致，且易于在现有代码上试验。

3. `idea_responsibility_rejection_sampling_2026-03-11-1610.md`  
   - 理由：直接利用现有 mixture posterior（责任度）做生成后过滤，不依赖重训练；对“cluster-between 无效点”有直接阻断作用，工程改造成本低。

## 删除（7）

1. `idea_empirical_latent_histogram_sampling_2026-03-11-2128.md`  
   - 删除原因：与 CELS 同方向但表达更冗长，直方图离散化在维度提升时退化明显，作为主保留价值低于 CELS。

2. `idea_kmeans_epoch_hard_em_2026-03-11-2131.md`  
   - 删除原因：与 `kmeans_independent_pretrain` 高度重叠；epoch-level hard EM 对训练稳定性和实现复杂度更敏感，性价比不如“先独立预训练再可选联合微调”。

3. `idea_latent_gmm_sampling_2026-03-11-2132.md`  
   - 删除原因：与 CELS / empirical-latent-gmm 基本重复，保留一份更完整版本即可，避免同类文档堆叠。

4. `idea_responsibility_confidence_filter_2026-03-11-2133.md`  
   - 删除原因：与 `responsibility_rejection_sampling` 核心机制重复（均为 posterior-based filtering），后者更直接、阈值语义更清晰。

5. `idea_empirical_latent_gmm_base_distribution_2026-03-11-2153.md`  
   - 删除原因：与 CELS / latent-gmm 高度重复，且实现细节更分散；保留 CELS 作为统一“经验 latent 重采样”入口更利于维护。

6. `idea_kde_training_density_rejection_sampling_2026-03-11-2155.md`  
   - 删除原因：data-space KDE 过滤在高维可扩展性较弱，且对带宽与阈值更敏感；相比责任度过滤，和当前模型内生概率结构耦合更弱。

7. `idea_kmeans_warmstart_annealing_hardem_2026-03-11-2157.md`  
   - 删除原因：与 `kmeans_independent_pretrain` 同属训练期分工路线但更复杂、超参更多、稳定性调参成本更高；在当前仓库阶段优先保留更稳健简洁方案。

