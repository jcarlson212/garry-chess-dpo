Experiment 2 paper plot story
=============================

Main-paper figure families:
1. fig_tau_sweeps.pdf/png
   - Training-tau sweep for MRR / Recall@1 / hard-gap.
   - Clarifies retrieval quality as training tau changes.

2. fig_phi_comparisons.pdf/png
   - Metadata story: phi1 vs phi0 at matched family budget.
   - Read as: phi1 should help most on retrieval and hard-negative-aware metrics.

3. fig_finalist_main_results.pdf/png
   - Finalist overall story with automatic fallback when metrics are missing.
   - In this dataset, mean_logp_gap/KL are largely unavailable, so available metrics are shown instead.

4. fig_hard_negative_results.pdf/png
   - Application story: v3 > v2 > v1 should show up most clearly here.
   - Read as: hard-negative-aware metrics are main decision metrics.

5. fig_pair_score_components.pdf/png
   - Score-level diagnostics: positive vs hardest-negative vs soft-negative cosine means and resulting gaps.
   - Useful for diagnosing cases where hard gap goes negative but other metrics seem stable.

6. fig_classification_summary.pdf/png
   - Classification summary: ROC AUC, AP, best F1, threshold.

7. fig_eval_tau_sensitivity.pdf/png
   - Eval-time tau sensitivity (hard-gap and InfoNCE-like loss from pair metrics).
   - This is not retrieval recall-vs-threshold; raw threshold curves are not emitted by current eval artifacts.

8. fig_super_vs_final.pdf/png
   - Stage handling overview for final vs super runs (including missing-stage cases).

9. fig_training_diagnostics_finalists.pdf/png
   - Stability story: finalists should train smoothly enough without obvious collapse.

10. fig_test_player_pca.pdf/png
   - PCA(2) of per-player test centroids for each finalist model.
   - Samples up to 100 players with bias toward high-coverage/famous labeled players; centroid average is stratified across phase and game buckets.

11. fig_promotion_scatter.pdf/png
   - Process story: screening metrics should roughly predict final metrics, justifying promotion logic.

Metric definitions:
- Recall@1: fraction of anchors where the true positive is ranked #1 among candidates.
- MRR: mean reciprocal rank of the true positive; 1.0 is perfect, higher is better.
- Pair acc vs hardest neg: probability positive score > hardest negative score.
- Hard gap: mean(pos - hardest_neg) in cosine space.

Ablations:
- fig_ablation_batchsize.pdf/png
- fig_ablation_lr.pdf/png

Appendix-only figures:
- appendix_conditionals.pdf/png
- appendix_ranked_metrics.pdf/png

Deliberately omitted from main paper:
- AP / F1 / ROC AUC
- giant spaghetti loss overlays
- raw histogram-heavy spread plots
- single highlighted PCA panel

Naming fixes:
- tau0_10 and tau0_1 are canonicalized to tau=0.1
- run discovery is split-aware and does not assume only *_val.json

Coverage diagnostics:
- See tables/metric_coverage.csv for which metrics are actually present.
- See main_paper/super_vs_final.csv for stage availability and comparisons.
- See tables/famous_players_in_test_split.csv for famous players found per run.
- See tables/test_player_pca_samples.csv for sampled players used in PCA plots.