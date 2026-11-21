# DTB – Decode the Bioreactor (Team 23): models and experiments

All scores are **cross-validated MAPE (%)** on the 49 training batches (5 batch-level folds, seed 0) unless stated otherwise.

## Model versions

| Version | Description | VCD | TCD | Substrate | Byproduct | Mean |
|---|---|---|---|---|---|---|
| v1 | 0.7 x per-day ridge (+-1 day, half weight) + 0.3 x gradient boosting; inputs: day-0 state, volume, cumulative feeds C_k(t) and counts | 7.13 | 7.05 | 9.40 | 10.42 | 8.50 |
| v3 | average of 12 ridge settings + per-phase boosting (days 1-4 / 5-9 / 10-13, +-1 day margin) | 6.95 | 6.89 | 9.13 | 10.15 | 8.28 |
| **v4** | per-target blend of v3, similar-batch kNN (k=5/8), ridge with upcoming-feed inputs, ridge with upcoming feed_substrate inputs, per-phase boosting with feed_c windows; TCD floored at VCD x minimum dead share per day | 6.82 | 6.63 | 7.88 | 9.86 | **7.80** (honest nested) |

Leaderboard: v1 submission scored 8.64 (CV 8.50). Expected v4 ~7.9-8.0.

## Files

| File | Purpose |
|---|---|
| dtb_regression.py | shared pipeline: features (day-0 state, volume, C_k(t) = sum of feed_k with elapsed_time <= t), folds, ridge/boosting CV |
| train_v3.py | trains v3 on all 49 batches -> v3_models.joblib (~20 s; not committed, 24 MB) |
| train_v4.py | trains v4 (needs v3_models.joblib) -> v4_models.joblib incl. test predictions |
| final_blend_models.joblib, bioreaction_submission.csv | v1 models and the v1 submission (leaderboard 8.64) |
| exp_*.py and *_results/ | all experiments with held-out predictions |

Rebuild: python train_v3.py && python train_v4.py  (paths assume the Renku session layout /home/renku/work).

## Experiments (what helped / what did not)

- **Helped:** borrowing neighbouring days (+-1); averaging many settings; per-phase boosting; upcoming feed decisions as soft sensors (substrate 9.1 -> 7.9); similar-batch kNN (byproduct).
- **No gain:** recalibration / range-based bias correction; single-setting tuning (overfits); chained per-feed models; lagged feed inputs (1-3 days); two-stage substrate->byproduct; cell-density->byproduct functions (ridge, boosting, MLP; ceiling ~11.6 % even with true cells); kinetic rate law; residual correction from feed_c (0 % of residual variance explained out of sample).
- **Checked:** cumulative feeds C_k(t) are used directly and match the definition exactly (max difference ~1e-15).

## Process insights

- Byproduct responds to VCD x substrate with a ~2-day delay. Fitted rate law (2-day lag): dP/dt = 0.856 + 0.0118 XS(t-2) - 0.102 P - 0.0095 XS(t-2) P - 0.111 t.
- Day 4-5 is the switch point: byproduct peaks, substrate/byproduct is minimal, dead/viable starts rising (~1 % -> ~11 % by day 13).
- feed_substrate is a substrate top-up (bigger when substrate is low); feed_b supplies ~2/3 of added substrate and is dosed per live cell; feed_d is a fixed recipe; feed_c is on-demand and tracks byproduct consumption (its identity is unknown - worth asking J&J).
- Remaining byproduct error (~10 %) is not explained by feeds, substrate, cells or their histories.
