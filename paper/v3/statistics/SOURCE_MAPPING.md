# Final v3 statistical summaries: source keys and reading guide

All CSV rate columns use fractions in [0,1]; `_pp` columns are percentage points. Missing precision means no abstentions and is not zero. These compact files summarize supplied data; they contain no raw sample dataset.

The 2026-10-04 revision starts from `KDM_Paper_v3_20261003.zip` and adds the accepted Gemma native-SID result and saved closeout analyses. The inherited statistical source is `KDM_Nine_Complete_Review_20261003_final_v2.zip`; the historical artwork source was `KDM_v3_Final_20261003.zip`.

## Headline tables
- `food_nine_models_best_observed_J.csv`: 69 frozen rows drawn from `main/metrics_all_complete.csv`, plus one accepted Gemma SID row (70 total). Native/control methods retain one registered row; IP-VCD/IP-M3ID select maximum J among each model–method's four registered Food eval rows. `source_row` is 1-based CSV line including header. Gemma's pipe-delimited condition key explicitly contains model, dataset, split, method, kind, main/reference marker, replicate and implementation revision; its original input identities remain in `supplementary/gemma_sid/sources.csv`. All metrics come from one selected row. Food IP selection is descriptive eval-best observation; development-selected points are in `supplementary/dev_selection20.csv`, `supplementary/dev_selected_eval20.csv` and the corresponding source CSVs.
- `food_five_representative_models.csv`: requested five-model subset of the preceding table, without changing rows or selection.
- `vizwiz_nine_models_recorded_working_points.csv`: 68 rows: the frozen45 plus23 native baseline rows in `addendum/viz_native_baselines/metrics_actual.csv`, identified by `source_row` and the model/method/kind/marker/reference/guided tuple. Only one main IP point is recorded per model–method; no within-task sweep or validated Food-dev transfer is asserted. Primary `answer_quality_mean` is annotated answer-text consensus credit with semantic abstentions zero; `official_raw_score_mean` separately scores the original full reply.
- `vizwiz_five_representative_models.csv`: requested five-model subset.
- `nine_model_macro_summary.csv`: equal-model macro means of these headline rows; SID has seven available Food models. Defined-only precision mean has its number of defined models explicitly listed.
- `nine_model_paired_gains.csv`: per-checkpoint IP minus native VCD. IP-M3ID gains here are also versus native VCD, not native M3ID.
- `food_model_reference_denominators.csv`: frozen reference positives for each model, N=2,424 each, from all483 condition summaries.

## Controls and finite mechanism
- `food_matched_and_reference_off_ablation.csv`: fixed-marker 9×4×2×3=216 rows from `all_food_condition_metrics483.csv`; `condition_id` locates the exact row. Extension-four marker identity comes from source `main_marker` when `marker` was empty. IP same-marker conditions use `main/metrics_all_complete.csv`; guided/ref-off conditions use `mechanism/nine_model_all360_condition_metrics.csv`.
- `food_ablation_paired_J.csv`: 72 fixed-marker comparisons derived from those 216 rows. Means across all markers are descriptive, not task-optimal selected points.
- `qwen25_cross_prompt_4x4.csv`: 16 Qwen2.5 guided-VCD main×reference marker conditions from the all483 source; marker/reference columns retain actual identities.
- `nine_model_candidate_support_audit.csv`: original recorded representative101 answer/UNKNOWN candidate selector; sources `support/mechanism/core5/four_view_pair_margins.csv`, `extra4_representative101/four_view_pairs.csv` and matching `four_view_events.jsonl.gz`. Counts refer to fixed-prefix proxy pairs, not semantic complete replies.
- `carpaccio_four_view_margins.csv`: exact four log-odds from `support/cases/beef_carpaccio_eval039/beef_carpaccio_margins.csv`; pair-normalized probabilities are sigmoid(log-odds), not full-vocabulary probabilities.
- `carpaccio_counterfactual_math.csv`: arithmetic recombination of the preceding four observed margins at alpha1. Eta crossing is an analytical threshold, not a newly generated experiment.

## Flow provenance
- `llava_archived_original_reasonable_abstention_flow.csv`: inherited 287-input paired summary, immediate source latest-v3 `tables/F4_original_abstention_flow.csv`. Full C852/N2424, IP retained61 and subset denominator287 are corroborated in detailed main metrics. Original guided-Direct subset IDs are absent, so inherited 19/235 and21/205 splits are not freshly raw-recomputed.
- `llava_verified_reference_positive_flow.csv` and `llava_verified_paired_transitions.csv`: separately re-audited all949 current-reference-positive cohort from mechanism condition135 and LLaVA IP-VCD UNKNOWN main condition. Do not substitute/mix these counts with the original287 cohort.

## Runtime and validation
- `runtime_matched_full_cohorts.csv`: 42 method rows, 14 complete model×marker cohorts from `mechanism/runtime_records.csv.gz`. Source execution identities, GPU UUID, dtype, software and per-source count allocations match across the three compared methods. Each has2,424 inputs. Output lengths and broad per-answer wall interval remain explicit. Ratios are against guided matched-marker VCD, not native VCD. Original five models only.
- `audit_checks.csv`:26 aggregate-rate/partition checks.
- `raw_record_audit_checks.csv`:26 raw-record checks across all483 Food and56 VizWiz conditions. Sources: `main/main_scores.parquet`+condition dictionary; `mechanism/mechanism_scores.parquet`; `mechanism/extension4/scores.parquet`; old Viz31 `support/vizwiz/core5_512/new_scores.parquet`; new Viz25 `vizwiz_final/new25_complete_records.parquet`. All checks passed. Arithmetic verification does not revalidate the semantic judgments.

## 2026-10-04 additions
- `supplementary/gemma_sid/`: full 2,424-key SID acceptance, original identities, metrics and paper merge receipt. Combined Food coverage is 484 conditions / 1,173,216 responses. The six pre-existing SID full-cohort points retain alpha1 plus normal-support restriction; Gemma uses the author greedy operator alpha0.5 on the full vocabulary. The existing 101-image author-operator paired comparison is separate.
- `vizwiz_paired_J_intervals.csv`: 18 sample-paired 95% percentile bootstrap intervals (2,000 resamples, seed20260929) calculated from the saved 512-input decisions. This is a new statistical analysis of the existing responses.
- `supplementary/`: actual Food development choices, same101 DoLa/SID comparisons, CDA five-entropy measurements and replay-source checks. See `supplementary/source_rows.csv` for the available source mapping.

- `vizwiz_native_merge_receipt.json`: accepted23 new native conditions,11776 responses, zero semantic/reference pending; old45 headline and old56 sample scores unchanged. New source_row uses the one-based CSV line including the header.
- `vizwiz_SID_applicability.csv`: four whole-panel original-SID exclusions. No adapted or reduced-token SID enters these tables.
