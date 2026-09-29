# Export schema and source mapping

All line locators are 1-based lines in the decompressed JSONL. Integer dictionary IDs are local to this export. Strings and existing semantic values are copied verbatim; Parquet Boolean columns use true/false and retain native nulls when present. CSV empty cells denote unavailable fields, and CSV list/object cells use JSON.

## Main table mapping

`scores.parquet` primary key: `(condition_id, sample_id)`. Join `condition_id` to `conditions.csv` for model and split. The condition key is the exact tuple `(model, method, kind, marker, reference_marker, guided, reference_guided, replicate)`, sorted to assign IDs. `marker` is also exported as `main_marker`. Each of the 352 conditions has 2,424 eval samples.

| Export field | Frozen field/source |
|---|---|
| correct_canonical | score_rows.canonical_name_in_primary_score (0/1 to bool) |
| correct_literal | score_rows.literal_extracted_name_score (0/1 to bool) |
| abstain | score_rows.abstain |
| uniform_reference | uniform_reference_gt.gt joined on model/sample_id |
| accepted_reference | reference_gt.gt joined on model/sample_id |
| sample_id, target_class, seed, behavior, score_reason | same-named score_rows fields |
| source_file_id / source_line | dictionary encoding of score_rows.source_path / existing source_line |
| score_source_line | line in the frozen score_rows.jsonl.gz |
| qa_id | dictionary encoding of score_rows.qa_key |
| detail_id | dictionary of the copied extraction/review fields plus matched existing automatic_behavior fields |
| binding_id | dictionary of existing main/reference/neutral prompt and config source hashes |

`qa.parquet` retains existing qa_key, exact review question/answer and review variants when recorded. `raw_model/raw_source_file_id/raw_source_line` always locate one original formal response for that QA. Review text may be null. The complete cases include every requested full answer. To recover any other full answer, read the located line from the gzip listed in sources.csv. Its existing `key` is the original response key. One QA can occur in many scored rows with different targets; correctness remains per score row.

`score_details.parquet` uses JSON strings for each original field so lists, strings, null, false and empty lists keep their distinct values. Apply `json.loads` per cell. `canonical_name_in_primary` and literal extraction can be null even when the separate frozen score is decided. `stored_behavior_*` holds the auxiliary automatic_behavior file's original values and source; final `scores.abstain/behavior` remains the paper scoring source.

`conditions.prompt_id/config_id` connects to the arrays in prompts_and_configs.json. `prompt_ids/config_ids` preserves all actual variants per condition; the present export has one of each per condition. The JSON stores actual main/reference/neutral prompt text, complete DecodeConfig values and first/last source lines. `bindings.json` retains already-recorded hashes and locators.

## References and controls

`references.parquet` key: `(model, sample_id)`; split is retained for all 24,240 dev/eval rows. `gold_rank` maps to uniform gold_rank; `independent_correct_attempts` to uniform independent_correct_attempts_exact; `independent_attempts` to uniform attempts. Both source records are additionally retained verbatim as `uniform_original_json` and `accepted_original_json`, with separate source-file IDs/lines. Main retention denominator is fixed to the condition's Direct-abstain AND uniform_reference set. Historical accepted-reference results use the separate accepted field.

`control_conditions.csv` defines 200 IDs in a separate control namespace. `control_selections.parquet` key: `(control_condition_id,sample_id)`. Join `(selected_condition_id,sample_id)` to scores to obtain the frozen selected response. `base_condition_id` is the treatment condition and `direct_condition_id` its original Direct. The selected file/line and QA are retained. `source_line` here is the original selections.jsonl.gz line. No new selection rule is executed.

## Sources, runtime, cases and checks

`sources.csv.source_file_id` is the global export dictionary. Runtime's historical `source_id` is a separate 21-segment namespace bridged by `sources.runtime_source_id`. The current consolidated gzip paths are active sources. Archived segments use `archive.tar::member` notation, with preserved consolidated line ranges. Paths containing older host names describe historical records and were read from the central archive.

`runtime_records.csv` records 1,147 condition/source/identity groups. n_rows, wall_s_sum/mean/min/max and tokens_sum/mean/min/max come from saved per-response wall_s and len(tokens). Source ranges are filtered by the explicit model/condition/identity in each record. wall_s spans sample image loading, session generation and cleanup; source model initialization and ledger writing sit outside this timer. The source has no separate prefill/decode timings or peak memory fields. `runtime_availability.csv` and `availability.csv` preserve field-specific availability and JSON-key sources. Runtime comparison rows, when present, explicitly identify matching device/configuration/input sets and formulas. Existing scalar caches and their sizes/coverage are listed for later selection.

`cases.jsonl` includes 12 original inputs, Direct/VCD/IP-VCD/copied-control full response records, all ten saved independent attempts, reference values and image paths/dimensions. `cases/images/` contains byte-identical source images. `case_group_availability.csv` records all eligible counts and selected counts for the stated frozen-flag combinations. Group labels describe those combinations; the behavior values remain the original labels.

`baseline_tables/` copies the existing 352-condition metrics, 372 paired comparisons and control summaries. `context/paper_start/tables/instruction_pairs_all40.csv` is copied from the uploaded paper package. Confidence intervals are retained from those existing files. `export_checks.json` records CPU row/key/count/join checks and comparison of all 40 point estimates. The executed export scripts are in scripts/.

## Arrow schemas

### control_selections.parquet

| Field | Arrow type |
|---|---|
| control_condition_id | `int16` |
| sample_id | `string` |
| selected_condition_id | `int16` |
| base_condition_id | `int16` |
| direct_condition_id | `int16` |
| selected_source_file_id | `int16` |
| selected_source_line | `int32` |
| qa_id | `int32` |
| selection_reason | `string` |
| direct_abstain | `bool` |
| selected_abstain | `bool` |
| source_line | `int32` |

### qa.parquet

| Field | Arrow type |
|---|---|
| qa_id | `int64` |
| qa_key | `string` |
| question | `string` |
| answer | `string` |
| review_variants_json | `string` |
| review_source_file_id | `int64` |
| review_source_line | `int64` |
| raw_model | `string` |
| raw_source_file_id | `int64` |
| raw_source_line | `int64` |

### references.parquet

| Field | Arrow type |
|---|---|
| model | `string` |
| sample_id | `string` |
| split | `string` |
| target_class | `string` |
| uniform_reference | `bool` |
| accepted_reference | `bool` |
| gold_rank | `int64` |
| independent_correct_attempts | `int64` |
| independent_attempts | `int64` |
| uniform_source_file_id | `int64` |
| uniform_source_line | `int64` |
| accepted_source_file_id | `int64` |
| accepted_source_line | `int64` |
| uniform_original_json | `string` |
| accepted_original_json | `string` |

### score_details.parquet

| Field | Arrow type |
|---|---|
| detail_id | `int64` |
| literal_extracted_names | `string` |
| literal_extracted_name | `string` |
| canonical_class_candidates | `string` |
| canonical_name_in_primary | `string` |
| review_sources | `string` |
| review_statuses | `string` |
| multiple_primary_classes | `string` |
| stored_behavior_label | `string` |
| stored_behavior_abstain | `string` |
| stored_behavior_source | `string` |

### scores.parquet

| Field | Arrow type |
|---|---|
| condition_id | `int16` |
| sample_id | `string` |
| target_class | `string` |
| correct_canonical | `bool` |
| correct_literal | `bool` |
| abstain | `bool` |
| uniform_reference | `bool` |
| accepted_reference | `bool` |
| source_file_id | `int16` |
| source_line | `int32` |
| qa_id | `int32` |
| seed | `int64` |
| detail_id | `int32` |
| binding_id | `int16` |
| score_source_line | `int32` |
| behavior | `string` |
| score_reason | `string` |
