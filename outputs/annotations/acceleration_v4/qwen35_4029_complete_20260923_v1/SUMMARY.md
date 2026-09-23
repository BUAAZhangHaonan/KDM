# Food formal generation: preliminary character matching

Only completed and provenance-verified input shards are included. These are full-answer lexical matches, not final semantic labels or abstention GT. Unmatched text remains unresolved. No API request or human review is claimed.

Screened rows: 155136. Included complete model outputs: 1/5. Full panel present: False.

## Stage totals

| Model | Stage | Rows | Matched correct | Matched incorrect | Exact abstention | Unresolved |
|---|---|---:|---:|---:|---:|---:|
| qwen35_4b | prompt_matrix | 130896 | 55952 | 5946 | 29720 | 39278 |
| qwen35_4b | unknown_controls | 12120 | 4646 | 579 | 2552 | 4343 |
| qwen35_4b | unknown_main | 12120 | 5933 | 573 | 1904 | 3710 |

## UNKNOWN main-condition preliminary table

Every row below uses UNKNOWN for both main and reference where applicable. A subset of completed shards is explicitly marked by model_coverage in summary.json; it is not a whole-model result.

| Model | Method | Rows | Matched correct | Matched incorrect | Exact abstention | Unresolved |
|---|---|---:|---:|---:|---:|---:|
| qwen35_4b | deco | 2424 | 1118 | 91 | 669 | 546 |
| qwen35_4b | direct | 2424 | 1182 | 114 | 498 | 630 |
| qwen35_4b | dola | 2424 | 1191 | 107 | 510 | 616 |
| qwen35_4b | m3id | 2424 | 1256 | 141 | 142 | 885 |
| qwen35_4b | vcd | 2424 | 1186 | 120 | 85 | 1033 |

All prompt-matrix conditions are retained in summary.json. No unresolved item was counted as an incorrect answer. Generation completion does not establish semantic judging, should-abstain GT, mechanism completion or research completion.
