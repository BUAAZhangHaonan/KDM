# Supplementary remaining-model results

The remaining-model Food-101 supplement uses the registered evaluation split of 2,424 images, with 24 images in each of the 101 original classes. The complete registered matrix contains 832 conditions and 2,016,768 response records. At the 30 September 2026 checkpoint `v7_luna_reference1220_20260930`, 63,572 formal records and 351,162 independent-response scoring records were available. All eleven Direct/UNKNOWN conditions had complete sample coverage and resolved primary correctness and abstention judgments, providing 26,664 fully resolved primary-metric rows. Three complete Phi-3.5 method conditions increased eligible coverage to 14 conditions and 33,936 rows; the other 818 registered conditions still have generation or annotation gaps.

| Model | Correct / 2,424 | Canonical accuracy | Abstentions / 2,424 | Abstention rate | Answer coverage |
|---|---:|---:|---:|---:|---:|
| Gemma3-12B | 1,381 | 56.97% | 1 | 0.04% | 99.96% |
| GLM-4.6V | 1,308 | 53.96% | 6 | 0.25% | 99.75% |
| InternVL-3.5-8B | 623 | 25.70% | 37 | 1.53% | 98.47% |
| LLaVA-1.5-7B | 346 | 14.27% | 1 | 0.04% | 99.96% |
| LLaVA-1.5-13B | 522 | 21.53% | 77 | 3.18% | 96.82% |
| LLaVA-1.6-Vicuna | 527 | 21.74% | 249 | 10.27% | 89.73% |
| MiniCPM-V-4.5 | 1,493 | 61.59% | 13 | 0.54% | 99.46% |
| LLaVA-OneVision | 626 | 25.83% | 1 | 0.04% | 99.96% |
| Phi-3.5-Vision | 687 | 28.34% | 22 | 0.91% | 99.09% |
| Qwen3.5-9B | 1,181 | 48.72% | 651 | 26.86% | 73.14% |
| Qwen3-VL | 1,158 | 47.77% | 517 | 21.33% | 78.67% |

Canonical correctness uses the extracted primary dish and the original Food-101 class names, including their underscore-to-space forms. Literal equality of the complete extracted name is retained as a separate sensitivity measure; Gemma3-12B and InternVL each retain three unresolved literal-score judgments while their primary correctness and abstention fields are complete. These checkpoint observations describe the available baseline conditions. Method effects require complete conditions and paired evaluation on the same samples. The existing Gemma3-12B and GLM-4.6V Direct primary-correctness boundary cases have been adjudicated in appended records.

The uniform Food-101 knowledge-deficit reference is positive exactly when the correct category ranks below first and none of the ten independent responses is correct. The registered dev/eval reference manifest contains 53,328 model–sample keys, of which 25,342 were complete and 27,986 remained unresolved at this checkpoint, adding 930 complete references since v5. LLaVA-1.5-7B, LLaVA-1.6-Vicuna, OneVision, and Qwen3-VL now have all 2,424 dev and 2,424 eval reference keys resolved. Their Direct/UNKNOWN reference metrics below use the full eval set; numerators and reference-positive denominators are reported explicitly.

| Direct model | Reference positives | TP / FP / FN | Abstention precision | Abstention recall |
|---|---:|---:|---:|---:|
| LLaVA-1.5-7B | 1,296 | 1 / 0 / 1,295 | 100.00% | 0.08% |
| LLaVA-OneVision | 813 | 1 / 0 / 812 | 100.00% | 0.12% |
| Qwen3-VL | 364 | 188 / 329 / 176 | 36.36% | 51.65% |
| LLaVA-1.6-Vicuna | 1,179 | 200 / 49 / 979 | 80.32% | 16.96% |

The remaining eval-reference gaps are 2,255 for Gemma3-12B, 2,128 for GLM-4.6V, 2,424 for InternVL, 2,424 for LLaVA-1.5-13B, 757 for MiniCPM-V-4.5, 2,424 for Phi, and 1,575 for Qwen3.5-9B. Reference-dependent metrics for these models remain null. Additional cross-validation of whether abstention is appropriate remains deferred.

The original historical accepted-reference file contains 14,544 unique model–sample records for LLaVA-1.6-Mistral, MiniCPM-2.6, and Qwen2.5-VL, with 2,424 dev and 2,424 eval records per model. It has no records for the eleven supplemental models; their historical accepted-reference fields remain null.

Pairing matches model, sample, main marker, guidance, and replicate, and verifies the recorded main-prompt hash and random seed. Confidence intervals use 2,000 shared resamples of the 101 Food-101 category clusters with seed 20260929. Accuracy noninferiority uses the registered lower 95% bound greater than −0.01. Instruction-preserving variants are compared with their corresponding base methods and a registered control that preserves the Direct response whenever Direct abstains. The selection rule reads Direct abstention only; all selected source records remain traceable.

The first complete method comparisons are Phi-3.5's M3ID, DoLa, and DeCo, with `kind=main`, main and reference markers both UNKNOWN, guidance present on both sides, and replicate zero. Each comparison pairs all 2,424 samples against the same-main-prompt Direct baseline, with zero recorded prompt-hash or seed mismatches. Accuracy differences and 95% class-cluster paired bootstrap intervals are in percentage points.

| Phi-3.5 condition | Correct / 2,424 | Abstentions / 2,424 | Canonical accuracy | Difference vs Direct [95% CI], pp | New correct / original correct lost |
|---|---:|---:|---:|---:|---:|
| Direct | 687 | 22 | 28.34% | — | — |
| M3ID | 707 | 5 | 29.17% | +0.83 [+0.17, +1.49] | 31 / 11 |
| DoLa | 681 | 13 | 28.09% | −0.25 [−4.04, +3.42] | 197 / 203 |
| DeCo | 529 | 670 | 21.82% | −6.52 [−9.74, −3.30] | 68 / 226 |

These results apply to the three listed registered conditions. The corresponding M3ID copy control is complete: it selects the original Direct response on 22 Direct-abstention samples and the M3ID response otherwise, yielding 706 correct responses (29.13%) and 22 abstentions across the same 2,424 samples. Selection uses Direct abstention only. Phi still lacks all eval category-rank references, so reference-dependent abstention recall and reasonable-abstention retention for these conditions remain null. Original source identities, cohorts, and per-sample selections are preserved.

The complete analysis registration includes 964 paired comparisons and 440 copy-control conditions. This checkpoint has three complete method comparisons and one complete copy control; 961 comparisons and 439 controls remain incomplete. Other models, marker combinations, and instruction-preserving, CDA, and SID conditions require their registered full sample coverage. Exact missing contexts and sample keys are retained; generated subsets do not replace the 2,424-sample denominator.

Reproducible artifacts are saved under `outputs/supplemental/remaining11/run_20260930_140337/analysis/checkpoint_v7_20260930_1845`: `sample_results.jsonl.gz`, `condition_metrics.csv`, `condition_coverage.csv`, `reference_table.jsonl.gz`, `reference_coverage.csv`, `paired_comparisons.jsonl`, `missing_comparisons.jsonl`, `copy_control_coverage.jsonl`, `copy_control_selections.jsonl.gz`, and `cases.jsonl`. Baseline and complete-method figures are provided as PNG/PDF in `figures/direct_unknown` and `figures/unknown_method_effects`. Actual verification checked 14 output digests, all fourteen class quotas, and identity-pair/selection invariants on 33,936 real records; both PNGs were visually inspected. The case table contains 63 real responses. Earlier v5 and v4 artifacts remain saved in their original checkpoint directories. The current formal missing-key count is 1,953,196.

Illustrative source-bound examples are LLaVA-1.5-13B's “Apple pie” response to `food101:apple_pie_eval_025.jpg` (correct), “Cheesecake” to `food101:apple_pie_eval_024.jpg` (incorrect), and “UNKNOWN” to `food101:apple_pie_eval_032.jpg` (abstention). Each example retains the complete response and its source identity, path, line number, and SHA in the case table.
