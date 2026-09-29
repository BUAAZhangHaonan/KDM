# Main results analysis

Score source: 853,248 rows in 352 exact eight-key condition cells; denominator 2,424 per cell.

The mixed reference table retains the accepted Qwen25-VL, MiniCPM-V2.6, and LLaVA-1.6-Mistral references and uses the current primary-name rule for Qwen3.5-4B and Gemma3-4B. `uniform_reference_gt.jsonl` and `condition_metrics.csv` also report all five models under the same current rule. Boolean GT is resolved for every sample in both tables.

By exact model/sample join over all 24,240 samples, 23,529 rows have the same mixed and uniform GT, 400 transition from legacy true to uniform false, and 311 from legacy false to uniform true. The accepted three-model GT retains all 14,544 original values unchanged; 130 unique legacy conflict samples remain separately identified.

Canonical accuracy matches the predicted canonical class against the gold class, using only the extracted primary-name span; literal-name accuracy is a separate sensitivity column. Coverage and selective accuracy use the same score rows. Abstention precision and recall are shown under both reference choices, with unknown behavior/reference rows counted separately.

Paired rows: 372 (332 intervention cells); missing pair controls: 0. Every intervention is paired against the observed direct main-prompt row by model, sample, main marker, guided flag, and replicate. Instruction-VCD/M3ID also have exact corresponding comparisons against VCD/M3ID. Main-prompt SHA and seed match on all 901,728 paired samples.

Accuracy noninferiority is the lower 95% bound of the paired accuracy difference greater than -0.01. Bootstrap intervals use 2,000 shared resamples of the 101 Food-101 target-class clusters. Abstention-recall differences use micro totals of positive-reference numerators and denominators; “reasonable abstention retained” uses the direct-abstain and GT-positive subset. Instruction-versus-base outputs include preservation differences on that fixed direct subset, under both GT tables, and correction retention with specific-answer and all-input denominators.

Uniform-GT unknown rows: 0; mixed-reference unknown rows: 0. All ten independent attempts have exact correct-count values for all 24,240 samples. The exact UNKNOWN/UNKNOWN main and instruction-preserving condition rows are in `unknown_unknown_summary.csv`.

These paired results describe associations under the recorded prompts. Full eight-key tables, paired comparisons, prompt audit, and exact-condition figures accompany this report.

## Paired instruction and descriptive-slice results

Across 40 instruction-VCD/VCD and instruction-M3ID/M3ID paired cells, 26 meet the accuracy noninferiority criterion (paired 95% lower bound above -0.01). Preservation improves (fixed direct-abstain and GT-positive subset; 95% lower bound above zero) in 12/40 cells under the accepted three-model reference and 12/40 under the uniform five-model reference; both criteria hold in 2/40 and 2/40, respectively. The 40-row `instruction_vs_base_summary.csv` gives each condition’s numerator, denominator, and confidence interval under both references.

The UNKNOWN/UNKNOWN table and plot are descriptive: `unknown_unknown_full_conditions.csv` preserves 42 exact main and instruction-preserving conditions, with direct-paired results in `unknown_unknown_paired_methods.jsonl`. Per-model exact-condition figures are in SVG, PNG, and PDF; paired-instruction tradeoff is shown under both GT policies, with 95% intervals and marker/method legends; UNKNOWN/UNKNOWN accuracy and abstention heatmaps show the 9-method by 5-model descriptive slice in PNG/PDF.
