# Copy-original-abstention control

This derives the registered `copy_original_abstention` condition from the final scored rows. For each VCD/M3ID condition and sample, the control uses the full direct row when direct abstains; otherwise it uses the full corresponding VCD/M3ID row. The choice uses only the direct abstention flag. All 352 main condition rows remain unchanged.

Inputs: final score rows SHA-256 `39d0dc145207304abc7dc9a73f91413c8ed78acb1632b454ec666b8484e59dd1`; mixed and uniform reference-GT files are hash-bound by `reference_gt_manifest.json`.

The derivation contains 200 condition-level controls (100 each for VCD and M3ID), 484,800 per-question source decisions, and 40 matched instruction comparisons (96,960 paired sample-condition observations). Each condition has 2,424 samples across 101 target classes.

The matched comparisons use 2,000 paired bootstrap resamples of the 101 Food-101 target-class clusters with seed 20260929. Ratio metrics use pooled numerators and denominators within each resampled cluster multiset.

## Matched instruction and copy-control tradeoff

| Base method | Reference GT | Matched conditions | Sample-condition pairs | Accuracy Δ | Coverage Δ | Selective accuracy Δ | Abstention recall Δ | Reasonable-abstention retention Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vcd | mixed_reference_gt | 20 | 48,480 | +0.0501 | +0.1539 | -0.0417 | -0.2030 | -0.4417 |
| vcd | uniform_reference_gt | 20 | 48,480 | +0.0501 | +0.1539 | -0.0417 | -0.1987 | -0.4428 |
| m3id | mixed_reference_gt | 20 | 48,480 | +0.0235 | +0.0855 | -0.0387 | -0.1409 | -0.3099 |
| m3id | uniform_reference_gt | 20 | 48,480 | +0.0235 | +0.0855 | -0.0387 | -0.1360 | -0.3071 |

Accuracy-direction counts across the 20 matched conditions per method (mixed GT; uniform GT gives the same accuracy differences):
- vcd: accuracy Δ positive 19/20, negative 0/20; 95% cluster-bootstrap CI fully positive 14/20, fully negative 0/20.
- m3id: accuracy Δ positive 17/20, negative 2/20; 95% cluster-bootstrap CI fully positive 13/20, fully negative 0/20.

Two registered joint conditions: instruction minus copy-control differences, by reference GT:

| Model | Condition | GT | Accuracy Δ (95% CI) | Reasonable-abstention retention Δ (95% CI) |
|---|---|---|---:|---:|
| minicpm26 | instruction_vcd/UNCLEAR | mixed_reference_gt | +0.0747 [+0.0578, +0.0924] | -0.3435 [-0.4005, -0.2900] |
| minicpm26 | instruction_vcd/UNCLEAR | uniform_reference_gt | +0.0747 [+0.0578, +0.0924] | -0.3342 [-0.3968, -0.2790] |
| qwen35_4b | instruction_vcd/I cannot identify it | mixed_reference_gt | +0.0083 [-0.0037, +0.0186] | -0.5714 [-0.7143, -0.4255] |
| qwen35_4b | instruction_vcd/I cannot identify it | uniform_reference_gt | +0.0083 [-0.0037, +0.0186] | -0.5714 [-0.7143, -0.4255] |

Pooled differences above are arithmetic means across matched condition cells. The per-metric defined-cell counts are recorded in `instruction_tradeoffs.jsonl` (reasonable-abstention retention has 19 cells when the Gemma UNKNOWN cell has a zero denominator). Accuracy, coverage, selective accuracy, abstention precision/recall, and retention differences are paired within the same model, marker, prompt, seed, replicate, and sample; selection itself reads only direct abstention. Confusion totals and exact denominators are in `condition_metrics.jsonl`; all per-question source choices are in `selections.jsonl.gz`.
