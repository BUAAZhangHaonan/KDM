# Supplementary VizWiz baseline results

The separate VizWiz supplement uses the registered evaluation split of 3,501 questions, with 2,365 officially answerable and 1,136 unanswerable questions. At the 30 September 2026 checkpoint, all eleven Direct/UNKNOWN conditions contained the full evaluation set, providing 38,511 scored responses with resolved behavior and extracted-answer spans. Each response retains the complete generated text, the original answer span, all ten official human answers, and source identity.

Scores use the frozen official VQA normalizer and consensus-credit function. All-input accuracy divides total VQA credit by 3,501; the answerable-set Direct score divides credit on officially answerable questions by 2,365. Abstentions and invalid responses receive zero credit. Answer coverage counts concrete answers with assertive or uncertain wording, and invalid responses are reported separately.

| Model | All-input VQA credit | Abstention rate | Answer coverage | Answerable-set Direct credit | Invalid responses |
|---|---:|---:|---:|---:|---:|
| Gemma3-12B | 17.07% | 21.28% | 75.58% | 24.26% | 110 |
| GLM-4.6V | 26.55% | 24.16% | 75.09% | 37.84% | 26 |
| InternVL-3.5-8B | 23.62% | 23.11% | 73.46% | 33.72% | 120 |
| LLaVA-1.5-7B | 27.54% | 19.14% | 76.89% | 38.20% | 139 |
| LLaVA-OneVision | 30.07% | 12.11% | 84.63% | 42.36% | 114 |
| MiniCPM-V-4.5 | 22.21% | 22.65% | 74.32% | 31.70% | 106 |
| Phi-3.5-Vision | 21.91% | 34.39% | 60.35% | 31.17% | 184 |
| Qwen3-VL | 27.84% | 40.39% | 53.98% | 40.30% | 197 |
| Qwen3.5-9B | 29.06% | 33.33% | 62.33% | 42.00% | 152 |
| LLaVA-1.5-13B | 26.41% | 31.62% | 63.47% | 37.31% | 172 |
| LLaVA-1.6-Vicuna | 20.85% | 48.27% | 44.73% | 30.23% | 245 |

Human answerability and success on answerable questions from ten independent responses are separate registered evidence sources. The independent-response reference scores are still unavailable in this baseline checkpoint. No combined binary model knowledge-deficit label is defined. For example, Qwen3.5-9B abstained on 389 human-answerable questions and 778 unanswerable questions; both counts retain their corresponding fixed set denominators. These observations describe available baseline behavior, and the checkpoint contains zero method-effect comparisons.

The full registered VizWiz matrix contains 800 conditions and 2,800,800 response records; 2,762,289 formal scores remain absent from this checkpoint. Independent reference scoring requires 385,110 registered response records. Exact missing condition contexts and source-bound sample keys are preserved separately from Food-101.

The scoring entry point is `workflows/supplemental/remaining11/vizwiz_score.py`. Reproducible outputs are saved under `outputs/supplemental/remaining11/run_20260930_140337/scores/vizwiz_direct_baseline_20260930`, including the per-sample table, condition metrics, answerability metrics, complete condition coverage, 66 source-bound examples, and PNG/PDF baseline plots. Validation recomputed official VQA credit for all 38,511 responses, verified uniqueness of both formal and original keys, checked all eleven full sample sets and answerability quotas, and matched nine output digests and six input digests. The figure was visually inspected.

Baseline responses reuse the original guided/UNKNOWN census outputs and their final source-bound annotations. Reading the complete 293,344-row annotation closure yielded 21,930 exact question–full-response witnesses for this subset. Original keys, source identities, file paths, line numbers, row hashes, annotation identities, and historical authorship remain traceable. This CPU scoring run used no new API calls or GPU initialization. Historical accepted references contain no records for these eleven models and remain null.
