# Five-model main-results delivery

The five-model main-results delivery was completed on **2026-09-29 at 03:58 Asia/Shanghai**. The recorded deadline was 10:00 on the same date.

The delivered repository contains final reproducible code, current documentation, valid model assets, datasets, source responses, final annotations, complete registered method comparisons, paired statistics, figures, and the English main-results draft. Historical execution evidence is retained in verified archives outside the active repository.

## Scoring contract

The user accepts extracting an endorsed food name from a sentence: “The dish is fried rice.” maps to fried_rice. The current closeout uses deterministic canonical-category extraction from the identified primary dish. Match the original Food-101 class spelling and its underscore-to-space form at word boundaries. Preserve cooking/flavor modifiers as descriptive source information. `canonical_name_in_primary` is the primary column; `literal_extracted_name` records full extracted-name equality as a sensitivity column. Original text, explicit role relations and the class extraction rule are saved with each decision. Class extraction precedes comparison with the sample target.

Single canonical primary classes and explicit abstention markers are processed mechanically. Side dishes, ingredients, negation and competing primary candidates use role records and finite published rules. Remaining conflicts receive focused review. The final primary scorer uses the canonical Food-101 class inventory. Frozen census assets retain their original lexical configuration and provenance. Opaque text stays a response; explicit refusals, stated inability and task markers identify abstention. Confidence wording accompanies the extracted answer.

## Completed primary artifacts

- Formal score: 853,248 rows, 352 conditions, all primary/literal/abstention fields resolved. Independent replay matches every row and source binding.
- Reference GT: 24,240 model–sample pairs in each of the accepted and uniform tables; 242,400 independent answers with exact correct counts.
- Primary analysis: 352 condition tables and 372 paired comparisons, 2,000 class-cluster bootstrap replicates each.
- Registered copy-original-abstention control: 200 derived conditions and 40 paired instruction comparisons.
- Main-metric review: 193 consolidated root QA decisions plus finite rule and boundary decisions, stored with sources.
- Writing: paper/main.md contains the complete English main-results draft; docs/RESULTS.md contains the Chinese conclusions.

Current state: outputs/records/main_results/state.json. The final scorer, GT builder, analysis and control runners use stable source bundles. Original generation identities and valuable historical evidence are kept in verified archives outside the active repository.

## Review stage

The two-hour closeout automation was deleted after delivery. The manuscript and supporting evidence are prepared for external review. Supplementary mechanism measurements remain future work; their definition and required inputs are documented in `docs/THEORY.md`.
