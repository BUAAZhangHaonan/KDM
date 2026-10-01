# Finite VizWiz answer-span review holds

Actual root review found that some actual Luna behavior batches stored spans with
an automatic first-sentence rule. A recorded author/model does not establish that
the short answer was individually selected. Original annotations and scores are
preserved; corrections are appended with their actual authors and source SHA.

`prepare_viz_span_review.py` screens an explicitly bounded accepted QA authority.
It selects review candidates and reads no official answers. It creates no labels,
does not default unknown words to invalid, and does not alter answer text.
The actual 29200-QA population yielded 6312 candidates. Root reviewed 24 complete
QAs and appended nine corrections. A genuine 60-QA explicit Luna pilot was then
checked, with three additional root span corrections. The subsequent automatic
200-QA span batch remains unaccepted and is not included in the authority.

`compose_semantic_authority.py` binds a hashed finite review scope and preserves
the actual annotation history. A held QA remains provisional in behavior and/or
answer quality until a complete-QA actual per-QA choice is supplied. The actual
v7 authority has 6252 remaining candidates, with its resolution flags explicit.
Unknown call IDs remain empty; unverified timing claims are not measurements.

`apply_viz_reviews.py` and `score_received.py` honor both resolution flags: held
behavior is null and held continuous quality is null. They preserve the raw
official consensus score, original generation identity/config/reference, and
unchanged Food records. Source-line hash aliases retain their original field.

Actual CPU validation: 139185 unique received records, with 72776 Food objects
unchanged; 18810 additional sealed OneVision VizWiz replies were source-checked
and scored with finite holds. The genuine model final-message judgments are
stored separately from root serialization and unresolved review flags.
These are interim received cohorts, not completion of all registered experiments.
