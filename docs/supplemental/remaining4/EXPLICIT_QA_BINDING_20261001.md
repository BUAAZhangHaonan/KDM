# Explicit model decisions and accepted reference bindings

`persist_explicit_viz_reviews.py` serializes only the model's already explicit
per-QA judgments. The immutable complete-QA queue, assigned interval, decision
file and actual model-read attestation are hash-bound. The writer creates no
default semantic label, first-sentence extraction, answer span or target-based
candidate. It retains the original declaration and actual author/model/effort;
root is recorded separately as the serialization writer. Unknown call IDs,
sessions and elapsed time remain unknown.

Noncontinuous spans and nonempty invalid decisions from the observed problematic
family require actual review. They are retained, not automatically corrected or
scored. Complete source-label coverage remains separate from primary-answer and
behavior acceptance. At the v9 finite checkpoint, 29,400 QA source records were
bound, with 5,972 known candidate QAs still held in both scoring dimensions.
Actual explicit model batches totaling 420 additional QAs and 19 full-QA root
reads (13 appended decisions) passed binding; 120 previous candidate reviews
remain separately source-bound. Original automatic-span outputs are preserved
and are not counted as individually reviewed primary-answer spans.

The actual passed score application retained all 72,776 Food objects unchanged
and applied exact-QA decisions to the 85,219 Viz objects. The 157,995 new objects
and 51,569 accepted historical Food objects then joined into 209,564 unique
generation keys, with no duplicates. Seventeen Food conditions have all 2,424
inputs and exactly 24 inputs for each of 101 classes. Partial conditions remain
partial; Viz pending rows are not defaulted to zero.

`join_received_references.py` now preserves an existing accepted reference
binding when its source hash, boolean, gold rank and correct-count fields match
the specified accepted source. Conflicting bindings fail. This fixes a repeated
join that previously replaced its own provenance before the unchanged-object
check. The real 209,564-object join preserved all 124,345 existing Food bindings
and every original scientific score, abstention and source field. New generation
and frozen five-model outputs are unaffected.

Actual receipts are under `outputs/supplemental/remaining4/`:

- `explicit_review_bindings_20261001_2245/`: finite model-read bindings;
- `actual_finite_root19_20261001_2248/`: actual appended root decisions;
- `semantic_authority_20261001_2248_v9_actual540/`: source closure and review holds;
- `received157995_20261001_2248_actual540/`: score application;
- `received209564_20261001_2250_reference_bound/`: key, reference and quota checks.
