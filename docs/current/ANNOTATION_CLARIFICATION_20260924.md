# Annotation source clarification and completed census gaps

User requested Luna annotation of the 108 remaining census answers and zero correctness credit for the two incomplete formal replies. An actual GPT-6 Luna medium subagent read all 110 blinded question/answer pairs. Root review identified compound-question extraction omissions; a second semantic pass produced 74 versioned corrections, retaining the initial labels. Coverage, source hashes, exact quotes, frozen scoring and execution receipts all passed. No model inference or paid API call was made.

## Current result authority

Base: `outputs/annotations/luna_census_remaining_v1/remaining108_plus2_20260924`.

- `census_merged_v1`: 293344/293344 automatic behavior labels, zero unresolved. Preserves 293236 prior valid labels and adds 108 actual Luna labels (99 assertive, 8 uncertain, 1 invalid). Metrics reapply the unchanged Food aliases and official VizWiz scorer to original sources.
- `formal_fragments_resolved_v1`: the original 9047 semantic rows are now all labeled, including DON'T and The Celti as invalid with zero correctness. All 26664 full-stage rows remain. No regenerated response replaces either fragment.
- `completion.json`, source mapping, execution, corrections and acceptance records provide audit evidence. Automatic behavior annotation is not human review or final model-specific should-abstain GT.

The 9047 historical unresolved responses are from fifth-step formal experiments: LLaVA main-table stage 6880, Qwen2.5 control stage 2167. They are neither the sixteen-model census nor numerical candidate-class scores. Preserve old validated_v3 and its latest pointer byte-for-byte because the ongoing independent queue binds them; use the new separate formal view for updated statistics.

The new independent-answer Luna queue is different: completed Qwen2.5, MiniCPM and LLaVA production has 145440 original answers. Its free matching left 91281 unresolved, with 22051 covered by exact valid-label reuse; 49876 new question-answer groups cover the other 69230 rows. Two real Luna medium agents are processing the 499 frozen batches. Qwen3.5 generation is also complete, but is not in that three-model annotation queue; Gemma generation remains a separately monitored run. Do not confuse generation, preliminary matching, semantic labeling and final should-abstain evidence.

4029 remains withdrawn under PROTOCOL_AMENDMENT_20260924_RESOURCES.md. The automation retains its two-hour interval and references these corrected authorities without repeated labeling.
