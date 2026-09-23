# Completed three-model independent-answer semantic queue

This versioned workflow prepares only the completed Qwen2.5-VL, MiniCPM-V 2.6 and LLaVA independent Food-101 cohorts. It does not restart generation, repeat free postprocessing, modify frozen source code, or call an API. User authorization: batch completed unresolved answers with GPT-6 Luna, medium reasoning; keep automatic judgment separate from human review and final abstention GT.

Active queue: `outputs/annotations/luna_independent_v2/completed3_20260923_v1`.

All 145,440 answers pass original task/input/config/seed/EOS/identity validation and unique 4,848 questions times ten attempts per model. The existing 101-class score cohorts are independently rescanned by the existing validator. The 91,281 free-screening unresolved rows yield 50,711 exactly equal question-plus-answer groups. Of these, 835 active validated_v3 groups reuse prior valid semantic judgments for 22,051 rows. There are 49,876 new groups (69,230 rows), split into 499 batches of at most 100. The two previously unresolved groups have no exact matches here and are not requeued. No approximate matching or new semantic judgment was used to prepare the queue.

`prepare.py` writes blinded question/answer batches, full source-ID mapping, exact reuse lineage, original rubric, and a manifest binding every source and queue artifact by SHA256. The source mapping and reuse files contain identifying information and must not be shown to the blinded semantic judge. Supply only `RUBRIC.txt` and the assigned `batches/input_NNN.jsonl`.

First review batch 001 for short-answer quality before scaling. A subagent must actually read the original question and full answer and author the judgment. Remove articles, portion phrases and unrelated image description from the endorsed-answer span when semantically justified, but preserve food identity qualifiers. Do not pick the correct option from a multi-answer response or use the reference class to select a span. Evidence and answer spans must be literal original substrings. Uncertain semantic interpretation may remain unresolved. The validator checks literal/schema correctness, not semantic quality.

Each `results/batch_NNN.jsonl` contains exactly its input IDs and hashes:

    {"id":1,"group_sha256":"...","annotation":{"label":"answer_assertive","evidence_span":"...","answer_text":"..."}}
    {"id":2,"group_sha256":"...","status":"unresolved","reason":"..."}

Allowed behavior labels are the unchanged four rubric labels. Revisions are separate `results/corrections_vN.jsonl`, each with ID/hash, a reason, and annotation or unresolved state. Earlier outputs stay intact.

Before finalization, save `pilot_review.json` with model `gpt-6-luna`, reasoning_effort `medium`, short_answer_quality_passed true, reviewed_ids matching input_001, reviewed_files_sha256 mapping the input, result and any reviewed corrections, human_reviewed false, final_gt false. Save `batch_execution.json` with the same model/reasoning and false flags plus `batches`: each entry has input_file, input_sha256, result_file, result_sha256 and the actual subagent identifier. These receipts must describe real work, never an assumed review.

Validation:

    venv/bin/python workflows/luna_independent_v2/finalize.py --batch 1
    venv/bin/python workflows/luna_independent_v2/finalize.py --check-only
    venv/bin/python workflows/luna_independent_v2/finalize.py --export validated_v1

The full finalizer requires all batches, exact coverage and literal spans, source hashes, pilot and execution receipts. It maps group judgments back to individual rows, rescoring endorsed spans against each target sample with frozen Food scoring; reused source correctness is never copied. Original free-screening labels remain unchanged. New merged labels and question-level conjunction evidence are written separately. Unresolved judgments remain unknown. These are automatic preliminary results, not human review or final GT.

The source-host bundles and file manifests are retained at `outputs/records/independent_sync_20260923`. Completed raw/identity/annotation files are copied to canonical central paths only after byte verification. All ancillary workflow/verification/queue dependencies are preserved in source-specific snapshots. Dynamic queue states are observational snapshots rather than completed-result identity. An initial archive hash mismatch caused by concurrently changing queue state was retained; v2 binds each exact archived byte stream. A previously copied K100 progress snapshot was preserved before installing its completed source snapshot. No raw response was replaced.

Status at preparation: no new Luna judgment has been launched or produced by this workflow. Finalizer syntax/import checks pass; full export is intentionally unavailable until genuine batch outputs and review receipts exist.
