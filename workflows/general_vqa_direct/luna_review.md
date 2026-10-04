# Actual Luna review for this Direct supplement

Use the two batch IDs and role assigned by root. Do not claim another batch,
generate answers, change code, or alter the immutable assignment. The assigned
agent is the actual GPT-5.6 Luna reviewer at medium effort.

SSH alias: `4028-root`. Project:
`/home/g203-4028/projects/knowledge-deficit-mitigation`.
Local copy root:
`E:/OneDrive/文档/Playground/kdm_remaining11_20260930_140337/general_vqa_direct_20261004`.

Read each complete question and complete response, at most five items at a time,
then write the judgments. Finish and validate one batch before reading the next.
Do not replace semantic reading with keyword-generated decisions. Keep all text
exactly as supplied. Read JSONL with UTF-8-sig where necessary. Write actual
newlines using `chr(10).join(...)`, and reread each line with `json.loads`.

## Behavior

Inputs: `outputs/general_vqa_direct/annotations/<batch>/pending_review.jsonl`.
Read the schema and validators in `workflows/general_vqa_direct/score.py`.
Use the current completed batch 0106 as a format example.

A wrong answer remains an answer. An explicit answer outside the supplied
options remains an answer with `predicted_answer=null`. Abstention requires a
real refusal or inability to answer. A clear answer that is not withdrawn remains
an answer when later explanation is truncated; unfinished reasoning with no
answer is invalid. Extract the final clear original option letter; unresolved
competing answers get null. Binary extraction is `yes`, `no`, or JSON null,
never the string `"null"`.

Keep `answer_text` as the full response and select a meaningful continuous
original `evidence_span`. For invalid reasoning without a final answer, the full
response may be needed as evidence. Do not cut a decimal or formula at its first
period and present the fragment as a complete span.

Write complete `decisions.jsonl`. Validate unique full assigned-key coverage,
exact question/answer bindings, every `score.check_decision`, and Hallusion
`check_behavior_prediction` where applicable.

## Hallusion reference-relative quality

Inputs: `outputs/general_vqa_direct/quality_annotations/<batch>/pending_review.jsonl`
and `instructions.md`. Read `hallusion_scoring.py` and `validate_quality`.
Use completed batch 0103 as a format example.

Read the full official reference as well as the full question and response.
Check question polarity, the object, and the comparison. A matching first yes/no
does not make a conflicting factual answer correct. Use incorrect for a conflict
with the reference, and unclear for no clear answer or unresolved contradiction.
A later explicit final conclusion can resolve an earlier reversal; do not mark
every reversal unclear. Truncation alone does not negate an earlier clear answer.
Handle original `No answer` and inconsistent-reference items according to their
full reference, preserving the source text.

Write complete `quality_decisions.jsonl`. Validate every row with the frozen
`validate_quality`, all unique assigned keys, and exact question, response, and
reference bindings. Do not modify the independently assigned behavior labels.

## Provenance and completion

Use your actual canonical agent name as `author`, `model=gpt-5.6-luna`,
`effort=medium`, and `call_id=null` when an actual call ID is unavailable.
Only after all validation passes, write `write_receipt.json` with counts, SHA256,
actual author/model/effort and completed checks. An in-progress file is not a
completed batch. Repair a failed check before reporting completion.

Do not create a partial `ACTIVE.json`. Root merges any rereview patch with its
complete original batch. Keep original decisions and correction provenance.
Copy completed behavior files to local `annotations/<batch>_decisions.jsonl`
and quality files to local `quality_annotations/<batch>/`, with their receipts.
Report row count, SHA256 and actual validation, including any remaining problem.
