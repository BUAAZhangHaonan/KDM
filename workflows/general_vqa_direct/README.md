# Nine-backbone native Direct supplement

User-approved scope, 2026-10-04: MMMU original test 1,000; ScienceQA image test
1,000; original COCO POPE 1,000; all 951 visual HallusionBench questions.
The same 3,951 frozen inputs are used for each of nine registered backbones,
for 35,559 responses. This workflow does not modify Food/VizWiz results or paper.

`finalize_dataset.py` freezes exact prompts, question IDs, options, every image
and its SHA. `generate.py` reuses native project adapters and greedy Direct with
original model precision. `inputs.py` retains all referenced images for multi-image
MMMU. Limits are 128 tokens for MMMU and 512 for the other datasets, with native
EOS stopping and explicit truncation accounting; the previous Food limit is not
used. POPE and Hallusion use their original question without an added suffix.

`launch.py` dispatches an explicitly assigned physical GPU through the existing
registered lock wrapper. Each owner records PID, start tick, boot ID, complete
sample-key assignment and source/runtime identity. A completed chunk is immutable
and linked to the completed-key ledger by hashes. `STOP_AFTER_CHUNK` releases only
remaining keys after the current chunk is sealed. Transfer only released or never
claimed keys, and verify their cross-host intersection before dispatch.

A failure before the first input needs a separate transfer attestation: verify
the original PID/start tick has exited, the failed claim contains no raw or
pending rows, and all transferred keys equal the failed assignment. Keep the
failure and owner files, record the replacement owner, and check intersection
with every other live assignment. Do not describe this as a completed chunk.

On 2026-10-05, Gemma's registered Torch 2.6/CUDA 12.4 build passed CPU identity
checks on K100 but failed model loading with `no kernel image`. Its 185 untouched
keys moved to compatible 4029 GPUs as disjoint 93/92 shards, with the same model,
BF16 precision and registered runtime versions. CPU readiness alone does not
establish GPU-kernel compatibility. The original failure and transfer evidence
are in `outputs/general_vqa_direct/assignments/`.

`native_fast.py` is an optional execution optimization. It must reproduce all
tokens and EOS states for eight sealed original responses before claiming any
production keys. Preserve a failed gate and its source bindings; do not relax
the equality test or count gate responses as new experimental observations.
The primary `generate.py` Direct entry remains the registered implementation.
Record any move back to this entry, keep the same ungenerated assignment and
configuration, and measure its actual speed before further scheduling.

An annotation `ACTIVE.json` selects a complete batch file, not a one-row patch.
Merge a corrected row with the other original judgments, verify all assigned
keys and full question/answer bindings, then select the merged file by hash.

`score.py` extracts a declared answer before joining the gold label. It first uses
whole-answer rules and identical QA review reuse; semantic boundary cases retain
the complete question and response for actual Luna medium annotation. Unresolved
scores stay missing. A binary `No` is an answer. POPE's original script score is
retained separately from semantic abstention; MMMU uses the official correctness
functions without random-choice fill for unparsed responses. Hallusion reports
image-question accuracy with the actual rule/Luna reviewer, not an official GPT-4
judge claim. `check_scoring.py` covers contradictory and uncertain replies and
checks rule agreement against the actual first reviewed batch.

Data, raw output, labels and transport bundles stay under
`data/general_vqa_direct_20261004` and `outputs/general_vqa_direct`; none belongs in
Git. The current run state is in the latter directory. Authorized hosts are
4028-root GPU0/1/4/5, 4029 GPU1/2, 6403 GPU0/1 and K100 GPU0. Do not connect d4030.

Official source indices and frozen hashes are saved with the input manifests:
[MMMU](https://github.com/MMMU-Benchmark/MMMU),
[ScienceQA](https://github.com/lupantech/ScienceQA),
[POPE](https://github.com/RUCAIBox/POPE),
[HallusionBench](https://github.com/tianyi-lab/HallusionBench).
