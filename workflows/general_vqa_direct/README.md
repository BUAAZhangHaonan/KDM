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
