# Completed Qwen3.5 formal evidence consolidation

`merge_screen.py` is a one-time CPU-only receipt for the completed four-shard `panel5_food_formal_4029_v1/qwen35_4b` run. All raw files, identity sidecars and record directories were copied from 4029 to the matching 4028 relative paths. The unchanged `formal_postprocess.validate_completed_shards` verified the source run on 4029 and again on 4028: 155136 unique task records, all four shards, original samples/method plans/source identities and completion hashes, no error records.

The new consolidation reused 24240 existing UNKNOWN-main/control preliminary labels after binding each one to its original row hash and provenance; it only ran the existing free matcher for the remaining 130896 prompt-matrix rows. Existing annotations and original raw records were retained unchanged. It made no API calls and did not run GPU inference.

Completed output: `outputs/annotations/acceleration_v4/qwen35_4029_complete_20260923_v1/{summary.json,SUMMARY.md,labels.jsonl}`. The script refuses an existing output directory; do not rerun this completed operation. The summary retains all conditions, stage totals, source hashes, prior-label file hashes and script/rule identities. Unresolved lexical matches remain unresolved, not errors. This is automatic preliminary matching, not human review or final should-abstain GT.

The independently completed MiniCPM UNKNOWN-controls stage was processed once with the existing `formal_stage_postprocess.py`; its output is `outputs/annotations/acceleration_v4/minicpm26_unknown_controls_20260923_v1`. LLaVA controls was still incomplete and was not processed.
