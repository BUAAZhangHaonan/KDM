# Luna semantic labeling of 9047 unresolved answers

User explicitly requested GPT-6 Luna, medium reasoning, as a Codex subagent on 2026-09-23. This exception is limited to the two already completed formal stages: LLaVA unknown_main (6880 unresolved) and Qwen2.5 unknown_controls (2167 unresolved). It does not restart DeepSeek or Ministral and does not alter GPU generation.

`prepare.py` verifies the original completed raw prefixes and each source-line SHA256 before preparing 9047 mappings. Exact equality of question and answer yields 1868 blinded groups, in 19 batches. The agent sees only question and answer, not image, gold class, source model, method, or correctness. Reused judgments are mapped only across exactly identical inputs, never approximate clusters.

The agent actually reads each group and authors four-way behavior judgments (`abstain`, `answer_uncertain`, `answer_assertive`, `invalid`) plus exact original evidence and endorsed-answer spans. This is an AI semantic judgment, not human review and not an abstention-justification GT. Ground truth is used only afterward by the unchanged Food scoring rule. Unsupported or genuinely ambiguous judgments can remain unresolved.

Immutable batch outputs are under `outputs/annotations/luna_semantic_v1/unresolved_9047_20260923_v1/results`. If the agent revises a judgment after review, versioned `corrections_v*.jsonl` keeps the revised record and reason without replacing the original batch. `finalize.py` requires complete batch coverage, exact input hashes, valid literal evidence spans, and unchanged original source prefixes. It applies explicitly recorded corrections, expands to the 9047 source rows and produces a separate full-stage hybrid table over all 26664 rows. Original lexical screening is retained for the other rows; they are not falsely described as Luna reviewed.

Run final validation and export once all batches and corrections are final:

    venv/bin/python workflows/luna_semantic_v1/finalize.py --check-only
    venv/bin/python workflows/luna_semantic_v1/finalize.py

The preparation and finalization scripts do not perform semantic classification. They only verify, map and score the agent-authored labels. A successful schema/span check does not prove every semantic judgment correct. This workflow makes no separate API requests; the requested Codex subagent consumes normal Codex usage.

## Provisional v1 quality hold

The first schema-valid export had overbroad endorsed-answer spans: most full answers were copied verbatim, including side-dish descriptions. This yielded zero added exact-alias correct matches and is not evidence that all new answers are wrong. `QUALITY_HOLD_v1.json` marks `validated_v1` provisional and superseded pending a second blinded Luna semantic span review of all 1868 groups. The original export and judgments remain intact. Versioned semantic corrections will feed separate `validated_v2`; no alias expansion or truth-aware answer selection is authorized.

## Accepted automatic annotation export

`latest.json` selects `validated_v3`: all 9047 original rows are covered, 9045 have valid behavior labels and 2 remain unresolved (DON'T and The Celti). The 1868 exact input groups received complete second-pass blinded span review by two GPT-6 Luna medium subagents, followed by a targeted 47-group review. Coverage, original source hashes and literal evidence spans pass; semantic judgment error is still possible. The first two exports remain preserved intermediate results and are not the active statistics. Original source responses, frozen aliases and initial lexical labels are unchanged.
