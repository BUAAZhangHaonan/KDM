# Census consumption provenance review — 2026-09-19

The census selection path previously checked task coverage and human-reviewed raw-record hashes, but did not authenticate raw JSONL identity fields against the original `.identity.json` sidecar and frozen protocol. A read-only in-memory reproduction showed that a complete task set with two different identities, no sidecar, and nonprotocol temperature/token limits passed `validate_census_collection`. Human-review hashes protected records after review; they did not establish that the records submitted for review came from the registered frozen run.

## Implemented boundary

`src/kdm/provenance.py::validate_census_inputs` is a pure CPU reader. Its freeze argument must come from the common `validate_freeze(root)` gate. It checks the sidecar digest, every row identity, frozen source blobs and freeze receipt hash, full backend spec and spec-file hash, original manifest hash, fixed greedy base/row configuration, deterministic seed and prompt, successful token records, task coverage and shard assignment. Source JSONL and sidecar digests are retained, with a before/after read check for concurrent file changes. It does not instantiate models or alter ledger files.

The CLI annotation-queue branch validates complete model subsets; select requires the full frozen panel; analyze validates census files among its inputs. A single ledger mixing census and other stages is rejected. Non-census annotation/report inputs retain their existing separate checks; this change does not claim a new general experiment-ledger validator. Raw paths are normalized under the declared project root before both validation and consumption, so a different working directory cannot redirect the later read.

Each successful census-consuming command saves an adjacent `.sources.json` receipt with original JSONL/sidecar digests, per-model grouping, output SHA256, and the annotation SHA256 when applicable. The blinded queue rows do not gain model/source identity fields. Select still independently requires the existing complete human-review receipt and actual external review history.

## Legal shard aggregation

The helper returns `models[model]` with `paths`, `identities`, `common_definition_sha256`, `record_count`, and `formal_evidence`, plus a top-level `sources` list describing each original ledger. Different source files for a model may have different identities only because their `shard` differs: all other definition fields, including `n_shards`, must agree. Every sample must belong to its declared deterministic shard. The union must contain exactly one copy of every guided and unguided task.

Same-identity fragments or concatenated same-identity records can use their unchanged original sidecar and pass when their union is complete. Directly concatenating different shard identities under a single sidecar fails; callers should pass the original shard paths separately. `protocol.validate_census_collection` and `pipeline.select_models` aggregate those paths by model. Selection denominators and abstention counts are calculated once per model/dataset, and `selection_sources` retains all source paths.

## Verification and limits

The focused suite passed 118 tests across provenance, protocol, pipeline, census scheduler, method gates, dataset plans, human review, and IO/data. It covers stale/missing/mixed identities, wrong source/spec/manifest/freeze/config/seed/prompt, shard mismatch, legal shard and same-identity aggregation, duplicate rejection, model subsets, successful selection counts, and all three CLI consumer gates. CLI gate unit tests inject an explicit synthetic freeze; they prove call ordering and input validation, not a real full-panel frozen-run integration.

A fresh actual CPU execution check also completed all 28 subprocess commands: `outputs/verification/cli_fixture_20260919T114921197859Z/CLI_REVIEW.json`. That fixture exercises mock census/experiment/probe generation and resumption, completeness checks, experiment-plus-probe annotation queue construction, replay/mechanism/complete-response mock paths and explicitly synthetic external-review fixtures. It does **not** execute census annotation or selection under the new formal freeze gate. Its success must not be described as acceptance of that full chain.

There is intentionally no mock-census CLI bypass around the complete formal freeze. The helper's explicit `allow_mock` support still requires verification input paths, the actual `MockBackend`, `CPU_TEST_ONLY`, and matching synthetic frozen definitions; formal output paths do not enable it. A real mock census→annotation/select CLI integration therefore remains unavailable without a separate valid fixture freeze. No real-model proof was fabricated to fill that gap. Formal census consumption uses the real validated freeze and requires all registered identities.

No GPU execution, source freeze, Git commit or push was performed for this change. Existing four-condition/native16 results are unrelated to this provenance software test.
