# Reproduction

## Inputs and frozen identities

`data/current/all.jsonl` is the 9,167-record sample inventory. `data/responses/census/manifest.json` binds the 16-model census streams to their original byte identities, model sidecars, row counts, and canonical copies. `data/annotations/census_baseline_auto/` retains the automatic label files and producer/source provenance. `data/responses/formal/manifest.json` binds the 21 source shards across five models to the ordered 853,248-row formal response bundle.

The frozen contract is stored at `data/provenance/frozen_contract/contract.json`. Its 95 registered files are stored as SHA-addressed blobs under `data/provenance/frozen_contract/blobs/`; `manifest.json` maps each historical path to its blob and SHA-256. The reusable validator in `src/kdm/frozen.py` checks the contract bytes, every proof blob, the original Git source commit, and all sixteen runtime proof chains. Formal generation and the general census, selection, and reporting commands use this validator. Runtime model, processor, checkpoint, method, and environment values remain those recorded in the frozen contract.

## CPU verification

Run from the repository root using the interpreter registered in each `configs/runtime/<model>.json`:

```sh
<environment_python> -m py_compile workflows/formal/generate.py workflows/formal/verify_provenance.py
<environment_python> workflows/formal/verify_provenance.py --full-contract
<environment_python> workflows/formal/generate.py --model qwen25vl --check-plan
```

Repeat `--check-plan` for `qwen35_4b`, `llava16_mistral`, `minicpm26`, and `gemma3_4b`. The check verifies canonical proof-blob hashes, the historical source commit, census-to-label bindings, model/runtime proofs, and full task coverage. It runs without loading a model or initializing a GPU.

## Registered generation

The execution entry validates the model environment, host, device allocation, checkpoint and inherited GPU locks. Launch through the worker so the selected physical GPU and lock are bound to the run:

```sh
scripts/worker.sh "$PWD" 0 <environment_python> workflows/formal/generate.py --model qwen25vl --execute --run-name formal_YYYYMMDD
```

Use the GPU assignment registered for the chosen host and model in `configs/runtime/hosts.json`. Each run name identifies a fresh output path. The runner records admission, historical source identities, current migration-code identity, canonical proof-manifest SHA, progress, completion and output identities under `outputs/records/formal_generation/` and `data/responses/generated/`.

## Five-model Food-101 results

The completed main-results panel is the five-model Food-101 evaluation: 853,248 generated responses, 352 exact condition cells, and 2,424 images per cell. The 16-model census remains a separate broader study component. The final score, both accepted GT tables, the 352-condition analysis, 372 paired comparisons, and control/figure outputs are bound by SHA-256 in their receipts and manifests.

Run commands from the repository root with the project virtual environment. Each command below writes generated files to a fresh directory outside the frozen inputs and formal result directories; replace `RUN` with a new empty path. The default published paths remain available when the output option is omitted.

```sh
RUN=/tmp/kdm-main-results-replay
mkdir -p "$RUN"

# Score the frozen formal response shards using the stable review registry.
venv/bin/python workflows/main_results/score.py --run --output-dir "$RUN/score"

# Rebuild both accepted-reference tables from frozen canonical sources and
# the frozen target-blind authority package, without replacing published GT.
MAIN_RESULTS_REFERENCE_GT_OUT="$RUN/reference_gt" \
  venv/bin/python workflows/main_results/reference_gt.py

# Derive copy-original-abstention controls from the frozen final score and GT.
venv/bin/python workflows/main_results/preserve_abstention_control.py \
  --out "$RUN/controls"

# Analyze the external score and GT outputs. The scorer validates source hashes
# while reading; bootstrap settings and metric definitions remain fixed.
venv/bin/python workflows/main_results/analysis.py \
  --score "$RUN/score/score_rows.jsonl.gz" \
  --reference-gt "$RUN/reference_gt/reference_gt.jsonl" \
  --uniform-reference-gt "$RUN/reference_gt/uniform_reference_gt.jsonl" \
  --reference-gt-manifest "$RUN/reference_gt/reference_gt_manifest.json" \
  --out "$RUN/analysis"

# Render from analysis tables into an independent figure/report directory.
venv/bin/python workflows/main_results/render_analysis.py \
  --input-dir "$RUN/analysis" --out-dir "$RUN/rendered"
```

`reference_gt.py` reads the five-model compact response/rank bundles, the accepted three-model GT copies, and `data/reference_gt/final_target_blind_authority/`. Its `authority_manifest.json` binds the review inputs, builder, parser, and canonical class inventory. `analysis.py` can also read the published frozen score/GT defaults; its optional file arguments support replay into an isolated output directory. `render_analysis.py` reads the condition and pair tables from `--input-dir` and writes figures, tables, receipts, and the updated report to `--out-dir`. The control stage reads the official score/GT paths and writes only to the supplied `--out` directory.

The commands above reproduce the Food-101 result from recorded responses. The registered model-generation and 16-model census entries support the broader study; supplementary mechanism measurements remain future work. To check syntax without running the analysis:

```sh
venv/bin/python -m py_compile workflows/main_results/score.py workflows/main_results/reference_gt.py workflows/main_results/preserve_abstention_control.py workflows/main_results/analysis.py workflows/main_results/render_analysis.py
```
