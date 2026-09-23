# Native independent-prompt fixed16 exporter

`export_native16.py` produces the dedicated native reference needed by the new independent-answer fast-engine gate. It uses all 16 original interface images and exactly `task_prompt(question, guided=False, attempt=True)`.

The source checkpoint, processor file hashes, weight sizes, source adapter hashes, manifest, original fixed16 and frozen protocol are checked. Actual execution uses a project-local exact package-version copy on6403 physical GPU0 and inherited `food_closed_v3/worker.sh` GPU lock. The original runtime specification is retained in the reference; relocation is recorded separately. No original source or runtime file is edited.

For each sample it records the original chat template, unexpanded and actual expanded tokens, actual processor tensor shape/dtype/content hashes, native greedy generation including EOS, and original backend full-vocabulary selected-token log probabilities. The original backend greedy argmax must agree at every prefix of the native generation. Every generation retains BF16 and the32-token budget.

An output directory must be fresh. Failures retain partial rows, progress, and full error trace and are not automatically retried. A successful reference is engineering evidence for the native side only; it does not validate the fast engine or authorize production by itself.

Example for qwen25vl (project root as working directory):

```
bash workflows/food_closed_v3/worker.sh "$PWD" 0 \
 "$PWD/.environments/mprisk-tf553/bin/python" \
 workflows/independent_v5/export_native16.py --model qwen25vl \
 --out outputs/records/independent_v5/native16_v1/qwen25vl
```

Qwen3.5-4B and LLaVA use `.environments/native311/bin/python`, an exact copy of the already registered torch2.9/transformers5.17 native environment. Other three use the exact `.environments/mprisk-tf553` copy. Shared source models/environments are read-only. Copy relocation changes only the project-local Python links and pyvenv.cfg, not installed package contents.

The initial exporter version is preserved for the successful Qwen2.5 reference. `export_native16_v2.py` corrects its admission dependency set to match the existing native verifier: `hf.py`, `backbone.py`, and additionally `remote.py` for MiniCPM. SID is not called by independent native generation. The full frozen receipt still checks all source identities. MiniCPM v1 failed before generation because the initial exporter incorrectly required a stale SID hash in its model specification. Its failure is retained, and the implementation correction is recorded under `native16_v2/implementation_fix.json`; scientific parameters are unchanged.

For Qwen3.5 and LLaVA, `export_native16_v3.py` additionally requires `PYTHONNOUSERSITE=1` and verifies all nine package versions of the exact native311/base311 copy. This prevents6403 user-site packages from shadowing project-local numpy/Pillow. The first native311 Qwen3.5 launch was interrupted with zero generated rows when this isolation defect was found; its record remains under `native16_v2/qwen35_4b/interrupted_environment_isolation.json`. The corrected run uses `native16_v3` and preserves all model/input/generation settings.
