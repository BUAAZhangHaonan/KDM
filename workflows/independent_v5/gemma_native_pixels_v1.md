# Gemma exact native pixel bridge v1

The persistent CPU child runs the frozen Gemma image processor in the exact copied TF5.5.3/Torch2.6 environment. It receives the same image objects and preprocessing arguments, and returns tensor bytes without numerical conversion. Only preprocessing runs on CPU; no model weights are loaded or offloaded. The new engine keeps its existing BF16 model cast.

In the new versioned Gemma engine entry point, before constructing either Generator or LLM:

```python
import gemma_native_pixels_v1
bridge = gemma_native_pixels_v1.install()
# Construct Generator and LLM normally. The local Gemma processor class delegates
# preprocess to the same child for both the external and actual vLLM processor.
# Bind bridge.identity and the CPU receipt hash into validation/production identity.
# Close when all work in this process finishes:
bridge.close()
```

The original Gemma engine failure remains untouched. Bind all of `bridge.identity` (source SHA, exact package profile, native image/backend source hashes, processor files/configuration and native Python path) and the bridge audit receipt in the new entry's provenance. Install in the process that owns vLLM input preprocessing; the pipe is private to that process. It is not a service or network endpoint. The single connection serializes threaded requests and never restarts a failed child.

The CPU audit passed all16 original interface images: every external native tensor hash matched, and actual vLLM multimodal processing produced identical BF16 pixels and expanded prompt tokens. It made32 native child calls. Receipt: `outputs/records/independent_v5/gemma_native_pixels_cpu_v1/audit.json`. Processing the32 calls and comparing16 images took16.07s; process/module startup brought total audit time to51.43s. No GPU inference was performed. The new GPU greedy/EOS/ten-request/cache/attention validation is still required before production.

Files are versioned; no shared package, model, frozen source or old result was edited. IPC uses length-delimited pickle only on inherited private subprocess pipes; do not connect these functions to untrusted input or an external service.

## Versioned integration and one-shot launch

`gemma_native_pixels_entry_v2.py verify ...` installs the bridge before constructing Generator/LLM and injects the full immutable binding into `check.json` (or failure evidence). It closes the helper in a finally block. The binding contains bridge/environment identity, the16-image CPU receipt SHA, CPU audit code SHA, this entry SHA, and current verifier/generator/review/preparer code SHAs.

`gemma_native_pixels_entry_v2.py generate ...` requires an existing reviewed GPU check. In addition to the ordinary `reviewed_entry.py` fields, `review.json` must include `native_pixels_bridge_binding_sha256` equal to the GPU check and `native_pixels_entry_sha256` equal to this entry. It verifies that current code, native environment and CPU evidence match the GPU check. The full bridge binding is then included both in `independent_verification` and `execution` of the production identity and raw sidecar. No review is fabricated by this wrapper; generate cannot be admitted from the CPU audit alone.

`gemma_native_pixels_handoff_v2.py` is a single-use pidfd successor. It watches only MiniCPM PID228003 and its recorded GPU1 child228429, then verifies the48480-answer completion and raw SHA. It launches only the following GPU1 audit through the approved worker and never launches production or retries. Watcher PID242849 was observed in `waiting_pidfd` state. State is at `outputs/records/independent_v5/gemma_native_pixels_handoff_v2/status.json`; audit log is `audit.log` in that directory.

```
bash workflows/food_closed_v3/worker.sh "$PWD" 1 \
 /home/team/lvshuyang/anaconda3/envs/ST_LORA/bin/python -u \
 workflows/independent_v5/gemma_native_pixels_entry_v2.py verify \
 --model gemma3_4b \
 --native-reference outputs/records/independent_v5/native16_v2/gemma3_4b/reference.json \
 --out outputs/records/independent_v5/engine_verify_gemma_pixels_v1/gemma3_4b
```

The command is already scheduled by the single-use watcher; do not launch a duplicate. GPU inference has not yet been audited at integration time. If it completes, the check still requires separate review before generation.

The first integration entry was superseded before any GPU audit: its CPU binding self-check compared native tuples directly with JSON lists. Entry v2 canonicalizes only identity metadata through JSON. The actual pixels, bridge implementation, and successful16-image CPU audit are unchanged. The old watcher status records `superseded_before_launch`. The corrected binding was validated without GPU; hash `850f2a0674068a6902819a7afd189d0e73021d3667a84f7857689f4099b46eff`.
