# Real Viz completion and InternVL admission

The K100 OneVision VizWiz independent-answer producer sealed all 34,930 tail
answers after the accepted first 80, covering 3,501 inputs with ten attempts.
The original producer subsequently failed in its optional final CUDA-peak
Python-callable RPC. Its original failure, outputs and source bytes are retained.
Generation completion does not imply semantic scoring or reference completion.

`vllm_viz_metadata.py` forwards every scientific RPC unchanged. It bypasses only
that optional peak RPC, after the original progress and every part receipt show
complete generation and no current key. Unmeasured peak fields remain null. It
does not enable unsafe serialization, alter sampling, replay the first 80, or
retry failed scientific calls. All six source files accepted at `16c5b1b9` remain
unchanged. The separate accepted vLLM engine cohort remains explicit.

Three actual CPU contract tests passed: unchanged scientific arguments/results,
null optional metadata after completion, and refusal before completion. Qwen3-VL
then produced a real 520-answer part using the wrapper; its original verifier
passed. Actual source, admission, launch, identity, part and test-log hashes are
bound in `outputs/supplemental/remaining4/dispatch_20261001_1600/event_2203/source_commit_manifest_v7_metadata_wrapper_actual520.json`.

Both original Phi owners on 4029 naturally exited. The independent InternVL
registry uses physical GPUs 1 and 2, the exact copied original checkpoint and
Python/runtime/processor/template, bfloat16, the original 18+18 layer placement,
and 22 GiB placement limits. It requires 22,528 MiB free per card before loading.
Its actual eight-input mixed-method gate passed at 22:07 CST, then the explicit
22,879-key remainder started. The eight-input generation measurement was 23.278
seconds; it is a pilot measurement rather than a certified full-queue ETA.

CPU and actual GPU receipts live under the same run's
`cpu_prepare_4029_intern_2140/` and `event_2203/4029/`. No registered precision,
batch, prompt, method parameter or frozen five-model result changed.
