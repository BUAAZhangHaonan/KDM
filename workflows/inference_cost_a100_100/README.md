# Food eval100 A100 inference cost

This is the bounded 2026-10-07 cost run for nine frozen model registrations. It adds 2,700 measured replies: 100 common Food eval inputs × Guided VCD, IP-VCD and CDA × nine models. It does not produce task-performance scores.

`run.py` calls the existing `pipeline.sessions`, `decoding.generate` and `cda.generate_cda`. Guided VCD uses the frozen Food IP-VCD phrase on both guided branches; IP-VCD uses its original three branches. CDA uses the Food main-comparison UNKNOWN working point and all five original branches, including both null calibration branches. SOURCE.json and each model registration retain the full parameters and prompt identities.

All methods use batch 1, temperature 0, top_p 1, and a maximum of 32 new tokens. Eight models retain BF16. OneVision retains FP16. InternVL retains its registered explicit two-card layout. Every method of a model uses the same device allocation, environment, checkpoint and 100 IDs.

The sampling seed 20260929 is independently registered for this cost subset. The script sorts the actual Food eval2424 IDs, draws 100 without replacement with NumPy RandomState.choice, and draws eight disjoint warmup IDs from the remaining indices. It retains the existing per-input generation/noise seed function. Warmup8 through all three methods is stored separately; those 216 warmup replies are excluded from the 2,700 measured replies.

Timing begins after model loading. Each interval includes image reading, RGB conversion, prompt/session construction, actual generation, session/image cleanup and ending CUDA synchronization. CUDA is synchronized on every allocated device before the start and before the end. Loading and JSONL/CSV serialization are outside the intervals.

The first four measured inputs of each model are included in eval100. Their measured condition means produce the remaining-time estimate in pilot4.json. The worker continues all 100 inputs under the existing authorization. Exclusive GPU0/1 locks prevent this workflow from sharing a card with another KDM worker. Failures retain their traceback and registered source identity, then the supervisor proceeds to independent models. It does not repeat failed inputs or change parameters.

The initial supervisor used paired batches. At 2026-10-07 15:54:15 UTC, after Qwen3 had completed and Gemma remained active, `refill_scheduler.py` replaced only that owned CPU scheduling process so the free card could immediately take the next pending model. The original model workers, timing script and SOURCE.json retained their identities. `scheduler_transition.json` records the old supervisor command and starttick, the stopping event, and the surviving worker's unchanged PID before and after the transition. The finite refill queue covers only the original nine models; each model is dispatched once, with no repeat of completed or failed work.

The run resides at `/home/team/zhanghaonan/TAFFC/knowledge-deficit-mitigation/outputs/inference_cost_a100_100_20261007` on 6403. Its frozen registration and final results are copied to the same relative directory in the central 4028-root project. It was started with the project native311 CPU interpreter:

```bash
.environments/native311/bin/python workflows/inference_cost_a100_100/run.py prepare
nohup .environments/native311/bin/python -u workflows/inference_cost_a100_100/run.py supervise > outputs/inference_cost_a100_100_20261007/supervisor.log 2>&1 < /dev/null &
```

For the current run, inspect CURRENT_STATE.json, RUN_STATUS.json, and the per-model CURRENT_STATE.json files. `run.py summarize` collects current completed rows; `verify.py --partial` checks current real rows without initializing CUDA. After all models complete:

```bash
.environments/native311/bin/python workflows/inference_cost_a100_100/run.py summarize
.environments/native311/bin/python workflows/inference_cost_a100_100/verify.py
```

`condition_summary.csv` reports each condition mean and sample standard deviation with ddof=1, output-token means, and EOS/token-limit counts. `mean_ratios.csv` contains ratios of condition means. `per_input.csv` and each model's measurements.jsonl retain actual seconds, tokens, stopping state and physical GPU identities. `acceptance.json` verifies the complete roster, prompts, configs, seeds, raw counts and aggregate arithmetic. The cost interpretation must retain the generated-length and GPU-count differences.
