# HallusionBench blind panel

All951 official visual questions are the blind test; there is no Hall development split.
Nine checkpoints retain Food-frozen native parameters and one Food-dev-selected expression
for each IP method. Direct, native VCD, DoLa, DeCo, CDA visual transfer, IP-VCD,
IP-M3ID and the original SID where admitted comprise at most70 panels/66,570 answers.
Plain branches retain the exact existing benchmark prompt. Guided branches append
only the registered abstention instruction. No matrix, ref-off or independent trials.

Generation uses max_tokens=None and stops only on the checkpoint's real EOS.
Architectural/context/OOM errors are failures, never completed or truncated answers.
Each immutable chunk contains actual token IDs, selected probabilities, branch prompt
and token-input fingerprints, processed-noise tensor digest, seed, condition/source
identity, PID/start tick/boot ID ownership and a validated receipt.
Only source-bound naturally-ended legacy Direct outputs are reused; old capped
outputs remain historical and are regenerated.

Registration and artifacts: outputs/hallusion_blind_20261005/.
Primary scheduling excludes d4030; GPU locks use the existing registered worker.
Launch example from the registered source host:
python workflows/hallusion_blind/launch.py --models qwen25vl --phase main001 \
 --methods vcd dola deco cda_visual instruction_vcd instruction_m3id \
 --registry workflows/hallusion_blind/host_registry.json

CPU tests: tests/test_eos_only_decoding.py and tests/test_hallusion_blind.py.
