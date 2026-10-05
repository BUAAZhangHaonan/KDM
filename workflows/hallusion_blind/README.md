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

## CPU接续与失败范围

 audit_status.py校验不可变分片、跳过逐字相同的跨机副本并保存CURRENT_STATE/RUN_STATUS；score_delta.py按完整问题、回复和参考复用已决标签，语义边界进入实际Luna批次。未决不记错，951完整且全标注闭合才给最终指标。

 stage_queue.py只等待已登记PID/starttick退出后运行明确登记的独立后继任务，按原GPU、精度、参数与EOS终止生成；无GPU轮询、自动重试或失败方法降参。它使用独立文件名，避免遮蔽Python标准库queue。跨机迁移先封存并验证真实未完成键。

真实OOM保留原失败claim和完整来源，受影响方法停止。已生成自然EOS的未封存完整行可由root使用原sealer验收。其他独立方法以显式、零已生成交集的清单在runs_continuation继续；输出路径不改变注册身份，合并时仍检查全局条件—样本键唯一。
