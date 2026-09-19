# Definition and implementation checks — 2026-09-19

This is an execution-time source check, separate from the supplied package's historical review claims. No intervention results were inspected to make these decisions.

## CDA calibration sign: resolved by the user before formal execution

Source: [ACL 2025 formal paper](https://aclanthology.org/2025.acl-long.479.pdf), printed p.9714, equations (6) and (7); downloaded file is registered by the asset downloader. The primary agent extracted the text and rendered PDF page 5; visual check artifact: `cache/paper_text/CDA_page5.png`.

Equation (6) prints `r=max(H_input-H_null,0)/H_null`. Equation (7) uses `w_p=r_p^2/(r_p+r_c)`, `w_c=r_c^2/(r_p+r_c)`. The supplied `src/kdm/cda.py` instead computes `max(H_null-H_input,0)/H_null`. These are different algorithms. In addition, literal equation (6) does not generally bound the ratios by one, so the residual `1-w_p-w_c` need not be nonnegative. The task asks both for the main-paper version and three positive score weights. These requirements cannot be asserted jointly for arbitrary inputs without a further definition.

The discrepancy was raised before formal execution. The user explicitly selected the printed main-paper equation and authorized negative residual weights, requiring them to be recorded (outputs/records/protocol_user_decisions_20260919.json). The implementation therefore uses input entropy minus null entropy and does not clamp the residual weight. Every CDA step records entropy, calibration ratios, all three weights and a negative-residual flag. At rp+rc=0 the continuous extension is (wp,wc,wa)=(0,0,1); zero null entropy remains an explicit undefined-condition error, without an added epsilon. The prior package formula and its earlier synthetic tests remain historical evidence only. No author erratum or verified official implementation was located in the bounded source search.

CDA's related-work contribution remains joint accurate answering and abstention, including calibrated uncertainty and its separately evaluated momentum extension. KDM's intended comparison omits momentum according to the task and maps context to visual input; that mapping is an adaptation, not an original CDA experiment.

## VCD noise

Read official `reference_repos/vcd/vcd_utils/vcd_add_noise.py`, commit `d6568ff81b8fd306a49e630df44f2db5c2300191`. The source uses a 1,000-step float32 sigmoid schedule from -6 to 6, beta range 1e-5 to 0.005, and pixel-dtype `randn_like`. The supplied backend used float64 random draws and schedule. The model adapter review corrects this numerical difference at the processed-pixel level with fixed step 500; no PIL conversion or clipping is added. Detailed numerical and native-interface validation belongs in MODEL_ADAPTER_REVIEW.

## M3ID

Read downloaded `M3ID_CVPR2024.pdf` (arXiv 2403.14003v1) section 4/equations (1)-(4), Algorithm 1, and section 5.2 offset discussion. The active log-probability form `l_c + 1[max p_c < 0.3] * expm1(0.02*(t+offset))*(l_c-l_u)` follows the frozen operator. The paper's VQA offset counts tokens between image and output; KDM's supplied protocol operationalizes it using the question/prompt token count. That distinction must remain explicit when documenting model chat templates. Instruction-preserving M3ID uses the neutral clear condition for its gate and neutral prompt length, as specified before intervention results.

## DoLa paper definition retained; official-code discrepancy confirmed

Read official `reference_repos/dola/transformers-4.28.1/src/transformers/generation/utils.py` lines 3566-3598, commit `805230e57e63ca561cb759994681b122ff6f81f0`. Its calls `F.kl_div(log_p, M)` compute `KL(M||p)`, averaging the two reverse directions and vocabulary entries for selection. The supplied decoder computes standard Jensen-Shannon divergence with `KL(p||M)`. This can select a different premature layer. The final-layer-minus-premature log-probability operator alone does not prove selection equivalence. Independent review confirmed a concrete layer-selection counterexample in `MATH_PIPELINE_COUNTEREXAMPLE.json`. The original paper, arXiv 2309.03883v2 section 2.2, explicitly defines this distance as Jensen-Shannon divergence: https://arxiv.org/html/2309.03883v2. The primary agent read equations and context in sections 2.1-2.3. Therefore the supplied standard-JS definition is retained as the paper definition; no official-code selection-equivalence claim is made. The paper also describes repetition penalty 1.2; the supplied KDM operator uses its separately fixed short-answer configuration, so the task must not be described as reproducing every original DoLa generation setting.

## DeCo inspected source

Read official `reference_repos/deco/transformers/generation/utils.py` candidate cutoff, anchor-layer choice and additive correction around lines 3152-3175, commit `c1a9129f43de016fb60d14619d98301b0426f961`. The code uses top-k then cumulative probability cutoff, the maximum candidate probability over early layers, and an additive raw-logit correction. Layer projection and architecture-specific normalization still require model-specific validation; synthetic operator agreement alone is insufficient.

## Scope and remaining validation

The source package, historical experiment numbers and synthetic fixtures remain identified separately from new real records. Incomplete checkpoint transfers or interfaces, human semantic adjudication, and unresolved method definitions are not treated as zero effects, failed scientific criteria, or completed experimental conditions.

## SID fixed-code reference and isolated sessions

The independent source review found and repaired two concrete implementation faults: a second reference session removed the first session's hooks, and reference selection used an external clean branch's attention rather than the reference forward's own shallow attention. The corrected implementation installs hooks only during its own forward and restores them on success or failure. Fixed official commit `127dd412fa6b61ab1c9babf6979ec4da98002438` uses aggregation index 2 and rank 100; the later ICLR 2025 paper version uses different settings. The fixed source settings remain unchanged.

On the registered LLaVA-1.5-7B checkpoint, the actual GPU check compared the official selection/mask source transplant with the port: 14 visits across two prompts and repeated/backtracked prefixes, 600 layer events, 576 visual positions reduced to the declared 100. Both evolving and fresh-reference logits matched exactly; causal masks and session isolation passed. The oracle shares the native backbone/hook transport, so this is not an end-to-end run of the whole legacy official fork. Other architectures require their own evidence or a concrete structural reason, not inference from this one success. Details: `SID_SOURCE_REVIEW.md` and `outputs/verification/llava15_7b_sid_reference.json`.

## Native generation settings

Qwen2.5-VL's checkpoint generation defaults include repetition_penalty=1.05. The first native interface check inherited it while the supplied KDM protocol uses raw-score greedy decoding; two of sixteen images differed. The native check now explicitly uses the same fixed greedy settings (do_sample=False, num_beams=1, repetition_penalty=1.0), retaining the checkpoint defaults and overrides in its record. All sixteen then matched. This corrects an unmatched reference check; it does not tune or change the formal decoding operator. The failed check remains preserved.

## 全量输入前置检查与派生记录修订

全量原生视觉计数发现固定SID rank100对部分既定输入未定义：Qwen2.5VL与GLM的VizWiz各30条、Qwen3VL的VizWiz47条；MiniCPM两版各2,088条横跨两个数据任务。方法适用性因此在读取普查或干预结果前按模型与完整数据任务登记，其他方法仍保留全部样本。不得将SID仅在大图子集上的结果当作完整条件结果。

计数检查增加EXIF方向核对时曾在同名派生count JSON/log路径重新执行写入，初版文件和初版脚本未单独保存或提交。当前记录是扩充脚本的真实重新观察，不能声称保留了初次执行的原始字节；该记录保全缺陷无法通过事后重建消除。具体可审计范围见FULL_VISUAL_COUNT_REVIEW。后续资源检查采用新版本路径，保存完整失败，不覆盖既有结果。正式科研原始生成尚未启动。
