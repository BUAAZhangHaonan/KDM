# 九模型补充实验检查点

2026-10-03，CPU验收截止16:11；有限资源/进程快照16:13；新增d4030双卡实际生产16:12。生成量与已评分、复核、参考连接的验收量分列。旧五模型输出、评分及中文v1/英文正文冻结。

## Food完整注册范围

每模型53条件×2424图×9模型=1,156,248条；六模型实际注册原生SID另14,544条，总483条件/1,170,792条。Direct、原生VCD/DoLa/DeCo/SID、CDA视觉迁移原UNKNOWN单配置、IP-VCD/IP-M3ID各四表达属于主比较；矩阵/ref-off用于机制分析。SID探索交叉等不进入该范围。

| 实验 | 完整条件/目标条件 | 验收条数 |
|---|---:|---:|
| Direct、原生VCD、DoLa、DeCo、CDA | 45/45 | 109,080 |
| 原生SID（实际支持六模型） | 6/6 | 14,544 |
| IP-VCD四表达 | 36/36 | 87,264 |
| IP-M3ID四表达 | 36/36 | 87,264 |
| VCD 4×4机制矩阵 | 129/144 | 完整条件312,696；另有部分条件 |
| M3ID 4×4机制矩阵 | 129/144 | 完整条件312,696；另有部分条件 |
| VCD ref-off四表达 | 33/36 | 完整条件79,992；另有部分条件 |
| M3ID ref-off四表达 | 33/36 | 完整条件79,992；另有部分条件 |

主比较123条件/298,152条全完成，canonical、literal、弃权及参考未决0、重复及来源冲突0。冻结核心五模型机制200条件/484,800条复用。四扩展矩阵/ref-off已验收337,167/387,840条、124/160完整条件；九模型合计1,120,119条（95.67%）、447完整条件，尚未验收50,673条。部分条件仅记实际已验收回答，不提升为2424分母。

| 模型 | Food已验收/目标 | 矩阵已验收/96,960（扩展） | 尚未验收 |
|---|---:|---:|---:|
| qwen25vl | 130,896/130,896 | 冻结复用 | 0 |
| qwen35_4b | 128,472/128,472 | 冻结复用 | 0 |
| llava16_mistral | 130,896/130,896 | 冻结复用 | 0 |
| minicpm26 | 128,472/128,472 | 冻结复用 | 0 |
| gemma3_4b | 128,472/128,472 | 冻结复用 | 0 |
| qwen3vl | 130,896/130,896 | 96,960/96,960 | 0 |
| onevision | 130,896/130,896 | 96,960/96,960 | 0 |
| phi35 | 130,896/130,896 | 96,960/96,960 | 0 |
| internvl35_8b | 80,223/130,896 | 46,287/96,960 | 50,673 |

## 生成、长尾调度与ETA

16:13实际源快照：四扩展矩阵科学生成342,701，真未生成45,139；Phi/OneVision/Qwen3-VL各96,960完整，InternVL51,821/余45,139。八模型Food全部生成及评分/参考合并已完整。生成和验收截止不同，差值不作为实时Luna积压。

| 服务器/授权卡 | InternVL真余键（16:13） | 当前区间完成ETA（CST） |
|---|---:|---|
| 4028 0/4 | 3,923 | 18:12–18:31 |
| 4028 1/5 | 4,380 | 18:33–18:48 |
| 6403 0/1 slot0 | 4,352 | 18:06–18:08 |
| 6403 0/1 slot1，续接 | 6,409 | 新claim每方法不足100，待实际计时 |
| 4029 1/2 | 4,471 | 17:42–18:30 |
| d4030 0/1 | 6,921 | 19:45–19:55 |
| d4030 2/3，新增 | 6,632 | 新claim每方法不足100，待实际计时 |
| K100 0 | 8,051 | 18:22–20:06 |

区间为最近100条/方法和累计实测吞吐外推，回答长度会改变速率。此前23:30–次日02:00是新增两卡前的历史估计，不作为当前总ETA。新双卡首批足量计时后更新全量ETA。争取当日Food收尾，20:00检查实际已释放的10张3090并分担6403/K100尾段；Viz/dev/完整诊断缺项单列，不能据Food时刻宣称全部阶段齐全。

用户16:05新增授权：d4030 GPU0/1/2/3全部四卡，原最多两卡限制撤销；总资源10张3090、2张A100、1张RTX Pro 6000。实际新增2/3容量24,112MiB/卡、UUID/独占锁/原BF16 18+18/22GiB准入通过。动态最长源为6403旧slot1：完成当前输入封存5,868，真余13,135拆为d4030 2/3的6,656与6403原slot1的6,479，重复/缺失/foreign/已生成交集全0。真实PID123031/1917011分别在16:11启动；16:12生产19/42条status-ok、entry/owner/starttick/继承双锁和原runtime实核，无failure。原d4030 0/1 PID117527继续不动。新增registry仅扩允许卡、UUID及授权记录，方法/模型/精度参数均不变。

三次实际长尾拆分均已保存源prefix与原输出：K100→4028 0/4；旧4028 0/4→两个中央pair；6403旧slot1→d4030 2/3与原slot1。现八个互斥Intern owner全部运行，按真实退出事件继续调度，不做GPU周期轮询。

## 标注、参考及其他实验

| 项目 | 已完成 | 明确缺项 |
|---|---|---|
| 九模型Food候选排名及参考 | 43,632输入、已接受参考 | 无 |
| 九模型Food十次独立作答 | 436,320条、既有标签闭合 | 无 |
| 五模型Food开发集 | 32,320条，五个IP-VCD真实dev-selected配置 | IP-M3ID及扩展四模型dev选择 |
| 五模型VizWiz固定512 | 主表20条件=Direct/原生VCD/IP-VCD/CDA各五模型；机制11条件；共15,872条已决，166不可回答/346可回答 | 扩展四模型相应方法；现有512表尚无IP-M3ID/DoLa/DeCo/SID，不能称九模型全方法完成 |
| 扩展四模型VizWiz | eval Direct14,004条、独立140,040条 | 这些是3,501 eval问答/参考，不是方法结果或dev818 |
| 五模型四路归因 | 814位置、1049候选对、24路径/12例，代表471有效 | 历史来源未保存字段继续未知 |
| 扩展四模型四路 | 404代表首位、404自然扰动参考生成 | 诊断分岔/完整案例路径 |
| 五模型CDA原UNKNOWN trace | 12,120回答；Eq6/7误差0、权重和误差2.84e-14 | 缺完整词表向量，Eq4不能完整核对 |
| 原生M3ID五模型 | 0/12,120（另列） | 不在用户53条件清单外私自新建队列 |
| 硬件匹配VCD/IP/CDA时延 | 未新增正式benchmark | 未测量，不将Gemini预期比例写成结果 |

先精确QA/主菜字符与角色规则，再处理影响canonical/弃权/参考的真实边界。已验收主表/矩阵评分及参考未决0。最新11,744条仅35组新增真实Luna medium语义裁定（2,560中10组、9,184中25组），root全文复核35组，作用域纠正追加另存；此前root独立新裁定242组保持，root复核不重复计为新Luna标注。原始decision及纠正来源均保留。

新增5,120条与3,534+150条均已真实Luna判定、root全文复核、局部纠正评分并参考合并；矩阵累计本轮65组真实Luna新QA，root复核另列不重复计数。新4,332条已接续；6403接收sealed-prefix需同步既有CPU helper分支，原失败及恢复另存，GPU生成不改。按本批边界密度，最后一批生成后评分、Luna、复核和合并暂估30–60分钟；未来边界数量未知，新增Viz/dev/诊断的标注成本另计，不能据此宣称九模型全部实验今晚完成。

新增完整54条件指标已实算，旧70条件的所有计数及指标逐项保持。统一J=(C+TP)/N；不必要弃权率FP/(N-reference_positive)，另保留FP/N字段，不更改旧数据。主结果468组类bootstrap配对（2000次，seed20260929）已交付。九模型IP eval最高J观测与五模型真实dev-selected配置分别标记；完整主表包含所有注册baseline及必要结果。

## 实际来源与交付

- 当前矩阵合并：outputs/supplemental/remaining4/matrix_closed_union337167_20261003_1610/receipt.json；SHA256=6d180c320dcd9c5ca6b694099127a838e89aecd0eb4b09e982f5b6689c2323b8。
- 124条件指标：outputs/paper_core_20261002_dev_viz/mechanism_acceptance/matrix_full337167_20261003_1615/receipt.json；SHA256=27eeb8bb91c371f127bc89a05b38b85073c3901ad522b0fbff7a26e6c1e52475；仅CPU计算、生成增量0。
- 15:53资源：native_baselines/MATRIX_ACTUAL_RESOURCE_SNAPSHOT_20261003_155327.json；SHA256=c181b6ae85d4646387d4889e6082f7594a6b9a47c615c5d09b4ac6c4876f640a。
- 实际长尾handoff：native_baselines/INTERN_MATRIX_CENTRAL_PAIR04_ACTUAL_HANDOFF_20261003_1525.json；SHA256=9e18328390711ac27e4af685ec923a9cff325eef1c840b0ea8f3cafe88f7de8f。
- 第二次实际handoff：native_baselines/INTERN_MATRIX_CENTRAL_PAIR15_ACTUAL_HANDOFF_20261003_1550.json；SHA256=df6add3799e7916b0e059560f0d26679d0670e148e18fdd20500e3342f5aa5dd。
- 最新新增11,744条仍为原70完整条件，完整条件覆盖逐字段一致，70表直接复用，未重复重算。核对收据mechanism_acceptance/matrix_full70_reuse328363_20261003_1555/receipt.json。
- 全局CURRENT_STATE及before/after检查点已保存；检查点脚本只引用通过验收的不可变源，资源live数量单列，不冒充正式完成。
- 九模型主比较审阅包KDM_Nine_Main_Closed_20261003_0955_v4.zip（20.91MiB）于09:46:55 CST实际完成CRC、139个成员SHA/字节及离线数量核验；SHA256=bfd40f71ba3e8e946d63f04733db9503663caf40170f1254c51fb68cdae5b311。包内矩阵固定截止135,038条/16条件；包保持不变，最新矩阵另附检查点。

仅任务代码、小文档细粒度commit/push。大型raw、标签、权重、图像保留数据路径管理；未完成的Viz/选择/归因不记完成，也不新增探索矩阵。

2026-10-03 16:05接续：当前聊天单次20:00 CST检查任务已实际创建，automation_id=kdm-20、ACTIVE、仅一次。届时核实4028四卡/4029两卡/d4030四卡真实空闲容量与claim，将6403/K100实际未完成尾段互斥分担。d4030新增2/3卡16:06实查各24,112MiB空闲、无computePID；原0/1不动，新增pair准入与真尾分派正在执行，尚未提前计生产完成。

16:13实际资源证据：native_baselines/MATRIX_ACTUAL_RESOURCE_SNAPSHOT_20261003_161319.json SHA256=28fc2beaefb34b0f741b32f5d28d778a606d5dafdea7d81e7c278ccd66f4e11e；新pair交接INTERN_MATRIX_D4030_PAIR23_1610_ACTUAL_HANDOFF_20261003.json SHA256=ce069913b781f88aedbb4330243917fc544931426a4dd87bcf0eaac630542bf9；实际生产证据INTERN_MATRIX_D4030_PAIR23_1610_PRODUCTION_EVIDENCE_20261003_161255.json SHA256=006cba933fa53540669dc368c3955760170cd86a6eb97de1bb12fa5ac231d050。

16:18:39新增两队每方法均≥100真实回答，已有足量速度：d4030 2/3估19:22–19:25，6403续接slot1估18:07–18:17；其余4028 0/4估18:14–18:20、1/5估18:55–19:12；6403slot0估18:09–18:16；4029估17:53–18:24；d4030 0/1估18:39–19:39；K100估18:44–19:54。全部现有Food提取队列当前实测预计20:00前结束，仍按真实完成检查和必要重分。总科学已生成344,427、真余43,413，全为InternVL。资源source MATRIX_ACTUAL_RESOURCE_SNAPSHOT_20261003_161839.json SHA256=13957e25fb6892efd59c2dfb53d7f8c7814a991d8866b759002a48371ea97d59；新owner计时生产证据INTERN_MATRIX_D4030_PAIR23_1610_PRODUCTION_EVIDENCE_20261003_161831.json SHA256=3478384d6efc5064949fc9bd97817a9c501b64f9474845f62f2fdcc571b6e824。新4,332已完整接收，41组真实Luna边界正在裁定，100成员未决如实单列。


## 20:00检查与Food完整验收

更新时间：2026-10-03T12:24:29.629713+00:00。

Food九模型483条件、1,170,792条已全量验收；每条件2424，101×24，评分/弃权/参考未决0。四扩展矩阵387,840条已连接冻结参考。GPU生产19:04完成，20:07十三授权卡空闲、锁可取，无Food长尾迁移。

最后50,673条已收回并闭合：规则/精确QA优先，新增206唯一QA实际Luna medium和root完整复核。原始回执与纠正分别留存。

其余缺项：额外IP dev选择21,008键，先8题小批；6403双卡Intern pilot运行。扩展Viz方法和有限诊断/完整案例尚未完整，源索引见CURRENT_STATE。core5 Viz512已31条件15,872条，不能称九模型两个数据集全部完成。

下一步：完成Food审阅包离线验证与打包，依实测预算接续缺失dev/Viz。无重复Food生产，无新周期轮询。


## 20:52 IP开发选择接续

Food完整包已服务器全字段离线验收，并在本地校验ZIP CRC及152文件SHA：KDM_Nine_Food_Full_20261003_2035_v5.zip，23,722,821字节，SHA 0e3b7e52c8d3d39555d5220167742ada5bde7e1d37a3771c4a9dc26164bdd174。

新dev21008键：416真pilot全部评分闭合；首268真full分片闭合（完整404条件仍0）。20:52实生成6138、未生成14870；13授权卡已实际投入，原416与旧32320互斥。按各条件8题实测与加载成本估算11.76 GPU小时，低于24小时既有上限，预算已释放。Intern分至A100/4029/d4030四双卡路由，4028跑Phi/One，K跑One并用pidfd接续Q35，A100共享槽跑Q3/Gemma并由完成事件接续原core待队列。初测末尾Gemma约22:10，真实空闲卡出现时拆分其未写键。CPU/边界标注与选择另计，不将生成量冒充已选配置。

完整Viz方法与有限诊断尚未全部完成，不宣称九模型全阶段收口；旧core5主结果和正文不动。新增薄入口与已验CPU评分已细粒度推送，HEAD 845656a94eeda69e6cc381c4edccbc3aecc1dbd1。

### 2026-10-03 21:26 actual continuation

Food nine-model full panel remains accepted: 483 conditions / 1,170,792 rows. The verified 22.62 MiB full Food package has been delivered independently. New IP-dev generation at 21:14:55 was 14,029 / 21,008; InternVL generation complete. Closed CPU merge is 5,528 rows, zero duplicate/unresolved/reference changes, 9 / 52 complete 404 conditions. No new dev configuration selected before full coverage. Actual Luna medium has processed 59 new unique dev QA; root read and reviewed all 59 plus the separate pilot QA. The next 8,068-row scoring batch has final semantic authority ready, but its closure is not yet accepted. Remaining generation ETA was 21:48 before source-boundary redistribution; pending Viz and finite attribution remain separate. Linux pidfd event controller recovery preserved failed logs; K100 successor was actually observed. New selection code commit 87824ecb is pushed and exactly matches remote master.


2026-10-03 22:26 CST actual checkpoint (artifact labels may differ; use receipt UTC):
- Food full remains accepted: 483 conditions / 1,170,792 rows, zero missing/unresolved. Frozen core5 and manuscript unchanged.
- New IP dev complete: 21,008 unique rows / 52 x 404, zero pending. Real generation including pilots/loading/failed K attempt: 12.4342375584 GPUh, within original P2 24 GPUh. Gemma failed K original-runtime 192 unwritten keys were completed on verified original A100, without runtime/precision changes.
- Thirteen new selections plus five frozen core IP-VCD choices give eighteen actual dev-selected operating points. Existing same-config eval each n=2424; nine-model main69 table excludes guided matrices. Macro J: native VCD 40.5345%, original CDA 24.7479%, IP-VCD 46.0671%, IP-M3ID 43.9219%; IP-VCD wins native in eight of nine models. No significance claim follows from this mean alone.
- Additional Viz CPU accepted at this checkpoint: 4,160 unique keys = exact Direct 2,048 + full four-extension native VCD 2,048 + selected pilot 64. Core31 / 15,872 frozen rows preserved separately. Remaining selected main production originally registered 8,704 new keys, including real pilot136; live received progress kept in source ledger, not inferred from planned quantities.
- All actual new 93/190/74/13/103/103 Viz semantic batches completed with real Luna medium; root finite review only its actually read subsets, with original decisions preserved. Official answerability166/346 and continuous official answer scores are unchanged.
- Independent compact dev/J handoff: outputs/paper_core_20261002_dev_viz/nine_Food_dev_J_compact_handoff_20261003_2310.zip, 5,489,546 bytes, SHA6703896a78839284dfda53bed2b256ead6b5adba96e1ad68de874b107fc2bca3; local SHA/CRC checked. This supplements immutable full Food v5; Viz selected and finite extra diagnostic/cases are not yet claimed complete.
- Main Viz continues on original admitted routes with disjoint claims; finite diagnostic/case gaps remain within their original per-model budget and follow main production priority.


2026-10-03 23:26 CST GPU完成；23:33 CST CPU接续

- 九模型Food：483条件、1170792正式eval行，评分/参考连接/全量键全部闭合；原352条件/853248行保持冻结。
- 新Food dev21008行、52个404条件全部闭合；合旧核心5个IP-VCD选择共18个真实IP选点。主J比较69个同运行工作点。
- 新Viz25条件12800行已全部实际评分闭合，官方连续质量与完整回复原始分数保留，语义未决0。旧core31/15872原样；最终56合并与包验收正在进行，尚不把写入中包计交付。
- 21注册新Viz生成union10752键（native含精确复用），selected8704零缺失/重复。OneVision两次3090真实OOM保存失败来源；同原float16版本在A100三分支准入后只补真实缺键，未降参数。
- 五台服务器13授权GPU已实际空闲；本任务GPU剩余0。有限四扩展103诊断/11案例22路径均在原2880 GPU秒/模型预算内，175实际候选对代数闭合通过；首位缓存精确复用。
- 本轮Viz真正新Luna medium QA2341；root实际复核1791。814批发现误把错误/碎片/OCR当弃权后全批完整QA复核，原裁定不改、纠正追加并进入最后有效评分。
- 原UNKNOWN有限机制与dev_selected工作点分别标记；One第三案例、CDA全词表Eq4、历史9个replay来源未知字段保留具体缺项，不新增探索。
