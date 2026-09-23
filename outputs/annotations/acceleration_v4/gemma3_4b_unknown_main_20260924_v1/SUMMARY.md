# Completed-stage preliminary Food matching

Each included shard has exact stage coverage. Later stages may still be running. This is zero-API whole-answer character matching, not final semantic labeling or abstention GT.

Stage: unknown_main. Rows: 12120. Full five-model stage present: False.

| Model | Method | Kind | Rows | Matched correct | Matched incorrect | Exact abstention | Unresolved |
|---|---|---|---:|---:|---:|---:|---:|
| gemma3_4b | deco | main | 2424 | 585 | 16 | 0 | 1823 |
| gemma3_4b | direct | main | 2424 | 690 | 20 | 0 | 1714 |
| gemma3_4b | dola | main | 2424 | 650 | 18 | 0 | 1756 |
| gemma3_4b | m3id | main | 2424 | 689 | 20 | 0 | 1715 |
| gemma3_4b | vcd | main | 2424 | 647 | 15 | 0 | 1762 |

Unresolved answers were not counted as wrong. Input model/shard coverage is explicit in summary.json. The remaining prompt matrix, full model, semantic annotation, and final GT are not declared complete.
