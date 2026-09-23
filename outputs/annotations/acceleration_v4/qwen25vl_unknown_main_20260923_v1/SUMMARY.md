# Completed-stage preliminary Food matching

Each included shard has exact stage coverage. Later stages may still be running. This is zero-API whole-answer character matching, not final semantic labeling or abstention GT.

Stage: unknown_main. Rows: 14544. Full five-model stage present: False.

| Model | Method | Kind | Rows | Matched correct | Matched incorrect | Exact abstention | Unresolved |
|---|---|---|---:|---:|---:|---:|---:|
| qwen25vl | deco | main | 2424 | 724 | 23 | 1450 | 227 |
| qwen25vl | direct | main | 2424 | 740 | 27 | 1418 | 239 |
| qwen25vl | dola | main | 2424 | 930 | 73 | 667 | 754 |
| qwen25vl | m3id | main | 2424 | 756 | 27 | 1360 | 281 |
| qwen25vl | sid | main | 2424 | 1038 | 71 | 754 | 561 |
| qwen25vl | vcd | main | 2424 | 1002 | 69 | 769 | 584 |

Unresolved answers were not counted as wrong. Input model/shard coverage is explicit in summary.json. The remaining prompt matrix, full model, semantic annotation, and final GT are not declared complete.
