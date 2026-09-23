# Completed-stage preliminary Food matching

Each included shard has exact stage coverage. Later stages may still be running. This is zero-API whole-answer character matching, not final semantic labeling or abstention GT.

Stage: unknown_main. Rows: 12120. Full five-model stage present: False.

| Model | Method | Kind | Rows | Matched correct | Matched incorrect | Exact abstention | Unresolved |
|---|---|---|---:|---:|---:|---:|---:|
| qwen35_4b | deco | main | 2424 | 1118 | 91 | 669 | 546 |
| qwen35_4b | direct | main | 2424 | 1182 | 114 | 498 | 630 |
| qwen35_4b | dola | main | 2424 | 1191 | 107 | 510 | 616 |
| qwen35_4b | m3id | main | 2424 | 1256 | 141 | 142 | 885 |
| qwen35_4b | vcd | main | 2424 | 1186 | 120 | 85 | 1033 |

Unresolved answers were not counted as wrong. Input model/shard coverage is explicit in summary.json. The remaining prompt matrix, full model, semantic annotation, and final GT are not declared complete.
