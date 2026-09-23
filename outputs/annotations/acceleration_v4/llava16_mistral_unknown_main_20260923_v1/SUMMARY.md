# Completed-stage preliminary Food matching

Each included shard has exact stage coverage. Later stages may still be running. This is zero-API whole-answer character matching, not final semantic labeling or abstention GT.

Stage: unknown_main. Rows: 14544. Full five-model stage present: False.

| Model | Method | Kind | Rows | Matched correct | Matched incorrect | Exact abstention | Unresolved |
|---|---|---|---:|---:|---:|---:|---:|
| llava16_mistral | deco | main | 2424 | 668 | 172 | 504 | 1080 |
| llava16_mistral | direct | main | 2424 | 753 | 206 | 482 | 983 |
| llava16_mistral | dola | main | 2424 | 758 | 207 | 471 | 988 |
| llava16_mistral | m3id | main | 2424 | 859 | 227 | 138 | 1200 |
| llava16_mistral | sid | main | 2424 | 849 | 197 | 16 | 1362 |
| llava16_mistral | vcd | main | 2424 | 880 | 232 | 45 | 1267 |

Unresolved answers were not counted as wrong. Input model/shard coverage is explicit in summary.json. The remaining prompt matrix, full model, semantic annotation, and final GT are not declared complete.
