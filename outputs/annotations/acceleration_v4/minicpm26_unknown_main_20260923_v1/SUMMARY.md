# Completed-stage preliminary Food matching

Each included shard has exact stage coverage. Later stages may still be running. This is zero-API whole-answer character matching, not final semantic labeling or abstention GT.

Stage: unknown_main. Rows: 12120. Full five-model stage present: False.

| Model | Method | Kind | Rows | Matched correct | Matched incorrect | Exact abstention | Unresolved |
|---|---|---|---:|---:|---:|---:|---:|
| minicpm26 | deco | main | 2424 | 955 | 248 | 216 | 1005 |
| minicpm26 | direct | main | 2424 | 953 | 249 | 200 | 1022 |
| minicpm26 | dola | main | 2424 | 732 | 317 | 26 | 1349 |
| minicpm26 | m3id | main | 2424 | 964 | 251 | 130 | 1079 |
| minicpm26 | vcd | main | 2424 | 964 | 204 | 42 | 1214 |

Unresolved answers were not counted as wrong. Input model/shard coverage is explicit in summary.json. The remaining prompt matrix, full model, semantic annotation, and final GT are not declared complete.
