# Completed-stage preliminary Food matching

Each included shard has exact stage coverage. Later stages may still be running. This is zero-API whole-answer character matching, not final semantic labeling or abstention GT.

Stage: unknown_controls. Rows: 12120. Full five-model stage present: False.

| Model | Method | Kind | Rows | Matched correct | Matched incorrect | Exact abstention | Unresolved |
|---|---|---|---:|---:|---:|---:|---:|
| qwen25vl | cda_visual | instruction_preserving | 2424 | 948 | 161 | 357 | 958 |
| qwen25vl | instruction_m3id | instruction_preserving | 2424 | 856 | 54 | 1128 | 386 |
| qwen25vl | instruction_vcd | instruction_preserving | 2424 | 933 | 64 | 858 | 569 |
| qwen25vl | m3id | reference_instruction_removed | 2424 | 692 | 20 | 1504 | 208 |
| qwen25vl | vcd | reference_instruction_removed | 2424 | 248 | 2 | 2128 | 46 |

Unresolved answers were not counted as wrong. Input model/shard coverage is explicit in summary.json. The remaining prompt matrix, full model, semantic annotation, and final GT are not declared complete.
