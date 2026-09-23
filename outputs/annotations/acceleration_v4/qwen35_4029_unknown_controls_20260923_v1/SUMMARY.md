# Completed-stage preliminary Food matching

Each included shard has exact stage coverage. Later stages may still be running. This is zero-API whole-answer character matching, not final semantic labeling or abstention GT.

Stage: unknown_controls. Rows: 12120. Full five-model stage present: False.

| Model | Method | Kind | Rows | Matched correct | Matched incorrect | Exact abstention | Unresolved |
|---|---|---|---:|---:|---:|---:|---:|
| qwen35_4b | cda_visual | instruction_preserving | 2424 | 603 | 215 | 0 | 1606 |
| qwen35_4b | instruction_m3id | instruction_preserving | 2424 | 1265 | 168 | 104 | 887 |
| qwen35_4b | instruction_vcd | instruction_preserving | 2424 | 1149 | 123 | 116 | 1036 |
| qwen35_4b | m3id | reference_instruction_removed | 2424 | 1056 | 53 | 854 | 461 |
| qwen35_4b | vcd | reference_instruction_removed | 2424 | 573 | 20 | 1478 | 353 |

Unresolved answers were not counted as wrong. Input model/shard coverage is explicit in summary.json. The remaining prompt matrix, full model, semantic annotation, and final GT are not declared complete.
