# Completed-stage preliminary Food matching

Each included shard has exact stage coverage. Later stages may still be running. This is zero-API whole-answer character matching, not final semantic labeling or abstention GT.

Stage: unknown_controls. Rows: 12120. Full five-model stage present: False.

| Model | Method | Kind | Rows | Matched correct | Matched incorrect | Exact abstention | Unresolved |
|---|---|---|---:|---:|---:|---:|---:|
| minicpm26 | cda_visual | instruction_preserving | 2424 | 416 | 213 | 0 | 1795 |
| minicpm26 | instruction_m3id | instruction_preserving | 2424 | 967 | 257 | 126 | 1074 |
| minicpm26 | instruction_vcd | instruction_preserving | 2424 | 1022 | 203 | 33 | 1166 |
| minicpm26 | m3id | reference_instruction_removed | 2424 | 894 | 201 | 471 | 858 |
| minicpm26 | vcd | reference_instruction_removed | 2424 | 815 | 149 | 610 | 850 |

Unresolved answers were not counted as wrong. Input model/shard coverage is explicit in summary.json. The remaining prompt matrix, full model, semantic annotation, and final GT are not declared complete.
