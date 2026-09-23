# Completed-stage preliminary Food matching

Each included shard has exact stage coverage. Later stages may still be running. This is zero-API whole-answer character matching, not final semantic labeling or abstention GT.

Stage: unknown_controls. Rows: 12120. Full five-model stage present: False.

| Model | Method | Kind | Rows | Matched correct | Matched incorrect | Exact abstention | Unresolved |
|---|---|---|---:|---:|---:|---:|---:|
| llava16_mistral | cda_visual | instruction_preserving | 2424 | 699 | 161 | 1 | 1563 |
| llava16_mistral | instruction_m3id | instruction_preserving | 2424 | 791 | 208 | 402 | 1023 |
| llava16_mistral | instruction_vcd | instruction_preserving | 2424 | 861 | 223 | 95 | 1245 |
| llava16_mistral | m3id | reference_instruction_removed | 2424 | 631 | 125 | 1106 | 562 |
| llava16_mistral | vcd | reference_instruction_removed | 2424 | 336 | 35 | 1800 | 253 |

Unresolved answers were not counted as wrong. Input model/shard coverage is explicit in summary.json. The remaining prompt matrix, full model, semantic annotation, and final GT are not declared complete.
