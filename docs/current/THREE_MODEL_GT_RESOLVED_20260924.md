# Three-model Food101 operational reference GT resolved

The user requested residual81 diagnosis and evaluation of census abstention precision using the completed cross-probe labels as ground truth. New accepted output: `outputs/analysis/three_model_joint_gt_20260924_v1/acceptance.json`. Original validated_v1 remains immutable and is retained as provenance; do not report its192 answers/81 questions as currently unresolved.

Actual GPT-6 Luna medium rejudged188 exact blind groups covering192 residual answers; root read the originals and labels and8 versioned corrections. Final residuals:180 incorrect,10 invalid,2 abstain,0 correct/unresolved. All145440 independent answers have automatic labels;14,544 model-question reference GTs have3,947 positive,10,597 negative,0 unresolved. Authoritative new questions: `resolved_questions_v2.jsonl`; the first resolved_questions.jsonl had stale auxiliary mean/boolean fields and is superseded, while GT/statistics were identical. Complete merged labels and source hashes are in the same output. Automatic review is not human review or a proof of intrinsic model inability.

Actual abstention precision across all4848 Food questions/model: Qwen25 unguided2/2, guided713/2841; Mini unguided6/7, guided215/384; LLaVA unguided7/11, guided563/947. Aggregate unguided15/20=75%, guided1491/4172=35.7383%. Eval-only values are separate in the report.

There are130 model-question conflicts where reference GT is positive but original census has a correct answer in at least one condition (81 unguided responses and76 guided responses). Retain response correctness and referenceGT independently; flag conflicts. Do not rewrite correct responses to incorrect or use the evaluated response to redefine its referenceGT. The81 original unresolved questions arose from uncertain semantic labels of82 independent answers, not these conflicts. No new inference or paid API was used in this resolution.

Remaining Qwen35/Gemma annotation is still a separate unfinished queue. Do not treat all five models as complete or resubmit this188-group residual batch.
