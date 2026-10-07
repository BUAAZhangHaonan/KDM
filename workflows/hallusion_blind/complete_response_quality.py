"""Exact complete replies for the supplemental Hallusion 128-token scorer."""
from workflows.general_vqa_direct.hallusion_scoring import (
    HallusionScoring, QUALITY_CODE, official_score,
)

RULE_VERSION = 'hallusion_complete_response_quality_v1'
COMPLETE_ABSTENTIONS = frozenset({
    'unknown', 'unknown.', 'unclear', 'unclear.', 'unsure', 'unsure.',
    'i cannot identify it', 'i cannot identify it.',
})


class CompleteResponseScoring(HallusionScoring):
    """Keep existing judgments; fill only whole-response exact matches."""

    def literal_binary_quality(self, sample, answer):
        existing = super().literal_binary_quality(sample, answer)
        if existing is not None:
            return existing
        record = self.source_record(sample)
        text = answer.strip().casefold()
        reference = record['gt_answer_details'].strip().casefold()
        if text in COMPLETE_ABSTENTIONS:
            label = 'correct' if self.reference_kind(sample) in {
                'no_answer', 'inconsistent_reference',
            } else 'unclear'
            rule = 'complete_abstention_with_full_reference_kind'
        elif text and text == reference:
            label, rule = 'correct', 'complete_reply_equals_full_reference'
        else:
            return None
        return {'quality_key': self.quality_key(sample, answer),
            'quality_label': label, 'official_correctness': QUALITY_CODE[label],
            'score': official_score(record, label), 'source': RULE_VERSION,
            'rule': rule}
