#!/usr/bin/env python3
"""Save the root's five finite judgments after complete-response source review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from kdm.io import atomic_json, file_hash, within
from workflows.main_results.score import qah
from workflows.paper_core.score_native import save_rows
from workflows.supplemental.remaining11.score import rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue', required=True)
    p.add_argument('--luna', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    queue, luna = within(ROOT, a.queue), within(ROOT, a.luna)
    source = {r['qa_key']: r for _, r, _ in rows(queue)}
    if len(source) != 5:
        raise ValueError('Only the five actually reviewed QA are permitted')
    choices = {
        'd99d0bb7b79d11918ea5ad9c777fe7ce455f7f17a8e193dedd69d6c8e267aacc':
            (True, None, None, [], [], 'it is not possible to definitively identify the specific food item without more information',
             'Explicit inability to identify the requested specific food; no endorsed food name.'),
        'd611f5f0e522fd4754886e0f3fae4d53e11e88922f47d74f334fa324d9622dcc':
            (True, None, None, [], [], 'without more specific details, it is difficult to identify the',
             'Only a broad cuisine description; explicit identification difficulty is present before truncation, with no specific food answer.'),
        '51af5e91908ce36e5b58b1909a5f208e1c7ef0cd2b964a0e678f7e67b4b2e7a5':
            (False, 'multiple_primary', 'type of fried dough', ['fritters', 'donuts'],
             [('type of fried dough', 'description'), ('fritters', 'competing'), ('donuts', 'competing')],
             '"fritters" or "donuts"',
             'Two alternative same-level food names follow the broad fried-dough description; do not select the one canonical candidate by the target.'),
        '0c3786283bf5584fa34e129af55f748ca57f010491db5f08c02205fe294d8761':
            (False, 'dumplings', 'Steamed dumplings', ['Steamed dumplings'],
             [('Steamed dumplings', 'main'), ('baozi', 'description'), ('jiaozi', 'description')],
             'Steamed dumplings',
             'Dumplings is an explicit unique main category; uncertain possible subtypes do not withdraw it. Preserve the Steamed modifier in the literal name.'),
        '5a71976e920a0b18eb4443723e25910a5bd12be78dd3e48570abb17f28213c33':
            (False, 'explicit_outside_101', 'spinach and pork soup', ['spinach and pork soup'],
             [('spinach and pork soup', 'main'), ('Vietnamese Pho-style soup', 'description')],
             'spinach and pork soup',
             'The explicit primary name is spinach and pork soup; the parenthetical Pho-style characterization is a style description, not a second endorsed canonical name. Appears does not mean abstention.'),
    }
    if set(source) != set(choices):
        raise ValueError('The finite root QA identities differ')
    reviewed = []
    for line, old, line_sha in rows(luna):
        qkey = old['qa_key']
        packet = source[qkey]
        if (old['question'], old['answer']) != (packet['question'], packet['answer']) or qah(old['question'], old['answer']) != qkey:
            raise ValueError('Luna judgment differs from the complete source response')
        abstain, override, name, names, relations, span, reason = choices[qkey]
        if span not in old['answer'] or (name and name not in old['answer']):
            raise ValueError('The actual root evidence span is not in the generated answer')
        for member in packet['source_members']:
            proof = member['source']
            actual = next((r, sha) for n, r, sha in rows(within(ROOT, proof['path'])) if n == proof['line'])
            raw, sha = actual
            if (sha != proof['line_sha256'] or raw['identity'] != proof['identity'] or raw['key'] != proof['key']
                    or raw['model'] != member['model'] or raw['sample_id'] != member['sample_id']
                    or raw['noise_generated_r']['text'] != old['answer']):
                raise ValueError('Root source member byte/key/response proof differs')
        author = {'agent': '/root', 'model': 'gpt-6.1-sol', 'effort': 'max', 'call_id': '',
                  'session_id': '01a0f0e4-4fe9-7940-b763-e0b428e5d5d8'}
        reviewed.append({**old, 'abstain': abstain, 'canonical_override': override,
                         'endorsed_primary_names': names, 'endorded_primary_names': names,
                         'name_relations': [{'name': n, 'role': role, 'span': n} for n, role in relations],
                         'primary_answer_span': name, 'fullspan_primary_name': name, 'literal_full_name': name,
                         'literal_score_override': 0, 'needs_root': False, 'root_reason': reason,
                         'root_evidence_span': span, 'source_memberships': packet['source_members'],
                         'annotation_model': author['model'], 'annotation_effort': author['effort'],
                         'annotation_call_id': '', 'annotation_agent': '/root', 'root_review_author': author,
                         'source_luna_decision': old,
                         'source_luna': {'path': str(luna.relative_to(ROOT)), 'line': line, 'line_sha256': line_sha},
                         'root_review_performed': True})
    if len(reviewed) != 5 or len({r['qa_key'] for r in reviewed}) != 5:
        raise ValueError('Finite reviewed coverage differs')
    out = within(ROOT, a.out)
    out.mkdir(parents=True, exist_ok=False)
    save_rows(out / 'root_reviewed.jsonl', reviewed)
    atomic_json(out / 'receipt.json', {'actual_root_reviews': 5, 'raw_members': 5,
                                     'abstentions': sum(r['abstain'] for r in reviewed),
                                     'source_queue_sha256': file_hash(queue), 'source_luna_sha256': file_hash(luna),
                                     'root_review_sha256': file_hash(out / 'root_reviewed.jsonl'),
                                     'source_spans_and_members_verified': True, 'original_results_preserved': True,
                                     'author': author, 'review_saver_sha256': file_hash(Path(__file__))})
    print(json.dumps({'root_reviews': 5, 'raw_members': 5, 'pending': 0, 'source_proof_passed': True}))


if __name__ == '__main__':
    main()
