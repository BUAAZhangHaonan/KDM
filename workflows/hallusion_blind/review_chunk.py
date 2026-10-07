"""Display full QA and bind explicit actual-Luna judgments without labeling defaults."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from kdm.io import atomic_json, file_hash
from workflows.general_vqa_direct.hallusion_scoring import HallusionScoring
from workflows.general_vqa_direct.score import check_decision
from workflows.hallusion_blind.generate128 import now

parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['read', 'write'])
parser.add_argument('--batch', required=True)
parser.add_argument('--start', type=int)
parser.add_argument('--end', type=int)
parser.add_argument('--rows')
parser.add_argument('--manual')
parser.add_argument('--output')
parser.add_argument('--author')
parser.add_argument('--model')
parser.add_argument('--effort')
args = parser.parse_args()
assert args.batch.isdigit()
folder = ROOT / 'outputs/hallusion_blind128_20261006/annotations' / ('budget128_delta_' + args.batch)
source = folder / 'pending_review.jsonl'
requests = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
if args.rows:
    assert args.start is None and args.end is None
    positions = [int(item) for item in args.rows.split(',')]
    assert positions == sorted(set(positions))
else:
    assert args.start is not None and args.end is not None
    assert args.start <= args.end
    positions = list(range(args.start,args.end+1))
assert 1 <= len(positions) <= 32 and 1 <= positions[0] <= positions[-1] <= len(requests)
assert not (folder / 'ACTIVE.json').exists(), 'Already accepted; do not repeat'
hall = HallusionScoring(ROOT)
samples = {sample['id']:sample for sample in hall.samples}

def lines_of(text):
    lines = text.splitlines(keepends=True)
    assert ''.join(lines) == text
    return lines

def selected_span(text, interval):
    if not text:
        assert interval is None
        return ''
    assert isinstance(interval, list) and len(interval) == 2
    first, last = interval
    lines = lines_of(text)
    assert isinstance(first, int) and isinstance(last, int)
    assert 1 <= first <= last <= len(lines)
    span = ''.join(lines[first - 1:last])
    assert span.strip() and span in text
    return span

if args.action == 'read':
    for position in positions:
        request = requests[position - 1]
        sample = samples[request['source']['sample_id']]
        q_request = hall.quality_request(sample, request['answer'])
        print(json.dumps({
            'source_row_number':position, 'question':request['question'],
            'answer_lines':[{'line':i+1,'text':text} for i,text in enumerate(lines_of(request['answer']))],
            'gt_answer_details':request['gt_answer_details'],
            'need_quality':request['need_quality'], 'need_behavior':request['need_behavior'],
            'answer_type':sample.get('answer_type'),
            'expected_answer_type':q_request['expected_answer_type'],
            'reference_kind':q_request['reference_kind'],
        }, ensure_ascii=False))
    print(json.dumps({'complete_display':True,'range_1_based':[positions[0],positions[-1]],
                      'selected_rows_1_based':positions,
                      'displayed_rows':len(positions),'source_sha256':file_hash(source)}))
    raise SystemExit(0)

assert args.model == 'gpt-5.6-luna' and args.effort == 'medium'
assert isinstance(args.author, str) and args.author.startswith('/root/luna_')
assert args.manual and args.output and Path(args.output).name == args.output
manual_path = Path(args.manual)
manual = [json.loads(line) for line in manual_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
assert [row['source_row_number'] for row in manual] == positions
output = folder / args.output
assert not output.exists(), 'Keep original sources; choose a new output filename'
records = []
provenance = {'author':args.author,'model':args.model,'effort':args.effort,'call_id':None}
for item in manual:
    position = item['source_row_number']
    request = requests[position - 1]
    sample = samples[request['source']['sample_id']]
    record = {key:request[key] for key in ['joint_key','qa_key','quality_key','question','answer','gt_answer_details']}
    record['source_row_number'] = position
    record['quality_review'] = None
    record['behavior_review'] = None
    # Missing labels or evidence selections raise errors. No default judgment.
    if request['need_quality']:
        q = item['quality']
        label = q['label']
        assert label in {'correct','incorrect','unclear'} and q['reason'].strip()
        review = {key:request[key] for key in ['quality_key','question','answer','gt_answer_details']}
        review.update(provenance)
        review.update({'quality_label':label,'official_correctness':{'correct':1,'incorrect':0,'unclear':2}[label],
            'answer_evidence':selected_span(request['answer'],q['answer_lines']),
            'reference_evidence':request['gt_answer_details'],'reason':q['reason'],
            'selected_answer_lines_1_based':q['answer_lines']})
        hall.validate_quality(sample, request['answer'], review)
        record['quality_review'] = review
    else:
        assert item['quality'] is None
    if request['need_behavior']:
        b = item['behavior']
        assert b['reason'].strip()
        prediction = b['predicted_answer']
        if isinstance(prediction,str) and prediction.casefold() in {'yes','no'}:
            prediction = prediction.casefold()
        decision = {'label':b['label'],'abstain':b['abstain'],'predicted_answer':prediction,
            'answer_text':request['answer'],
            'evidence_span':selected_span(request['answer'],b['answer_lines']),
            'reason':b['reason'],'decision_source':'actual_luna_full_qa_chunk_v1'}
        check_decision(sample, request['answer'], decision)
        hall.check_behavior_prediction(sample, prediction)
        review = {key:request[key] for key in ['qa_key','question','answer']}
        review.update(provenance)
        review.update({'decision':decision,'selected_answer_lines_1_based':b['answer_lines']})
        record['behavior_review'] = review
    else:
        assert item['behavior'] is None
    records.append(record)
assert len({record['joint_key'] for record in records}) == len(records)
with output.open('x') as handle:
    for record in records:
        handle.write(json.dumps(record,ensure_ascii=False) + '\n')
receipt = {'schema':'actual_luna_explicit_chunk_binding_v1','written_utc':now(),
    'range_1_based':[positions[0],positions[-1]],'selected_rows_1_based':positions,
    'rows':len(records),'filename':output.name,
    'source_sha256':file_hash(source),'manual_sha256':file_hash(manual_path),
    'output_sha256':file_hash(output),'reviewer':provenance,
    'full_qa_and_exact_evidence_validated':True,'active':False}
atomic_json(output.with_suffix('.receipt.json'),receipt)
print(json.dumps(receipt,ensure_ascii=False))
