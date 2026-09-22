import json, os, tempfile, unittest
from pathlib import Path
from annotate import digest, file_hash
from score import score_run

class ScoreTests(unittest.TestCase):
    def test_full_denominators_missing_bounds_and_official_vizwiz(self):
        real_root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root=Path(directory); (root/'src').symlink_to(real_root/'src', target_is_directory=True)
            alias=root/'configs/kdm/food_aliases.json'; alias.parent.mkdir(parents=True)
            alias.write_text(json.dumps({'apple_pie':['apple pie'],'baklava':['baklava']}))
            out=root/'outputs/annotations/deepseek_v2/test'; out.mkdir(parents=True)
            raws=[{'key':'a','text':'apple pie','model':'m','guided':True,'sample':{'dataset':'food101','class':'apple_pie','split':'dev'}},
                {'key':'b','text':'baklava','model':'m','guided':True,'sample':{'dataset':'food101','class':'baklava','split':'dev'}},
                {'key':'c','text':'red','model':'m','guided':False,'sample':{'dataset':'vizwiz','split':'eval','gold':[{'answer':'red'}]*10}},
                {'key':'d','text':'UNKNOWN','model':'m','guided':False,'sample':{'dataset':'vizwiz','split':'eval','gold':[{'answer':'red'}]*10}}]
            raw_path=root/'raw.jsonl'; raw_path.write_text(''.join(json.dumps(r)+'\n' for r in raws))
            (out/'queue.sources.json').write_text(json.dumps({'sources':[{'path':'raw.jsonl','sha256':file_hash(raw_path)}]}))
            definition={'frozen_scoring_sha256':file_hash(root/'src/kdm/scoring.py'),
                'frozen_aliases_sha256':file_hash(alias),
                'frozen_vqa_normalizer_sha256':file_hash(root/'src/kdm/models/official_vqa_normalizer.py')}
            (out/'identity.json').write_text(json.dumps({'identity':'test','definition':definition}))
            labels=[]
            for r in [raws[0],raws[2],raws[3]]:
                label='abstain' if r['key']=='d' else 'answer_assertive'
                labels.append({'key':r['key'],'text':r['text'],'identity':'test','raw_record_sha256':digest(r),
                    'label':label,'answer_text':'' if label=='abstain' else r['text']})
            (out/'labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
            report=score_run(root,out)
            food=next(r for r in report['rows'] if r['dataset']=='food101' and r['split']=='all')
            viz=next(r for r in report['rows'] if r['dataset']=='vizwiz' and r['split']=='all')
            self.assertEqual((food['expected'],food['annotated'],food['unresolved']),(2,1,1))
            self.assertEqual((food['accuracy_lower'],food['accuracy_upper'],food['accuracy']),(0.5,1.0,None))
            self.assertEqual((viz['expected'],viz['accuracy'],viz['abstention_rate']),(2,0.5,0.5))
if __name__=='__main__':unittest.main()
