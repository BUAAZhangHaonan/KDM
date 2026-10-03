"""Append accepted extension matrices to the existing nine-model review package."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import pandas as pd

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def load(p): return json.loads(Path(p).read_text())
def write(p, d): Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n')

def verify(package):
    sys.path.insert(0, str(package/'scripts'))
    from verify_nine_review_package import verify_main, verify_references_and_selection, verify_support_panels
    manifest = load(package/'PACKAGE_MANIFEST.json')
    assert {p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()} == set(manifest)|{'PACKAGE_MANIFEST.json'}
    for name, proof in manifest.items():
        p=package/name; assert p.stat().st_size==proof['bytes'] and sha(p)==proof['sha256'],name
    frame,_=verify_main(package);verify_references_and_selection(package,frame);verify_support_panels(package)
    folder=package/'mechanism/extension4'
    scores=pd.read_parquet(folder/'scores.parquet')
    conditions=pd.read_parquet(folder/'conditions.parquet')
    qa=pd.read_parquet(folder/'qa.parquet')
    sources=pd.read_parquet(folder/'sources.parquet')
    assert len(scores)==387840 and len(conditions)==160
    assert not scores.duplicated(['condition_id','sample_id']).any()
    merged=scores.merge(conditions,on='condition_id',validate='many_to_one').merge(qa,on='qa_id',validate='many_to_one').merge(sources,on='source_id',validate='many_to_one')
    assert len(merged)==387840 and merged.groupby('model').size().eq(96960).all()
    assert merged.groupby(['condition_id','target_class']).size().eq(24).all()
    assert merged.groupby('condition_id').sample_id.agg(frozenset).nunique()==1
    assert merged[['canonical','literal','abstain','reference_G']].notna().all().all()
    assert not (merged.canonical.eq(1)&merged.abstain).any()
    assert merged.groupby(['model','sample_id']).reference_G.nunique().eq(1).all()
    refs=pd.read_parquet(package/'support/references/reference4.parquet')[['model','sample_id','reference_G']]
    connected=merged[['model','sample_id','reference_G']].drop_duplicates().merge(refs,on=['model','sample_id'],suffixes=('','_frozen'),validate='one_to_one')
    assert len(connected)==9696 and connected.reference_G.eq(connected.reference_G_frozen).all()
    expected={(m,method,kind):n for m in ('internvl35_8b','onevision','phi35','qwen3vl') for method in ('vcd','m3id') for kind,n in [('main',16),('reference_instruction_removed',4)]}
    assert conditions.groupby(['model','method','kind']).size().to_dict()==expected
    for r in qa.itertuples(index=False):
        assert hashlib.sha256(json.dumps([r.question,r.answer],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()==r.qa_key
    metrics=pd.read_csv(folder/'metrics.csv').set_index('condition_id')
    for cid,g in merged.groupby('condition_id'):
        row=metrics.loc[cid]; c=int(g.canonical.sum());a=int(g.abstain.sum());tp=int((g.abstain&g.reference_G).sum())
        assert row.N==2424 and row.C==c and row.A==a and row.TP==tp and row.FP==a-tp and abs(row.J-(c+tp)/2424)<1e-12
    proof=load(package/'PACKAGE_RECEIPT.json')
    assert proof['extension_mechanism_complete'] and proof['Food_full_rows']==1170792 and proof['Food_full_conditions']==483
    return {'passed':True,'Food_full_rows':1170792,'Food_full_conditions':483,'extension_rows':len(scores),'files':len(manifest),'offline_no_server_access':True}

def build(args):
    root=Path(args.root).resolve(); source=root/args.source; old=root/args.previous; out=root/args.output
    rec=load(source/'receipt.json'); assert rec['passed'] and rec['rows']==rec['unique_keys']==387840 and rec['complete_received_Food_conditions']==160
    for name,digest in rec['outputs'].items(): assert sha(source/name)==digest
    prior_manifest=load(old/'PACKAGE_MANIFEST.json')
    for name,proof in prior_manifest.items(): assert sha(old/name)==proof['sha256']
    assert not out.exists();shutil.copytree(old,out)
    folder=out/'mechanism/extension4';folder.mkdir()
    coverage=[json.loads(l) for l in (source/'condition_coverage.jsonl').read_text().splitlines()]
    assert len(coverage)==160 and all(c['full_input_coverage'] for c in coverage)
    conditions={c['condition_id']:c for c in coverage}
    from kdm.io import stable_hash
    from workflows.supplemental.remaining11.generate import condition_of
    qa={};sources={};compact=[];counts=defaultdict(Counter)
    with gzip.open(source/'score_rows.jsonl.gz','rt',encoding='utf-8') as stream:
        for line_no,line in enumerate(stream,1):
            r=json.loads(line); assert r['reference_complete'] and type(r['abstain']) is bool
            ctx=condition_of(r['model'],r['dataset'],r['stage'],{'sample':{'split':r['split']},**r})
            cid=stable_hash({'condition':ctx,'decode_config':r['config']}); assert cid in conditions
            qid=qa.setdefault(r['qa_key'],{'qa_id':len(qa),'qa_key':r['qa_key'],'question':r['question'],'answer':r['answer']})['qa_id']
            binding=(r['source_path'],r['source_identity'],r.get('source_claim'))
            sid=sources.setdefault(binding,{'source_id':len(sources),'raw_path':r['source_path'],'source_identity':r['source_identity'],'source_claim':r.get('source_claim'),'accepted_score_path':str((source/'score_rows.jsonl.gz').relative_to(root)),'accepted_score_sha256':rec['outputs']['score_rows.jsonl.gz']})['source_id']
            c=r['canonical_name_in_primary_score'];a=r['abstain'];g=r['reference_G']
            assert c in (0,1) and r['literal_extracted_name_score'] in (0,1) and type(g) is bool
            compact.append({'condition_id':cid,'sample_id':r['sample_id'],'target_class':r['target_class'],'canonical':c,'literal':r['literal_extracted_name_score'],'abstain':a,'reference_G':g,'qa_id':qid,'source_id':sid,'raw_line':r['source_line'],'accepted_score_line':line_no,'seed':r.get('seed'),'terminated':r.get('terminated')})
            counts[cid].update(N=1,C=c,A=int(a),TP=int(a and g),FP=int(a and not g),reference_positive=int(g))
    assert len(compact)==387840
    table=pd.DataFrame(compact)
    table.to_parquet(folder/'scores.parquet',index=False,compression='zstd')
    pd.DataFrame(list(qa.values())).to_parquet(folder/'qa.parquet',index=False,compression='zstd')
    pd.DataFrame(list(sources.values())).to_parquet(folder/'sources.parquet',index=False,compression='zstd')
    for c in coverage: c['decode_config_json']=json.dumps(c.pop('decode_config'),sort_keys=True)
    pd.DataFrame(coverage).to_parquet(folder/'conditions.parquet',index=False,compression='zstd')
    metrics=[]
    for cid,n in sorted(counts.items()):
        assert n['N']==2424
        metrics.append({**conditions[cid],**n,'W':2424-n['C']-n['A'],'FN':n['reference_positive']-n['TP'],'J':(n['C']+n['TP'])/2424,'accuracy':n['C']/2424,'precision':n['TP']/n['A'] if n['A'] else None,'recall':n['TP']/n['reference_positive'] if n['reference_positive'] else None,'precision_null_reason':None if n['A'] else 'no_abstentions','recall_null_reason':None if n['reference_positive'] else 'no_reference_positive','result_scope':'mechanism_only','selection':'none'})
    pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False)
    shutil.copyfile(source/'receipt.json',folder/'source_acceptance_receipt.json')
    write(folder/'projection_receipt.json',{'passed':True,'rows':387840,'conditions':160,'source':str(source.relative_to(root)),'source_receipt_sha256':sha(source/'receipt.json'),'original_score_objects_modified':False,'full_raw_tokens_probabilities_and_semantic_decisions_recoverable_by_accepted_score_SHA_and_one_based_line':True,'GPU_initialized':False})
    shutil.copyfile(Path(__file__),out/'scripts/append_complete_food.py')
    shutil.copyfile(root/args.status,out/'ACTUAL_TASK_STATUS.json')
    proof=load(out/'PACKAGE_RECEIPT.json');proof.update(extension_mechanism_complete=True,Food_full_rows=1170792,Food_full_conditions=483,extension_mechanism_rows=387840,extension_mechanism_conditions=160,created_utc=datetime.now(timezone.utc).isoformat(),extension_source_receipt_sha256=sha(source/'receipt.json'),actual_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    write(out/'PACKAGE_RECEIPT.json',proof)
    readme=(out/'README.zh.md').read_text();readme=readme.replace('扩展四模型相应160条件继续运行，实际去重进度见ACTUAL_TASK_STATUS.json。','扩展四模型160条件、387,840条已通过全量评分与参考验收，见mechanism/extension4/。')
    readme=readme.replace('扩展矩阵、扩展VizWiz方法面板、扩展诊断/完整案例仍有具体缺项','Food九模型483条件、1,170,792条已全部验收；开发选择、扩展VizWiz方法面板及扩展诊断/完整案例的具体缺项')
    readme=readme.replace('python scripts/verify_nine_review_package.py --package .','python scripts/append_complete_food.py --verify .')
    (out/'README.zh.md').write_text(readme)
    manifest={p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='PACKAGE_MANIFEST.json'}
    write(out/'PACKAGE_MANIFEST.json',manifest);result=verify(out)
    archive=out.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    result.update(archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive))
    write(out.parent/(out.name+'.VERIFICATION.json'),result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',type=Path)
    for name in ('root','source','previous','output','status'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.verify:print(json.dumps(verify(a.verify.resolve())))
    else:
        sys.path[:0]=[a.root,str(Path(a.root)/'src')];build(a)
