"""Executable commands for the current research, independent of stage numbers."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path


def setup(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    for key,sub in [('HF_HOME','hf'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','xdg'),('MPLCONFIGDIR','mpl'),('NLTK_DATA','nltk')]:
        os.environ[key]=str(root/'cache'/sub)
    os.environ['HF_HUB_DISABLE_XET']='1'
    return root


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--root',default='/home/g203-4028/projects/knowledge-deficit-mitigation')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare-food');q.add_argument('--source',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('prepare-vizwiz');q.add_argument('--annotations',required=True);q.add_argument('--images',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('run');q.add_argument('--manifest',required=True);q.add_argument('--model-spec',required=True);q.add_argument('--model',required=True)
    q.add_argument('--mode',choices=['census','experiment','probe'],required=True);q.add_argument('--gpu',type=int,choices=[0,1,4,5],required=True)
    q.add_argument('--out',required=True);q.add_argument('--shard',type=int,default=0);q.add_argument('--n-shards',type=int,default=1)
    q.add_argument('--methods',default='vcd,m3id,dola,deco');q.add_argument('--marker',default='all')
    q=sub.add_parser('annotation-queue');q.add_argument('--records',nargs='+',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('select');q.add_argument('--census',nargs='+',required=True);q.add_argument('--manifest',required=True);q.add_argument('--annotations',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('analyze');q.add_argument('--records',nargs='+',required=True);q.add_argument('--annotations',required=True);q.add_argument('--aliases',required=True);q.add_argument('--out',required=True);q.add_argument('--vqa-normalizer')
    q=sub.add_parser('closed-probe');q.add_argument('--manifest',required=True);q.add_argument('--model-spec',required=True);q.add_argument('--model',required=True);q.add_argument('--gpu',type=int,choices=[0,1,4,5],required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('replay');q.add_argument('--records',required=True);q.add_argument('--model-spec',required=True);q.add_argument('--model',required=True);q.add_argument('--gpu',type=int,choices=[0,1,4,5],required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('mechanism');q.add_argument('--methods',default='vcd,m3id,dola,deco');q.add_argument('--records',required=True);q.add_argument('--model-spec',required=True);q.add_argument('--model',required=True);q.add_argument('--gpu',type=int,choices=[0,1,4,5],required=True);q.add_argument('--out',required=True)
    a=p.parse_args(argv);root=setup(a.root)
    from .io import within,read_jsonl,atomic_json,file_hash,Ledger,stable_hash,stable_seed
    out=within(root,a.out)
    if a.command=='prepare-food':
        from .data import food_from_existing
        food_from_existing(a.source,out);return
    if a.command=='prepare-vizwiz':
        from .data import vizwiz_manifest
        vizwiz_manifest(a.annotations,a.images,out);return
    if a.command=='annotation-queue':
        from .annotation import build_queue
        out.parent.mkdir(parents=True,exist_ok=True);print(build_queue(a.records,out));return
    if a.command=='select':
        from .annotation import validate_annotations
        from .pipeline import select_models
        from .human_review import validate_human_review
        ann=validate_human_review(a.annotations,a.census);samples=list(read_jsonl(a.manifest))
        from .protocol import validate_census_collection
        candidates=[r['key'] for r in json.load(open(root/'configs/kdm/models.json'))]
        validate_census_collection(a.census,samples,candidates)
        for path in a.census:
            for record in read_jsonl(path):
                if record['key'] not in ann or ann[record['key']].get('text')!=record['text']:
                    raise ValueError('Every census response, including lexical labels, requires matching unified annotation')
        ids=[r['id'] for r in samples]
        atomic_json(out,select_models(a.census,ann,ids));return
    if a.command=='analyze':
        from .analysis import annotated_rows,evaluate,export_csv,probe_summary
        from .annotation import validate_annotations
        ann=validate_annotations(a.annotations);aliases=json.load(open(a.aliases));normalizer=None
        if not a.vqa_normalizer and any(r['sample']['dataset']=='vizwiz' for path in a.records for r in read_jsonl(path)):
            a.vqa_normalizer=str(root/'src/kdm/models/official_vqa_normalizer.py')
        if a.vqa_normalizer:
            import importlib.util
            spec=importlib.util.spec_from_file_location('official_vqa',a.vqa_normalizer);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            ev=mod.VQAEval(None,None);normalizer=lambda text:ev.processDigitArticle(ev.processPunctuation(text))
        rows=annotated_rows(a.records,ann,aliases,normalizer)
        results=evaluate(rows);atomic_json(out,results);export_csv(results,out.with_suffix('.csv'))
        atomic_json(out.with_name(out.stem+'_probes.json'),probe_summary(rows));return
    # Set physical visibility before importing torch/model code.
    allocated=os.environ.get('CUDA_VISIBLE_DEVICES')
    if allocated:
        cards=allocated.split(',')
        if any(c not in {'0','1','4','5'} for c in cards) or str(a.gpu)!=cards[0]:
            raise ValueError('Requested GPU disagrees with allocated physical devices')
    else:os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    from .pipeline import make_backend,run_tasks,census_tasks,experiment_tasks,probe_tasks,closed_rank,sessions,json_safe
    from .decoding import DecodeConfig,replay
    from .prompts import MARKERS
    spec=json.load(open(a.model_spec))
    from .protocol import code_identity
    if spec.get('purpose')=='CPU_TEST_ONLY':
        if not out.is_relative_to(root/'outputs/verification'):
            raise ValueError('Synthetic model outputs must remain in outputs/verification')
        source_blobs={'software_fixture':True,'formal_evidence':False}
    else:
        from .protocol import validate_runtime
        validate_runtime(root,spec,a.model,os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        if (a.command=='run' and a.mode=='experiment') or a.command=='mechanism':
            methods=tuple(a.methods.split(','))
        elif a.command=='replay':
            methods=tuple(sorted({row['method'] for row in read_jsonl(a.records)} & {'vcd','m3id','dola','deco','sid'}))
        else:methods=None
        if methods is not None:
            from .protocol import validate_method_runtime
            validate_method_runtime(root,spec,methods)
        source_blobs=code_identity(root)
    backend=make_backend(spec,'cuda:0')
    identity={'backend':spec,'backend_spec_sha256':file_hash(a.model_spec),'schema':'kdm_current_v2','source_blobs':source_blobs}
    if a.command=='run':
        samples=list(read_jsonl(a.manifest));identity['manifest_sha256']=file_hash(a.manifest)
        cfg=DecodeConfig()
        if a.mode=='census':tasks=census_tasks(samples)
        elif a.mode=='experiment':
            markers=MARKERS if a.marker=='all' else (a.marker,)
            tasks=experiment_tasks(samples,tuple(a.methods.split(',')),markers)
        else:tasks=probe_tasks(samples);cfg=DecodeConfig(temperature=1.,top_p=1.)
        print(run_tasks(backend,a.model,tasks,out,identity,cfg,a.shard,a.n_shards));return
    if a.command=='closed-probe':
        from PIL import Image
        samples=list(read_jsonl(a.manifest));names=sorted(json.load(open(root/'configs/kdm/food_aliases.json')))
        if len(names)!=101:raise ValueError('Closed measurement requires all 101 frozen Food-101 classes')
        ledger=Ledger(out,{**identity,'manifest':file_hash(a.manifest),'names':names})
        for sample in samples:
            if sample['dataset']!='food101' or sample['split']!='eval':continue
            key=stable_hash([a.model,sample['id'],'closed'])
            if key in ledger.keys:continue
            with Image.open(sample['image_path']) as image:r=closed_rank(backend,image.convert('RGB'),sample['question'],names,sample['class'])
            ledger.add(key,{'status':'ok','model':a.model,'sample':sample,**r})
        return
    if a.command=='mechanism':
        from PIL import Image
        from .mechanism import measure_path
        ledger=Ledger(out,{**identity,'records':file_hash(a.records),'mechanism':'four_condition_shared_prefix'})
        for record in read_jsonl(a.records):
            if record['method']!='direct' or not record['guided'] or record['sample']['split']!='eval':continue
            if record.get('tokens') is None:raise ValueError('Mechanism requires matching backend token records')
            for method in a.methods.split(','):
                reference_markers=MARKERS if method in {'vcd','m3id','sid'} else (record['marker'],)
                for refmarker in reference_markers:
                    key=stable_hash([record['key'],method,refmarker,'mechanism'])
                    if key in ledger.keys:continue
                    with Image.open(record['sample']['image_path']) as image:
                        result=measure_path(backend,image.convert('RGB'),record['sample']['question'],record['tokens'],method,record['marker'],refmarker,record['seed'])
                    ledger.add(key,{'model':a.model,'sample':record['sample'],**json_safe(result)})
        return
    if a.command=='replay':
        from PIL import Image
        ledger=Ledger(out,{**identity,'records':file_hash(a.records),'replay':'direct_response_path'})
        records=list(read_jsonl(a.records))
        donors={}
        for row in records:
            if row['method']=='direct' and row.get('guided') and row['sample']['split']=='eval':
                donor_id=(row['sample']['id'],row['marker'])
                if donor_id in donors:raise ValueError('Duplicate direct replay donor')
                donors[donor_id]=row
        for record in records:
            if record.get('tokens') is None:raise ValueError('Replay requires token records from matching backend')
            if record['method'] not in {'vcd','m3id','dola','deco','sid'}:continue
            key=record['key']
            if key in ledger.keys:continue
            donor=donors.get((record['sample']['id'],record['marker']))
            if donor is None or not donor.get('tokens'):raise ValueError('Missing matching direct response token donor')
            cfg=DecodeConfig(**record['config']);task=record
            with Image.open(record['sample']['image_path']) as image:
                main,ref,_,_,_=sessions(backend,image.convert('RGB'),task,cfg,record['seed'])
                groups={'declared_marker_initial_tokens':sorted({backend.encode(text)[0] for text in MARKERS if backend.encode(text)})}
                try:trace=replay(main,ref,cfg,donor['tokens'],groups)
                finally:
                    if getattr(backend,'sid_control',None):
                        backend.sid_control.close();backend.sid_control=None
            ledger.add(key,{'status':'ok','model':a.model,'sample':record['sample'],'method':record['method'],'marker':record['marker'],'reference_marker':record['reference_marker'],'reference_guided':record['reference_guided'],'seed':record['seed'],'config':record['config'],'donor_key':donor['key'],'donor_tokens':donor['tokens'],'trace':json_safe(trace)})

if __name__=='__main__':main()
