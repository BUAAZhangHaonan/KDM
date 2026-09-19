#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from kdm.visualization import (plot_reported_fates,plot_lexical_boundary,plot_decomposition,
                              plot_instruction_invariance,plot_semantic_matrix,plot_method_tradeoff)
from kdm.io import atomic_json,within

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out-dir',default='figures')
    p.add_argument('--matrix');p.add_argument('--tradeoff');p.add_argument('--source-note');p.add_argument('--examples',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve();out=within(root,a.out_dir);out.mkdir(parents=True,exist_ok=True);files=[]
    if a.examples:
        files+=plot_reported_fates(root/'data_example/reported_counts.json',out/'01_reported_abstentions')
        files+=plot_lexical_boundary(out/'02_lexical_boundary')
        files+=plot_decomposition(out/'03_exact_decomposition')
        files+=plot_instruction_invariance(out/'04_instruction_invariance')
        markers=['UNKNOWN','UNCLEAR','UNSURE','I cannot identify it']
        rows=[{'marker':x,'reference_marker':y,'retention':round(.08+.06*i+.09*j,2)} for i,x in enumerate(markers) for j,y in enumerate(markers)]
        atomic_json(root/'data_example/matrix_fixture.json',rows)
        files+=plot_semantic_matrix(rows,out/'05_matrix_layout','SIMULATED FIXTURE | Layout validation only; not KDM experimental results')
        rows=[{'method':'Direct','accuracy':.40,'retention':1.},
              {'method':'VCD','accuracy':.45,'retention':.25},
              {'method':'Candidate','accuracy':.46,'retention':.73}]
        atomic_json(root/'data_example/tradeoff_fixture.json',rows)
        files+=plot_method_tradeoff(rows,out/'06_tradeoff_layout','SIMULATED FIXTURE | Layout validation only; not KDM experimental results')
    if a.matrix or a.tradeoff:
        if not a.source_note:raise ValueError('Actual data provenance is required')
        if a.matrix:files+=plot_semantic_matrix(json.load(open(a.matrix)),out/'semantic_matrix',a.source_note)
        if a.tradeoff:files+=plot_method_tradeoff(json.load(open(a.tradeoff)),out/'method_tradeoff',a.source_note)
    if not files:raise ValueError('Select examples or explicit data files')
    atomic_json(out/'FIGURE_MANIFEST.json',{'files':files,'examples':a.examples})
if __name__=='__main__':main()
