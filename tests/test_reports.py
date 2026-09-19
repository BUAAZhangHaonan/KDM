from kdm.reports import behavioral_validity,method_comparison

def test_report_joins_and_method_retention():
    rows=[];probes=[];closed=[]
    for i in range(4):
        sample={'id':str(i),'cluster':str(i),'dataset':'food101','split':'eval'}
        common={'model':'m','sample':sample,'marker':'UNKNOWN','reference_marker':'UNKNOWN','guided':True}
        d=0. if i<2 else 1.;abs_=i<2
        for method,credit,abst,kind,refg in [('direct',d,abs_,'main',True),('vcd',float(i!=0),False,'main',True),('instruction_vcd',float(i>0),i==0,'instruction_preserving',False)]:
            rows.append({**common,'method':method,'correct':credit,'label':'abstain' if abst else 'answer_assertive','kind':kind,'reference_guided':refg})
        probes.append({'model':'m','sample_id':str(i),'n_full_correct':0 if i==0 else 10,'mean_correctness':0 if i==0 else 1})
        closed.append({'model':'m','sample':sample,'gold_rank':2 if i==0 else 1})
    validity=behavioral_validity(rows,probes,closed,20);assert sum(r['n'] for r in validity)==4
    out=method_comparison(rows,probes,closed,20)[0]
    assert out['behavioral_retention_new']['value']==1
    assert out['behavioral_retention_original']['value']==0
    assert out['corrected_answer_retention']['value']==1
    assert out['copy_corrected_answer_retention']['value']==0
