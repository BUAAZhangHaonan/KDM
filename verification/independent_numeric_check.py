"""Numerical review independent of the implementation's log-sum-exp derivation."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from kdm.probability import event_decomposition,instruction_preserving,log_probs

def main():
    rng=np.random.default_rng(94813);group_error=0.;instruction_error=0.;count=1000
    for _ in range(count):
        p,q=rng.dirichlet(np.ones(12),size=2);alpha=float(rng.uniform(.05,2));e=np.arange(12)<5
        # Long double direct powers constitute the independent oracle.
        pl=p.astype(np.longdouble);ql=q.astype(np.longdouble)
        weights=pl**(1+alpha)/ql**alpha;weights/=weights.sum()
        expected=np.log(weights[e].sum()/weights[~e].sum())-np.log(pl[e].sum()/pl[~e].sum())
        result=event_decomposition(np.log(p),np.log(q),e,alpha,0)
        group_error=max(group_error,abs(float(expected)-result['actual_change']),abs(result['closure_error']))
        g,c,r=rng.dirichlet(np.ones(12),size=3)
        computed,_=instruction_preserving(np.log(g),np.log(c),np.log(r),alpha,0)
        oracle=g.astype(np.longdouble)*(c.astype(np.longdouble)/r.astype(np.longdouble))**alpha;oracle/=oracle.sum()
        instruction_error=max(instruction_error,float(np.max(np.abs(np.exp(computed)-oracle))))
    result={'case_count':count,'oracle':'long_double_direct_power_sums',
            'maximum_group_log_odds_error':group_error,'maximum_instruction_probability_error':instruction_error,
            'passed':group_error<1e-10 and instruction_error<1e-10}
    target=Path(__file__).with_name('NUMERIC_REVIEW.json');target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
