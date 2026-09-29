#!/usr/bin/env python3
"""Build a full blinded review queue or merge explicitly submitted human decisions."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from kdm.io import within
from kdm.human_review import build_human_review_queue,merge_human_revisions


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('queue');q.add_argument('--records',nargs='+',required=True);q.add_argument('--annotations',required=True);q.add_argument('--errors',nargs='*',default=[]);q.add_argument('--out',required=True)
    q=sub.add_parser('merge');q.add_argument('--queue',required=True);q.add_argument('--decisions',nargs='+',required=True);q.add_argument('--out',required=True)
    a=p.parse_args();out=within(a.root,a.out)
    result=build_human_review_queue(a.records,a.annotations,out,a.errors) if a.mode=='queue' else merge_human_revisions(a.queue,a.decisions,out)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
