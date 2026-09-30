"""Summarize only verified finite four-view exports."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    args=p.parse_args()
    root=args.input
    source_paths=[root/"four_view_pair_margins.csv",root/"four_view_events.jsonl",root/"verification.json"]
    assert json.loads(source_paths[2].read_text())["status"]=="complete"
    pairs=list(csv.DictReader(source_paths[0].open()))
    counts=Counter()
    for row in pairs:
        if row["panel"]=="representative101" and row["a_nonmarker_b_marker_proxy"]=="True":
            counts["proxy_pairs"]+=1
            e=float(row["interaction"])
            label="negative" if e < -1e-8 else "positive" if e > 1e-8 else "zero"
            counts["E_"+label]+=1
            if row["pair_in_guided_support"]=="True":
                counts["both_guided"]+=1
                counts["both_guided_"+label]+=1
    events=[json.loads(line) for line in source_paths[1].read_text().splitlines()]
    panels=Counter(row["panel"] for row in events)
    arg=Counter()
    for row in events:
        if row["panel"]=="representative101":
            d=row["distributions"]
            arg["native_IP_argmax_changed"]+=d["native"]["argmax"]!=d["ip"]["argmax"]
            arg["native_on_Sg_IP_argmax_changed"]+=d["native_on_guided_support"]["argmax"]!=d["ip"]["argmax"]
            arg["guided_IP_argmax_changed"]+=d["guided"]["argmax"]!=d["ip"]["argmax"]
            arg["common_empty"]+=row["support"]["common_size"]==0
    assert panels["representative101"]==505
    result={"representative_pair_counts":dict(counts),"position_counts":dict(panels),
            "first_token_argmax_counts":dict(arg),"n_representative_model_inputs":505,
            "source_sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limits":"First-token proxies and local argmaxes are not semantic abstention rates or population causal effects."}
    output=root/"aggregate_summary.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
