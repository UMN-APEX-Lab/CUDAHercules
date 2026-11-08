#!/usr/bin/env python3
"""Re-score a copied completed task using recorded kernel times; never execute it.

The original run remains immutable. A reference change must preserve the input
and operation contract and be validated independently before using this tool.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil


def rebase_metrics(record, references):
    rebased=copy.deepcopy(record)
    cases=rebased.get('per_size') or []
    if not cases:
        if rebased.get('speedup',0)>0:
            raise ValueError('Cannot rebase a scored record without per-case kernel times')
        return rebased
    names=[case['name'] for case in cases]
    if len(set(names))!=len(names) or set(names)!=set(references):
        raise ValueError('Reference case identities must match every recorded kernel case')
    for case in cases:
        kernel=float(case['kernel_min_ms']);reference=float(references[case['name']])
        if not all(math.isfinite(v) and v>0 for v in [kernel,reference]):
            raise ValueError('Kernel/reference times must be finite and positive')
        case['ref_min_ms']=reference
        case['speedup']=reference/kernel
    rebased['speedup']=math.exp(sum(math.log(case['speedup']) for case in cases)/len(cases))
    return rebased


def rebase_task(destination, source, cache_entry):
    destination=Path(destination).resolve();source=Path(source).resolve()
    if destination==source or source in destination.parents or destination in source.parents:
        raise ValueError('Rebase into a separate copied run to retain immutable history')
    old=json.loads((source/'result.json').read_text())
    current=json.loads((destination/'result.json').read_text())
    if current!=old:raise ValueError('Destination must contain an unmodified copy of the source result')
    if old.get('final_status')!='PASS':raise ValueError('Only completed PASS results can be rebased')
    if cache_entry['source_hash']==old['source_hash']:raise ValueError('Reference fingerprint has not changed')
    cases=cache_entry['per_size'];references={c['name']:c['ref_min_ms'] for c in cases}
    if len(cases)!=len(references):raise ValueError('Duplicate reference case identity')
    result=copy.deepcopy(old);history=destination/'reference_history'/old['source_hash'];history.mkdir(parents=True)
    for name in ['result.json','submissions.jsonl','judge_state.json']:
        if (source/name).exists():shutil.copy2(source/name,history/name)
    submissions=[]
    if (source/'submissions.jsonl').exists():
        for line in (source/'submissions.jsonl').read_text().splitlines():
            record=rebase_metrics(json.loads(line),references)
            record['evaluation_source_hash']=old['source_hash']
            if record.get('per_size'):record['scoring_source_hash']=cache_entry['source_hash']
            submissions.append(record)
    timed={r['index']:r['speedup'] for r in submissions if r.get('mode')=='full' and r.get('status')=='PASS'}
    final=rebase_metrics(old['final_reeval']['final'],references)
    final['in_loop_speedup']=timed.get(final['index'],final.get('in_loop_speedup',0))
    result['final_reeval']['final']=final
    result['final_reeval']['tried']=[]
    for record in old['final_reeval'].get('tried',[]):
        record=rebase_metrics(record,references)
        if record['index'] in timed:record['in_loop_speedup']=timed[record['index']]
        result['final_reeval']['tried'].append(record)
    if timed:
        best=max(timed,key=timed.get)
        result.update(in_loop_best_submission=best,in_loop_best_speedup=round(timed[best],4))
    result.update(source_hash=cache_entry['source_hash'],best_speedup=final['speedup'])
    result['reference_rebase']=dict(mode='reuse_recorded_kernel_latencies',
        previous_source_hash=old['source_hash'],scoring_source_hash=cache_entry['source_hash'],
        previous_best_speedup=old['best_speedup'],source_task_directory=str(source),
        original_result_sha256=hashlib.sha256((source/'result.json').read_bytes()).hexdigest(),
        kernel_execution_performed=False,model_calls_performed=0,
        correctness_basis='Retained prior PASS; new and old references independently cross-validated',
        previous_per_size=old['final_reeval']['final']['per_size'],new_reference_per_size=cases)
    (destination/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    if submissions:
        (destination/'submissions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in submissions))
    if (source/'judge_state.json').exists():
        state=json.loads((source/'judge_state.json').read_text())
        state['best_speedup']=result.get('in_loop_best_speedup',0)
        state['promoted']=state['best_speedup']>=state.get('target_speedup',1)
        state['summary']=(f'Reference rebased without execution; evaluations {state["evals_used"]}/{state["evals_limit"]}; '
                          f'best recorded timed PASS {state["best_speedup"]:.4f}x versus the new reference.')
        state['scoring_source_hash']=cache_entry['source_hash']
        (destination/'judge_state.json').write_text(json.dumps(state,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--task-id',required=True)
    parser.add_argument('--reference-cache',type=Path,required=True)
    args=parser.parse_args();name=args.task_id.replace('/','_')
    entry=json.loads(args.reference_cache.read_text())['benchmarks'][args.task_id]
    result=rebase_task(args.run_dir/name,args.source_run/name,entry)
    print(json.dumps(dict(task_id=args.task_id,previous_speedup=result['reference_rebase']['previous_best_speedup'],
        speedup=result['best_speedup'],kernel_execution_performed=False),indent=2))


if __name__=='__main__':main()
