#!/usr/bin/env python3
"""Validate generated FA3 wrappers against a bounded, independent fp32 oracle.

Each task runs in its own process so illegal accesses cannot poison later tasks.
JSON reports include accuracy margins, negative controls, and H200 timings.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def worker(name, seeds, tune_splits=False):
    sys.path.insert(0, str(ROOT/'src'))
    import torch
    from cuda_hercules.eval import load_task_def, _build_call_args, check_correctness
    from cuda_hercules.compiler import compile_cuda_module
    from cuda_hercules.timing import time_cuda_function
    from cuda_hercules.utils import task_source_hash
    path = ROOT/'tasks/class2/hopper'/name
    task = load_task_def(str(path))
    build = ROOT/'.cache/ref_builds'/name
    build.mkdir(parents=True, exist_ok=True)
    fn = compile_cuda_module(cuda_source=str(path/'reference.cu'),
        function_signature=task['FUNCTION_SIGNATURE'],
        extra_include_dirs=[str(ROOT/p) for p in task['REFERENCE_EXTRA_INCLUDES']],
        extra_cuda_cflags=task['REFERENCE_EXTRA_CUDA_FLAGS'],
        build_dir=str(build), verbose=False)
    # Compilation may overlap; correctness and timing own the GPU exclusively.
    import fcntl
    gpu_lock = open(ROOT/'.cache/hopper_gpu.lock', 'w')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    report = dict(task=name, gpu=torch.cuda.get_device_name(), trials=[],
                  source_hash=task_source_hash(str(path)),scalar_args=dict(task['SCALAR_ARGS']))
    for seed in seeds:
        torch.manual_seed(seed)
        inputs = task['get_inputs']()
        outputs = task['get_outputs'](inputs)
        args = _build_call_args(task, inputs, outputs)
        fn(*args)
        torch.cuda.synchronize()
        expected = task['reference_fn'](inputs)
        ok, details = check_correctness(expected, outputs, task['TOLERANCES'], task['CHECKED_OUTPUTS'])
        margins = {}
        out = dict(outputs)
        for key, value in expected:
            tol = task['TOLERANCES'].get('per_output',{}).get(key, task['TOLERANCES'])
            error = (value.float()-out[key].float()).abs()
            margins[key] = dict(max_abs=error.max().item(),
                max_tolerance_fraction=(error/(tol['atol']+tol['rtol']*value.float().abs())).max().item(),
                normalized_rms=details[key].get('normalized_rms'))
        controls = {}
        for label, change in [('empty', lambda t: torch.full_like(t,float('nan'))),
                              ('zero', torch.zeros_like),
                              ('nan', lambda t: torch.full_like(t,float('nan')))]:
            bad=[(k,change(v)) for k,v in expected]
            controls[label] = not check_correctness(expected,bad,task['TOLERANCES'])[0]
        # Perturb one required output while retaining all the others.
        key = 'dV' if name.startswith('fa3_bwd') else 'lse'
        bad=[(k, v*0.9 if k==key and key=='dV' else (v+0.3 if k==key else v)) for k,v in expected]
        controls['dv_10pct' if key=='dV' else 'lse_bias'] = not check_correctness(expected,bad,task['TOLERANCES'])[0]
        if seed == seeds[0]:
            # Simulate lower-precision operands, including fractional mantissas.
            quantized=[(k, t.to(torch.float8_e4m3fn).to(t.dtype) if k in ['Q','K','V'] else t) for k,t in inputs]
            fp8=task['reference_fn'](quantized)
            controls['fp8_operands']=not check_correctness(expected,fp8,task['TOLERANCES'])[0]
            globals_=task['reference_fn'].__globals__
            if globals_['CAUSAL']:
                globals_['CAUSAL']=False
                try:
                    unmasked=task['reference_fn'](inputs)
                finally:
                    globals_['CAUSAL']=True
                controls['ignored_causal_mask']=not check_correctness(expected,unmasked,task['TOLERANCES'])[0]
        report['trials'].append(dict(seed=seed,correct=ok,margin=margins,negative_controls=controls))
        if not ok or not all(controls.values()):
            report['passed']=False
            return report
    stats=time_cuda_function(fn,args,num_trials=10)
    report.update(passed=True,ref_min_ms=stats.min_ms,ref_mean_ms=stats.mean_ms,
                  peak_memory_bytes=torch.cuda.max_memory_allocated())
    if tune_splits and 'fwd_split' in name:
        tuning={}
        for splits in [1,2,4,8,16,32]:
            task['SCALAR_ARGS']['num_splits']=splits
            outputs=task['get_outputs'](inputs)
            candidate=_build_call_args(task,inputs,outputs)
            fn(*candidate);torch.cuda.synchronize()
            ok,_=check_correctness(expected,outputs,task['TOLERANCES'],task['CHECKED_OUTPUTS'])
            if ok:
                tuning[splits]=time_cuda_function(fn,candidate,num_trials=10).min_ms
        report['split_tuning_ms']=tuning
        report['best_num_splits']=min(tuning,key=tuning.get)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filter',default='fa3_')
    parser.add_argument('--seeds',default='42,43,44')
    parser.add_argument('--output',default='results/hopper_fa3_validation.json')
    parser.add_argument('--worker')
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--tune-splits',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.worker:
        try:
            report=worker(args.worker,[int(x) for x in args.seeds.split(',')],args.tune_splits)
        except Exception:
            import traceback
            report=dict(task=args.worker,passed=False,error=traceback.format_exc())
        print('RESULT_JSON:'+json.dumps(report),flush=True)
        return
    reports=[]
    output=ROOT/args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    def validate(task):
        print(task.name,flush=True)
        log=output.parent/(task.name+'.log')
        try:
            cmd=[sys.executable,__file__,'--worker',task.name,'--seeds',args.seeds]
            if args.tune_splits:cmd.append('--tune-splits')
            proc=subprocess.run(cmd,
                                cwd=ROOT,capture_output=True,text=True,timeout=600,
                                env={**os.environ,'MAX_JOBS':'2'})
            log.write_text(proc.stdout+proc.stderr)
            lines=[x for x in proc.stdout.splitlines() if x.startswith('RESULT_JSON:')]
            report=json.loads(lines[-1].split(':',1)[1]) if lines else dict(task=task.name,passed=False,error=f'exit {proc.returncode}; see {log}')
        except subprocess.TimeoutExpired:
            report=dict(task=task.name,passed=False,error='600s timeout')
        print(task.name+(' PASS' if report['passed'] else ' FAIL: '+report.get('error','numeric mismatch')[-1000:]),flush=True)
        return report
    from concurrent.futures import ThreadPoolExecutor, as_completed
    tasks=[t for t in sorted((ROOT/'tasks/class2/hopper').glob('fa3_*')) if args.filter in t.name]
    if args.resume and output.exists():
        from cuda_hercules.utils import task_source_hash
        previous={r['task']:r for r in json.loads(output.read_text())}
        pending=[]
        for task in tasks:
            cached=previous.get(task.name,{})
            seeds=[t['seed'] for t in cached.get('trials',[])]
            if cached.get('passed') and cached.get('source_hash')==task_source_hash(str(task)) and seeds==[int(x) for x in args.seeds.split(',')] and (not args.tune_splits or cached.get('best_num_splits')):
                reports.append(cached)
            else:pending.append(task)
        tasks=pending
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(validate,t) for t in tasks]
        for future in as_completed(futures):
            reports.append(future.result())
            reports.sort(key=lambda r:r['task'])
            output.write_text(json.dumps(reports,indent=2)+'\n')
    print(f'{sum(r["passed"] for r in reports)}/{len(reports)} passed; {output}')
    if not reports or not all(r['passed'] for r in reports): raise SystemExit(1)

if __name__=='__main__': main()
