#!/usr/bin/env python3
"""H200 reference and adversarial harness validation for Make-based Hopper tasks.

Builds isolated task copies, validates all default reference cases, and verifies
that an actual no-op CUDA submission cannot pass. Compilations may overlap;
GPU execution is serialized with the FA3 validation script. Results and complete
logs are saved after every task. This script never modifies solution.h in tasks.
Use test_fa3_references.py for the 60 def.py tasks and
 test_tk_hopper_oracles.py for independent TK math validation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]


from cuda_hercules.utils import task_source_hash as source_hash


def empty_solution(source):
    # Keep the exact public API, with a launched CUDA kernel doing no work.
    match=re.search(r'(?m)^(?:cudaError_t|void)\s+(?:Hopper\w+|Bf16Gemm|Fp8GemmScaled|Fp8Gemm|FusedRotary|solution_mqa_logits|BasedLinearAttention|HedgehogAttention|LinearAttention|Mamba2Forward)\s*\([^)]*\)\s*\{',source)
    if match is None: raise ValueError('Cannot find public solution entry point')
    signature=match[0].rstrip()[:-1].rstrip()
    includes='\n'.join(line for line in source.splitlines() if line.startswith('#include'))
    body='kh_empty<<<1,1>>>();'
    if signature.startswith('cudaError_t'):body+='return cudaGetLastError();'
    return includes+'\n__global__ void kh_empty() {}\n'+signature+'{'+body+'}\n'


def validate(task, output_dir, negative, positive=False):
    import yaml
    config=yaml.safe_load((task/'task.yaml').read_text())
    name=f'{task.parent.parent.name}_{task.name}'
    logs=[]
    import torch
    record=dict(task=config['task_id'],gpu=torch.cuda.get_device_name())
    env={**os.environ,'CUTLASS_DIR':str(ROOT/'reference_sources/cutlass'),
         'TK_DIR':str(ROOT/'reference_sources/ThunderKittens'),
         'DEEPGEMM_DIR':str(ROOT/'reference_sources/deep_gemm'),
         'KH_BENCHMARK':'10'}
    def run(cmd,cwd,timeout):
        result=subprocess.run(cmd,cwd=cwd,env=env,capture_output=True,text=True,timeout=timeout)
        logs.append('$ '+' '.join(cmd)+'\n'+result.stdout+result.stderr)
        return result
    try:
        with tempfile.TemporaryDirectory(prefix='hercules_hopper_') as temp:
            work=Path(temp)/'task'
            shutil.copytree(task,work,ignore=shutil.ignore_patterns('ref','test','*.o','__pycache__'))
            record['source_hash']=source_hash(work)
            ref=run(['make','-B','ref'],work,600)
            record['ref_compiled']=ref.returncode==0
            if ref.returncode: raise RuntimeError('reference build failed')
            if positive:
                build=run(['make','-B','test'],work,600)
                record['positive_compiled']=build.returncode==0
                if build.returncode:raise RuntimeError('positive control build failed')
                shutil.copy2(work/'test',work/'positive')
            if negative:
                sol=work/'solution.h'
                sol.write_text(empty_solution(sol.read_text()))
                build=run(['make','-B','test'],work,600)
                record['empty_compiled']=build.returncode==0
                if build.returncode:raise RuntimeError('empty control build failed')
            # Lock the GPU only after compilation.
            with (ROOT/'.cache/hopper_gpu.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                ref=run(['./ref'],work,600)
                record['reference_passed']=ref.returncode==0 and bool(re.search(r'\bPassed\b',ref.stdout))
                record['reference_cases']=[]
                sections=re.split(r'===\s*(.*?)\s*===',ref.stdout)
                for label,body in zip(sections[1::2],sections[2::2]):
                    timing=re.search(r'Ref time:[^\n]*min:\s*([\d.]+)\s*ms',body)
                    if timing:record['reference_cases'].append(dict(name=label,ref_min_ms=float(timing[1])))
                if not record['reference_passed']:raise RuntimeError('reference execution failed')
                if not record['reference_cases']:raise RuntimeError('reference emitted no per-case timings')
                if positive:
                    task_name=task.name
                    if task_name.startswith('tk_') and task.parent.parent.name=='class2':
                        small=['1','2','256']
                    elif task_name=='tk_rotary':small=[]
                    elif task_name.startswith('tk_'):small=['128','256','256']
                    elif task_name=='deepgemm_mqa_logits':small=['128','512']
                    elif task_name.startswith('61_'):small=['--m=128','--n=128','--k=256','--iterations=1']
                    elif task_name.startswith('54_'):small=['128','256','256']
                    elif task_name.startswith('57_'):small=['--m=128','--n=256','--k=2048','--groups=8','--iterations=1']
                    elif task_name.startswith('52_'):small=['128','512','256','256','1']
                    elif task_name.startswith(('48_','49_','50_','51_')):small=['128','256','256','1']
                    elif task_name.startswith('88_'):small=['--b=1','--h=2','--q=256','--k=256','--d=128']
                    else:small=['--m=128','--n=256','--k=256','--groups=8','--l=4','--iterations=1']
                    record['positive_args']=small
                    good=run(['./positive',*small],work,600)
                    record['positive_passed']=good.returncode==0 and bool(re.search(r'\bPassed\b',good.stdout))
                    if not record['positive_passed']:raise RuntimeError('positive implementation rejected')
                if negative:
                    # No-op fails before performance measurement in the first case.
                    bad=run(['./test'],work,600)
                    record['empty_rejected']=bad.returncode!=0
                    if not record['empty_rejected']:raise RuntimeError('EMPTY SOLUTION PASSED')
            record['passed']=True
    except Exception as error:
        record.update(passed=False,error=str(error))
    (output_dir/(name+'.log')).write_text('\n'.join(logs))
    print(record['task']+(' PASS' if record['passed'] else ' FAIL: '+record['error']),flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filter',default='')
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--no-negative',action='store_true')
    parser.add_argument('--positive',action='store_true',help='Also verify the supplied simple CUDA implementation on a diagnostic size')
    parser.add_argument('--resume',action='store_true',help='Reuse passing records only when their source hash still matches')
    parser.add_argument('--output',default='results/hopper_make_validation.json')
    args=parser.parse_args()
    import torch
    if torch.cuda.get_device_capability()!=(9,0):raise SystemExit('Requires an SM90 GPU')
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    log_dir=output.parent/output.stem;log_dir.mkdir(parents=True,exist_ok=True)
    (ROOT/'.cache').mkdir(exist_ok=True)
    tasks=[p.parent for cls in ['class1','class2'] for p in (ROOT/'tasks'/cls/'hopper').glob('*/Makefile') if args.filter in p.parent.name]
    reports=[]
    if args.resume and output.exists():
        previous={r['task']:r for r in json.loads(output.read_text())}
        pending=[]
        for task in tasks:
            task_id='/'.join(task.parts[-3:])
            cached=previous.get(task_id,{})
            if (cached.get('passed') and cached.get('source_hash')==source_hash(task)
                    and (args.no_negative or cached.get('empty_rejected'))
                    and (not args.positive or cached.get('positive_passed'))):
                reports.append(cached)
            else:pending.append(task)
        tasks=pending
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(validate,t,log_dir,not args.no_negative,args.positive) for t in sorted(tasks)]
        for future in as_completed(futures):
            reports.append(future.result());reports.sort(key=lambda r:r['task'])
            output.write_text(json.dumps(reports,indent=2)+'\n')
    print(f'{sum(r["passed"] for r in reports)}/{len(reports)} passed; {output}')
    if not reports or not all(r['passed'] for r in reports):raise SystemExit(1)

if __name__=='__main__':main()
