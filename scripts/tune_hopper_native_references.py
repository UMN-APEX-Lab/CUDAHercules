#!/usr/bin/env python3
"""Compare semantically identical CUTLASS configurations in isolated copies.

Selection runs reference targets; adopted configurations must subsequently pass
test_hopper_references.py's positive/negative controls. GPU runs are serialized.
"""
from pathlib import Path
import argparse
import concurrent.futures
import fcntl
import json
import os
import re
import shutil
import subprocess
import tempfile
import posixpath

ROOT=Path(__file__).resolve().parents[1]
# Explicit replacements never alter input layouts, block-scale granularities,
# output types or the public operation. A failed build is retained as evidence.
CONFIGS={
 '51_':('gett_kernel.cuh',[
   ('c1x1',{'TileShape, Shape<_1,_2,_1>':'TileShape, Shape<_1,_1,_1>'}),
   ('n256c1x1',{'Shape<Shape<_128>, Shape<_128>, Shape<_64>>':'Shape<Shape<_128>, Shape<_256>, Shape<_64>>','TileShape, Shape<_1,_2,_1>':'TileShape, Shape<_1,_1,_1>'})]),
 '55_':('55_hopper_mixed_dtype_gemm.cu',[
   ('n256',{'Shape<_128,_128,cute::Int<TileShapeK>>':'Shape<_128,_256,cute::Int<TileShapeK>>'})]),
 '57_':('57_hopper_grouped_gemm.cu',[
   ('pingpong',{'run<Gemm>(options)':'run<GemmPingpong>(options)'})]),
 '62_':('62_hopper_sparse_gemm.cu',[
   ('c1x1',{'Shape<_1,_2,_1>':'Shape<_1,_1,_1>'}),
   ('n256c1x1',{'Shape<_128,_128,_128>':'Shape<_128,_256,_128>','Shape<_1,_2,_1>':'Shape<_1,_1,_1>'})]),
 '63_':('63_hopper_gemm_with_weight_prefetch.cu',[
   ('n128',{'Shape<_64,_64,_128>':'Shape<_64,_128,_128>'})]),
 '67_':('67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu',[
   ('c1x1',{'Shape<_1,_2,_1>':'Shape<_1,_1,_1>'}),
   ('c2x1',{'Shape<_1,_2,_1>':'Shape<_2,_1,_1>'})]),
 '68_':('68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling.cu',[
   ('c1x1',{'Shape<_1,_2,_1>':'Shape<_1,_1,_1>'}),
   ('n256c1x1',{'Shape<_128,_128,_128>':'Shape<_128,_256,_128>','Shape<_1,_2,_1>':'Shape<_1,_1,_1>'})]),
 '69_':('69_hopper_mixed_dtype_grouped_gemm.cu',[
   ('n64',{'Shape<_128,_16,cute::Int<TileShapeK>>':'Shape<_128,_64,cute::Int<TileShapeK>>'}),
   ('n128',{'Shape<_128,_16,cute::Int<TileShapeK>>':'Shape<_128,_128,cute::Int<TileShapeK>>'})]),
 'deepgemm_mqa_logits':('deepgemm_mqa_logits.cu',[
   ('kv128math256',{'BLOCK_KV  = 256':'BLOCK_KV  = 128','NUM_MATH = 512':'NUM_MATH = 256'}),
   ('kv_stages5',{'NUM_KV_STAGES = 3':'NUM_KV_STAGES = 5'})]),
 'tk_bf16_gemm':('bf16_h100_gemm.cu',[]),
}

def measure(item):
    task,source,name,replacements,base_ref=item
    record=dict(task=task.name,configuration=name,replacements=replacements,base_ref=base_ref,release=True)
    logs=[]
    try:
        with tempfile.TemporaryDirectory(prefix='hercules_native_tune_') as temp:
            work=Path(temp)/'task'
            shutil.copytree(task,work,ignore=shutil.ignore_patterns('ref','test','*.o','__pycache__'))
            # Recover the audited baseline even after a winning configuration
            # has been adopted. Materialize symlink dependencies at that commit.
            relative=task.relative_to(ROOT).as_posix()
            tree=subprocess.check_output(['git','ls-tree','-rz',base_ref,'--',relative],cwd=ROOT)
            for entry in tree.split(b'\0'):
                if not entry:continue
                metadata,path=entry.split(b'\t');path=path.decode()
                data=subprocess.check_output(['git','show',f'{base_ref}:{path}'],cwd=ROOT)
                if metadata.startswith(b'120000'):
                    target=posixpath.normpath(posixpath.join(posixpath.dirname(path),data.decode()))
                    data=subprocess.check_output(['git','show',f'{base_ref}:{target}'],cwd=ROOT)
                dest=work/Path(path).relative_to(relative);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
            make=work/'Makefile';lines=make.read_text().splitlines()
            for i,line in enumerate(lines):
                if line.startswith('NVCC_FLAGS') and '-DNDEBUG' not in line:lines[i]+=' -DNDEBUG'
            make.write_text('\n'.join(lines)+'\n')
            f=work/source;s=f.read_text()
            for before,after in replacements.items():
                if before not in s:raise ValueError(f'Missing replacement: {before}')
                s=s.replace(before,after)
            f.write_text(s)
            env={**os.environ,'CUTLASS_DIR':str(ROOT/'reference_sources/cutlass'),
                 'DEEPGEMM_DIR':str(ROOT/'reference_sources/deep_gemm'),
                 'TK_DIR':str(ROOT/'reference_sources/ThunderKittens'),'KH_BENCHMARK':'20'}
            result=subprocess.run(['make','-B','ref'],cwd=work,env=env,text=True,capture_output=True,timeout=600)
            logs.append(result.stdout+result.stderr)
            if result.returncode:raise RuntimeError('build failed')
            with (ROOT/'.cache/hopper_gpu.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                result=subprocess.run(['./ref'],cwd=work,env=env,text=True,capture_output=True,timeout=600)
            logs.append(result.stdout+result.stderr)
            if result.returncode or not re.search(r'\bPassed\b',result.stdout):raise RuntimeError('execution failed')
            record['ref_min_ms']=[float(v) for v in re.findall(r'Ref time:[^\n]*min:\s*([\d.]+)\s*ms',result.stdout)]
            record['passed']=bool(record['ref_min_ms'])
    except Exception as exc:record.update(passed=False,error=str(exc))
    dest=ROOT/'results/hopper_native_release_tuning';dest.mkdir(exist_ok=True)
    (dest/(task.name+'_'+name+'.log')).write_text('\n'.join(logs))
    print(record,flush=True)
    return record

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filter',default='')
    parser.add_argument('--base-ref',default='133ae153161b40e32b95334c5118867d2eb163b4')
    parser.add_argument('--output',default='results/hopper_native_release_tuning.json')
    args=parser.parse_args()
    (ROOT/'.cache').mkdir(exist_ok=True)
    items=[]
    for prefix,(source,variants) in CONFIGS.items():
        task=next((ROOT/'tasks/class1/hopper').glob(prefix+'*'))
        if args.filter not in task.name:continue
        for name,replacements in [('current',{}),*variants]:items.append((task,source,name,replacements,args.base_ref))
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True);reports=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for report in pool.map(measure,items):
            reports.append(report);output.write_text(json.dumps(reports,indent=2)+'\n')

if __name__=='__main__':main()
