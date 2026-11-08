"""Run with compute-sanitizer --tool memcheck --error-exitcode 99 python this_file.

Exercises noncausal hdim128's 80-row tiles with irregular Q/K sequence lengths,
multiple heads/batches, repeated launches, and canaries around each workspace.
"""
from pathlib import Path
import sys
import fcntl
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import torch
from cuda_hercules.compiler import compile_cuda_module
from cuda_hercules.eval import load_task_def,_build_call_args,check_correctness


def main():
    task_dir=ROOT/'tasks/class2/hopper/fa3_bwd_hdim128_fp16'
    task=load_task_def(str(task_dir))
    globals_=task['get_inputs'].__globals__
    globals_.update(BATCH_SIZE=2,NUM_HEADS=4,NUM_KV_HEADS=4,SEQ_LEN_Q=257,SEQ_LEN_K=385)
    task['SCALAR_ARGS'].update(B=2,H=4,Sq=257,Sk=385)
    build=ROOT/'.cache/ref_builds'/task_dir.name;build.mkdir(parents=True,exist_ok=True)
    fn=compile_cuda_module(cuda_source=str(task_dir/'reference.cu'),function_signature=task['FUNCTION_SIGNATURE'],
        extra_include_dirs=[str(ROOT/p) for p in task['REFERENCE_EXTRA_INCLUDES']],
        extra_cuda_cflags=task['REFERENCE_EXTRA_CUDA_FLAGS'],build_dir=str(build),verbose=False)
    with (ROOT/'.cache/hopper_gpu.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        torch.manual_seed(7641)
        inputs=task['get_inputs']();outputs=task['get_outputs'](inputs)
        guarded=[];canaries=[]
        for name,tensor in outputs:
            n=tensor.numel()
            storage=torch.full((n+2048,),123.0,device='cuda',dtype=tensor.dtype)
            view=storage[1024:-1024].reshape_as(tensor)
            if name in task['CHECKED_OUTPUTS']:view.fill_(float('nan'))
            guarded.append((name,view));canaries.append(storage)
        expected=task['reference_fn'](inputs)
        args=_build_call_args(task,inputs,guarded)
        for _ in range(5):
            fn(*args);torch.cuda.synchronize()
            for storage in canaries:
                assert bool((storage[:1024]==123).all())
                assert bool((storage[-1024:]==123).all())
            assert check_correctness(expected,guarded,task['TOLERANCES'],task['CHECKED_OUTPUTS'])[0]
        print('Passed: guarded hdim128 backward, B2 H4 Sq257 Sk385, five launches')

if __name__=='__main__':main()
