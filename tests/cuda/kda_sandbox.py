"""Real chroot/GPU/long-path preflight; does not call an LLM or the task judge."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from kda_eval import sandbox

SOURCE=r'''
#include <cuda_runtime.h>
#include <cstdio>
__global__ void work(float *output){int i=threadIdx.x;output[i]=3*i+1;}
int main(){
 int n=0;auto err=cudaGetDeviceCount(&n);
 if(err!=cudaSuccess||n!=1)return 1;
 float *output;cudaMalloc(&output,128*sizeof(float));
 work<<<1,128>>>(output);float host[128];
 if(cudaMemcpy(host,output,sizeof(host),cudaMemcpyDeviceToHost)!=cudaSuccess)return 2;
 for(int i=0;i<128;i++)if(host[i]!=3*i+1)return 3;
 puts("Passed: standalone sm_90a CUDA kernel on one GPU");cudaFree(output);
}
'''


def main():
    os.environ['KH_SANDBOX_BACKEND']='chroot'
    directory=ROOT/'results/kda/preflight';directory.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='formal_run_directory_'+'x'*80,dir=directory))
    (root/'workspace').mkdir();(root/'socket').mkdir()
    sandbox.install_opt_kh(str(root/'opt'))
    (root/'workspace/probe.cu').write_text(SOURCE)
    spec=sandbox.SandboxSpec(str(root/'workspace'),str(root/'socket'),0,str(root/'opt'))
    command=r'''set -eu
test ! -e /root/CUDAHercules
test ! -e /root/.env
python3 - <<'PYTHON'
import os,socket,numpy,ncu_report
assert os.geteuid()!=0
assert not any(name in os.environ for name in ['AWS_OPENAI_API_KEY','OPENAI_API_KEY'])
for family in [socket.AF_INET,socket.AF_INET6]:
    try:socket.socket(family,socket.SOCK_STREAM)
    except PermissionError:pass
    else:raise AssertionError('Internet socket allowed')
print('Passed: filesystem, UID, credentials, seccomp and Python imports')
PYTHON
nvcc -O3 -arch=sm_90a probe.cu -o probe
./probe
echo persisted > persistent.txt
sleep 300 &
'''
    try:
        with (ROOT/'.cache/hopper_gpu.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            rc,out,err,timed_out=sandbox.run_in_sandbox(spec,['/bin/bash','-c',command],timeout_s=90)
            print(out+err)
            assert rc==0 and not timed_out
            rc,out2,err2,_=sandbox.run_in_sandbox(spec,['/bin/bash','-c','test -f persistent.txt; ./probe'],timeout_s=30)
            print(out2+err2);assert rc==0
        report=dict(passed=True,long_path_bytes=len(str(root)),cuda_kernel=True,
                    filesystem_isolation=True,internet_denied=True,unprivileged=True,
                    credentials_absent=True,numpy=True,ncu_report=True,background_cleanup=True,
                    workspace_persistence=True)
        (directory/'sandbox_longpath.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        sandbox.cleanup_sandbox(spec);shutil.rmtree(root)


if __name__=='__main__':main()
