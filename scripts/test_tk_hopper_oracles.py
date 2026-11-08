#!/usr/bin/env python3
"""Compare small TK reference runs against independent fp32 mathematical oracles.

Build refs first. Dumps are binary and contain the actual quantized input data;
no text fixtures or duplicate batch/head values are involved.
"""
from pathlib import Path
import json
import struct
import subprocess
import tempfile
import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
    reports=[]
    torch.set_float32_matmul_precision('highest')
    for name in ['tk_based','tk_hedgehog','tk_linear_attention','tk_mamba2']:
        task=ROOT/'tasks/class2/hopper'/name
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data.bin'
            run=subprocess.run(['./ref','1','2','256',str(path)],cwd=task,capture_output=True,text=True,timeout=180)
            if run.returncode:
                raise RuntimeError(f'{name}: {run.stdout}\n{run.stderr}')
            with path.open('rb') as f:
                count=struct.unpack('Q',f.read(8))[0]
                raw=[]
                for _ in range(count):
                    size=struct.unpack('Q',f.read(8))[0]
                    raw.append(bytearray(f.read(size)))
            def tensor(i,dtype=torch.bfloat16):
                return torch.frombuffer(raw[i],dtype=dtype).cuda().float()
            dim=16 if name=='tk_based' else 128
            hq=1 if name=='tk_mamba2' else 2
            q=tensor(0).reshape(1,hq,256,dim)
            k=tensor(1).reshape(1,hq,256,dim)
            dv=64 if name in ['tk_based','tk_mamba2'] else 128
            v=tensor(2).reshape(1,2,256,dv)
            actual=tensor(3).reshape_as(v)
            def oracle(q,k,v):
                score=q@k.transpose(-1,-2)
                qi=torch.arange(256,device='cuda')[:,None]
                ki=torch.arange(256,device='cuda')[None,:]
                causal=ki<=qi
                if name=='tk_based':
                    x=score/(dim**0.5)
                    weights=(1+x+0.5*x.square()).masked_fill(~causal,0)
                elif name=='tk_linear_attention':
                    slope=tensor(4,torch.float32).view(1,2,1,1)
                    weights=(score*torch.exp(-slope*(qi-ki))).masked_fill(~causal,0)
                elif name=='tk_mamba2':
                    decay=tensor(4,torch.float32).reshape(1,2,256).cumsum(-1)
                    log_decay=decay[:,:,:,None]-decay[:,:,None,:]
                    weights=(score*log_decay.exp()).masked_fill(~causal,0)
                else:
                    qm=tensor(4).reshape(2,128,64)
                    km=tensor(5).reshape(2,128,64)
                    def feature(x,m):
                        y=x@m
                        return torch.cat([y.softmax(-1),(-y).softmax(-1)],-1)
                    alpha=tensor(8,torch.float32).reshape(1,2,1,1)
                    beta=tensor(9,torch.float32).reshape(1,2,1,1)
                    linear=(feature(q,qm)@feature(k,km).transpose(-1,-2))*alpha
                    sliding=causal & (ki >= (qi//64-1)*64)
                    logits=(score/(128**0.5)).masked_fill(~sliding,-float('inf'))
                    window=(logits-logits.max(-1,keepdim=True).values).exp()*beta
                    weights=window+linear.masked_fill(~(causal & ~sliding),0)
                    weights/=weights.sum(-1,keepdim=True)+1e-6
                return weights@v
            expected=oracle(q,k,v)
            diff=(actual-expected).abs()
            nrms=(diff.square().mean()/expected.square().mean()).sqrt().item()
            atol=0.15 if name=='tk_based' else (0.1 if name=='tk_linear_attention' else 0.012)
            max_fraction=(diff/(atol+0.06*expected.abs())).max().item()
            passed=bool(torch.isfinite(actual).all()) and nrms<0.015 and max_fraction<1
            quantized=[x.to(torch.float8_e4m3fn).to(x.dtype) for x in [q,k,v]]
            fp8=oracle(*quantized)
            fp8_nrms=((fp8-expected).square().mean()/expected.square().mean()).sqrt().item()
            controls={'fp8_operands':fp8_nrms>0.01}
            if name=='tk_based':
                cheap=v.cumsum(-2)
                controls['cumsum_only']=((cheap-expected).square().mean()/expected.square().mean()).sqrt().item()>0.01
            passed=passed and all(controls.values())
            from cuda_hercules.utils import task_source_hash
            report=dict(task=name,source_hash=task_source_hash(str(task)),negative_controls=controls,fp8_normalized_rms=fp8_nrms,passed=passed,max_abs=diff.max().item(),normalized_rms=nrms,max_tolerance_fraction=max_fraction)
            print(json.dumps(report),flush=True);reports.append(report)
    out=ROOT/'results/hopper_tk_oracles.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(reports,indent=2)+'\n')
    if not all(r['passed'] for r in reports):raise SystemExit(1)

if __name__=='__main__':
    import fcntl
    (ROOT/'.cache').mkdir(exist_ok=True)
    with (ROOT/'.cache/hopper_gpu.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
