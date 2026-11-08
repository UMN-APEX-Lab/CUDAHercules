#!/usr/bin/env python3
"""Reproducibly generate all 60 Hopper FA3 tasks (definitions, wrappers, prompts).

H200 workloads: long prefill, memory-bounded backward, GQA decode. Dimensions
64/96/192 remain explicit synthetic kernel-coverage cases, not model claims.
"""
import argparse
import json
from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / 'scripts/templates/fa3'


def render(path):
    name = path.name
    d = int(re.search(r'hdim(\d+)', name)[1])
    bf16 = '_bf16' in name
    causal = name.endswith('_causal')
    bwd = name.startswith('fa3_bwd')
    split = name.startswith('fa3_fwd_split')
    family = 'bwd' if bwd else ('fwd_split' if split else 'fwd')
    ctype = 'cutlass::bfloat16_t' if bf16 else 'cutlass::half_t'
    dtype = 'bfloat16' if bf16 else 'float16'
    if bwd:
        b, h, hk, sq, sk = 1, 16, 16, 4096, 4096
    elif split:
        b, sk = {64:(8,16421),96:(4,32787),128:(4,32797),192:(2,65549),256:(1,131079)}[d]
        h,hk,sq = (64,8,4) if d==128 else (32,4,4)
    else:
        b, h, hk, sq, sk = (1, 64, 8, 8192, 8192) if d == 128 else (1, 32, 8, 8192, 8192)
    bm = 128 if d == 64 else (80 if d == 128 and not causal else 64)
    profile=TEMPLATES/'h200_splits.json'
    tuned=json.loads(profile.read_text()) if profile.exists() else {}
    nsplits=tuned.get(name,8)  # H200 calibration; regenerate after retuning workloads
    source = (TEMPLATES / f'{family}.cu').read_text()
    source = source.replace('cutlass::half_t', ctype)
    # Change template head dimensions only, never tile/padding constants.
    source = source.replace('hdim128', f'hdim{d}')
    source = source.replace(f'{ctype}, 128, 128,', f'{ctype}, {d}, {d},')
    source = source.replace(f'{ctype}, 128, false>', f'{ctype}, {d}, false>')
    source = source.replace(f'{ctype}, float, 128>', f'{ctype}, float, {d}>')
    source = source.replace('params.is_bf16 = false;', f'params.is_bf16 = {str(bf16).lower()};')
    source = source.replace('params.is_causal = false;', f'params.is_causal = {str(causal).lower()};')
    if causal:
        source = source.replace('params.window_size_right = -1;', 'params.window_size_right = 0;')
    # Query device attributes once per device, outside steady-state timing.
    source = source.replace('cudaDeviceProp prop;\n    cudaGetDeviceProperties(&prop, device);',
        'static thread_local int cached_device = -1;\n    static thread_local cudaDeviceProp prop;\n'
        '    if (device != cached_device) {\n        cudaGetDeviceProperties(&prop, device);\n'
        '        cached_device = device;\n    }')
    if not bwd:
        source = source.replace('    int H,\n', '    int H,\n    int Hkv,\n')
        source = source.replace('params.h_k = H;', 'params.h_k = Hkv;')
        source = source.replace('params.k_batch_stride = static_cast<int64_t>(H)', 'params.k_batch_stride = static_cast<int64_t>(Hkv)')
        source = source.replace('params.v_batch_stride = static_cast<int64_t>(H)', 'params.v_batch_stride = static_cast<int64_t>(Hkv)')
    else:
        source = source.replace('params.seqlen_q_rounded = round_up(Sq, 128);',
                                f'params.seqlen_q_rounded = round_up(Sq, {bm});')
        bn = 96 if d == 192 else (80 if d == 256 else 128)
        source = source.replace('params.seqlen_k_rounded = round_up(Sk, 128);', f'params.seqlen_k_rounded = round_up(Sk, {bn});')
        source = source.replace('(size_t)B * H * Sq * sizeof(float)', '(size_t)B * H * params.seqlen_q_rounded * sizeof(float)')
        source = source.replace('(Sq + 127) / 128', f'(Sq + {bm-1}) / {bm}')
        source = source.replace('(Sk + 127) / 128', f'(Sk + {bn-1}) / {bn}')
        source = source.replace('(size_t)(num_blocks_m * B * H + 1)', '(size_t)(std::max(num_blocks_m, num_blocks_n) * B * H + 1)')
    source = source.replace('hdim=128, fp16, non-causal', f'hdim={d}, {dtype}, causal={causal}')
    source = source.replace('Uses split-KV parallelism with 4 splits.', 'Decode with GQA and split-KV parallelism.')
    signature = re.search(r'extern "C" (void launch_.*?\n\))', source, re.S)[1] + ';'
    # Pointer layout documentation belongs in the prompt, not stale inline comments.
    signature = re.sub(r'\s*//[^\n]*', '', signature)
    mode = 'right-aligned causal: key j is visible iff j <= i + Sk - Sq' if causal else 'non-causal (all keys visible)'
    desc = f'''Implement Hopper SM90 {family} attention, {dtype}, head dimension {d}.
Q: contiguous [{b}, {h}, {sq}, {d}]; K,V: contiguous [{b}, {hk}, {sk}, {d}].
Query head h uses KV head floor(h / ({h}//{hk})). O has Q's shape/dtype.
Compute P = softmax(Q K^T / sqrt({d})), O = P V; {mode}.
LSE is fp32 natural-log sum-exp, shape [B,H,Sq], not log2.
Reference and solution use cold L2 and the minimum of identical timing trials.
Accumulate dot products and softmax in fp32. Do not quantize inputs to FP8.
'''
    if bwd:
        desc += f'''Given O (rounded to {dtype}), dO and natural-log LSE, compute dQ,dK,dV.
Use delta = sum(dO * O) per row; dS = P * (dO V^T - delta).
dQ = scale*dS*K; dK = scale*dS^T*Q; dV = P^T*dO.
Scratch dsoftmax_sum is [B,H,round_up(Sq,{bm})] fp32 and dq_accum is
[B,H,round_up(Sq,{bm}),D] fp32. Scratch contents on entry are unspecified;
only the three gradients are checked. All gradient elements must be written.
'''
    if split:
        desc += f'num_splits={nsplits}; partition KV and combine partial softmax statistics stably.\n'
    desc += '\nRequired entry point:\n' + signature + '\n'
    atol = (0.04 if bf16 else 0.004) if bwd else (0.006 if bf16 else 0.0008)
    rtol = 0.06 if bwd else 0.01
    rms = 0.01 if bf16 else 0.002
    values = dict(B=b,H=h,HK=hk,SQ=sq,SK=sk,D=d,DTYPE=dtype,CAUSAL=causal,
                  BWD=bwd,SPLIT=split,BM=bm,NSPLITS=nsplits,ATOL=atol,RTOL=rtol,RMS=rms,
                  SIGNATURE=repr(signature),DESCRIPTION=repr(desc))
    definition = (TEMPLATES / 'definition.py.in').read_text()
    for key, value in values.items():
        definition = definition.replace(f'@{key}@', str(value))
    config = yaml.safe_load((path/'task.yaml').read_text())
    config['correctness']['tolerances'] = dict(atol=atol, rtol=rtol)
    config['hardware']['required_features'] = ['sm_90a']
    config['anti_cheat']['blocked_patterns'] = [
        r'#include\s*[<"][^">]*(?:flash_attn|flash_fwd|flash_bwd|reference)',
        r'\b(?:run_mha_|prepare_varlen_num_blocks)',
        r'\b(?:__nv_fp8|float_e[45]m[23]|__nv_cvt_.*fp8)',
        r'cvt\.[^;\n]*\.(?:e4m3|e5m2)',
    ]
    return {'def.py':definition, 'reference.cu':source, 'description.txt':desc,
            'task.yaml':yaml.safe_dump(config,sort_keys=False)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    stale=[]
    for path in sorted((ROOT/'tasks/class2/hopper').glob('fa3_*')):
        for name, content in render(path).items():
            target=path/name
            if args.check:
                if not target.exists() or target.read_text()!=content:
                    stale.append(str(target.relative_to(ROOT)))
            else:
                if not target.exists() or target.read_text()!=content:
                    target.write_text(content)
    if stale:
        raise SystemExit('Stale generated files:\n'+'\n'.join(stale))

if __name__=='__main__': main()
