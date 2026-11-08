#!/usr/bin/env python3
"""Generate accurate prompts and precision policies for Hopper Make tasks."""
from pathlib import Path
import argparse
import re
import yaml
ROOT=Path(__file__).resolve().parents[1]

CONTRACTS={
'48_':'''FP32/TF32 GEMM. A is row-major [M,K], B column-major [K,N], C column-major [M,N].
C = alpha*A@B + beta*C. Accumulate in fp32; TF32 tensor-core rounding is allowed.
BF16/FP16/FP8 quantization of the input is not allowed. Reciprocal per-K power-of-two factors spanning 2^-20..2^20
exercise the FP32/TF32 exponent range while keeping dot products well scaled. beta=0 means C is output-only.''',
'49_':'''FP16 GEMM, fp32 accumulation, fp16 output. A row-major [M,K], B column-major [K,N], D column-major [M,N].
D = alpha*A@B + beta*D. beta=0 means D is output-only.''',
'50_':'''INT8 GEMM with INT32 output/accumulation. A row-major [M,K], B column-major [K,N], D column-major [M,N].
D = alpha*A@B + beta*D. Results must be exact. This task measures integer GEMM with a cached cuBLASLt baseline;
there is no requirement to reproduce a particular internal swizzle. beta=0 means D is output-only.''',
'51_':'''Generalized tensor contraction, fp16 A/B/C, fp32 output/accumulation.
Contract over two K modes d=128 and h=K/128; K is a multiple of 128.
A[s,d,h] has address s + M*(d + 128*h). B[d,o,h] has address d + 128*(o + N*h).
C and D have address s+M*o. D[s,o] = alpha*sum_{h,d} A[s,d,h]*B[d,o,h] + beta*C[s,o].
The B tensor is permuted relative to ordinary column-major GEMM. lda/ldc/ldd=M; ldb=K is retained for ABI compatibility;
B's address formula above is authoritative. D is output-only.''',
'52_':'''N-axis gather/GEMM/scatter fusion. A is fp16 row-major [M,K], B fp16 column-major [K,N_orig].
gather_indices and scatter_indices are device arrays of length index_size, with distinct indices in each array.
For logical column j: D[i + scatter_indices[j]*ldd] = half(alpha*sum_k A[i*lda+k]*B[k+gather_indices[j]*ldb]).
D is fp16 column-major [M,N_orig]; lda=ldb=K, ldd=M, beta=0. Every selected column must be written;
unselected columns start at zero and must remain zero. Accumulate in fp32. For MoE token routing, view A as row-major weights [N_out,K], B as row-major tokens [T,K], and D as row-major [T,N_out].
Gather/scatter then select token rows in the transposed GEMM view. Defaults use N_out=1536/2048, T=4096..32768, K=1536/2048.''',
'54_':'''FP8 E4M3 GEMM with fp32 accumulation/output. A row-major [M,K], B column-major [K,N], C column-major [M,N].
C = alpha*A@B + beta*C; benchmark alpha=1,beta=0. No ReLU, amax, bias or auxiliary output.
C is output-only for beta=0. The untimed independent oracle decodes the actual FP8 inputs and runs device GEMM.''',
'55_':'''W8A16 mixed GEMM. A is fp16 row-major [M,K], B is actual signed INT8 column-major [K,N].
scale_B and zero_B are fp16 MN-major arrays [ceil(K/group_size),N], index n + (k/group_size)*N.
Dequantize using B_half[k,n] = half(float(scale_B[idx])*float(B[k+n*ldb]) + float(zero_B[idx])).
D is fp16 row-major [M,N]: D[m,n] = half(alpha*sum_k A[m,k]*B_half[k,n]).
Accumulate in fp32. beta=0 in all benchmark cases; D is output-only. No FP8 reinterpretation is used.''',
'56_':'''Pointer-array batched fp16 GEMM with fp32 accumulation and fp16 output.
A_ptrs/B_ptrs/D_ptrs are device arrays of device pointers. All batch elements have common M,N,K.
A row-major [M,K], B column-major [K,N], D column-major [M,N]; lda=ldb=K, ldd=M.
D = half(alpha*A@B), beta=0. Inputs differ across batch elements. All outputs must be written.''',
'57_':'''Grouped FP8 MoE GEMM. A is E4M3 row-major [M_g,K_g]; B is E5M2 column-major [K_g,N_g].
D is fp16 column-major [M_g,N_g], fp32 accumulation, D=half(alpha*A@B), beta=0.
Ms,Ns,Ks,ldas,ldbs,ldds and pointer arrays are on device. Each group may have a different M; do not assume uniform groups.
lda=ldb=K_g, ldd=M_g. Every output must be written; zero-row groups, when supplied, require no work.''',
'61_':'''MoE router: fp16 A row-major [M,K], B column-major [K,N], fp32 dot products, fp16 D row-major [M,N].
For each token row, select topk=8 of N=128 logits alpha*A@B, then softmax ONLY over those eight logits;
set all other probabilities to zero. Break equal-logit ties by smaller expert index, selecting exactly eight experts.
Select and normalize before rounding the final probabilities to fp16. Default alpha=1/sqrt(K).
The baseline selects a prepared CUTLASS fusion or cuBLASLt plus an adapted vLLM router, including all stages in timing.''',
'62_':'''Structured 2:4 sparse GEMM. A_compressed is fp16 row-major [M,K/2]; metadata is uint16 row-major [M,K/16].
Each metadata word contains four nibbles. Nibble g stores idx0 | (idx1<<2) for a group of four dense K elements;
0<=idx0<idx1<4. The two compressed values for group q are at m*(K/2)+2*q and +1.
B is fp16 column-major [K,N], D fp16 row-major [M,N]. Accumulate in fp32 and round D to fp16.
D=half(alpha*A_sparse@B), beta=0, lda for B is K, ldd=N. CUTLASS's private swizzle is converted to this canonical interface
outside timing. Compare against the correctly transposed and rounded fp32 reference.''',
'63_':'''Decode FP8 GEMM in a SiLU-to-GEMM pipeline. A E4M3 row-major [M,K], B E5M2 column-major [K,N],
D BF16 column-major [M,N]. D=bf16(alpha*A@B), fp32 accumulation, beta=0; D is output-only.
Before every measured invocation the harness launches the same predecessor computing A=e4m3(SiLU(X)).
The predecessor signals programmatic launch completion; the reference may prefetch B while activation computation finishes.
A normal default-stream solution may rely on standard stream ordering; a solution using PDL must synchronize its dependency
before reading A. The reported time includes the predecessor for both reference and solution.''',
'67_':'''Block-scaled FP8 E4M3 GEMM. A row-major [M,K], B column-major [K,N], D fp32 column-major [M,N].
Both operands use 128x128 block scales. scale_A[(m/128)+kb*ceil(M/128)] and
scale_B[(n/128)+kb*ceil(N/128)] are fp32, positive, nonconstant; kb=k/128.
D[m,n]=alpha*sum_k float(A[m,k])*float(B[k,n])*scale_A[...] * scale_B[...].
FP32 accumulation, no ReLU, bias, amax or auxiliary output. beta=0. All output elements must be written.''',
'68_':'''Grouped block-scaled FP8 E4M3 MoE GEMM, fp32 accumulation, BF16 column-major D.
Device pointer arrays and dimension/stride arrays specify unequal groups. A row-major [M_g,K_g], B column-major [K_g,N_g].
Activation scales have granularity 1x128: scale_A[m + kb*M_g]. Weight scales have granularity 128x128:
scale_B[(n/128) + kb*ceil(N_g/128)]. Both are positive nonconstant fp32. kb=k/128.
D[m,n]=bf16(alpha*sum_k A[m,k]*B[k,n]*scale_A[...]*scale_B[...]), beta=0.
lda=ldb=K_g, ldd=M_g; all outputs must be written.''',
'69_':'''Grouped W8A16 GEMM for uneven expert token counts. A_ptrs point to fp16 row-major [M_g,K_g],
B_ptrs to signed INT8 column-major [K_g,N_g], D_ptrs to fp16 row-major [M_g,N_g].
Ms,Ns,Ks,ldas,ldbs,ldds and pointer arrays are DEVICE arrays. lda=ldb=K_g, ldd=N_g.
D=half(alpha*sum_k float(A)*float(B)), fp32 accumulation, beta=0. There are no weight scales in this ConvertOnly variant.
Reference validation and timing use these exact same inputs and operation; do not reinterpret FP8 bytes as int8.''',
'88_':'''Hopper FMHA, fp16 contiguous Q[B,H,Sq,D], K/V[B,H,Sk,D], O[B,H,Sq,D].
Compute O=softmax(Q@K^T/sqrt(D))@V with fp32 dot products, softmax and accumulation, then round to fp16.
The default workloads use non-causal multi-head attention (Hq=Hkv), D=128, long context.
This ABI does not provide GQA or causal masking; those are explicitly covered by the FA3 tasks.
All output elements must be written. FP8 quantization of fp16 inputs is forbidden.''',
'deepgemm_mqa_logits':'''DeepGEMM FP8 MQA indexer logits. Q E4M3 [S,H=64,D=128], KV E4M3 [Skv,D], KV_scales fp32 [Skv], weights fp32 [S,H].
For each valid key k in [cu_seqlen_ks[s], cu_seqlen_ke[s]): logits[s,k] = sum_h weights[s,h]*ReLU(sum_d Q[s,h,d]*KV[k,d]*KV_scales[k]).
Write -infinity outside the interval; NaN is never valid. Output fp32 [S,Skv], contiguous, includes padded query rows.
Intervals are right-aligned causal, exclusive upper bound. Every unmasked logit must be written with fp32 accumulation.''',
'Bf16Gemm':'''BF16 GEMM with fp32 accumulation and BF16 output. A row-major [M,K], B row-major [K,N], C row-major [M,N].
C=bf16(A@B). The output is checked with both a rounding-aware elementwise bound and normalized RMS error.
Do not quantize BF16 inputs to FP8. All output elements must be written.''',
'Fp8Gemm':'''FP8 E4M3 GEMM with fp32 accumulation and BF16 output. A row-major [M,K], B row-major [N,K], C BF16 row-major [M,N].
C=bf16(A@B^T). Only the output is BF16; input pointers remain E4M3. All output elements must be written.''',
'Fp8GemmScaled':'''Token/channel scaled FP8 GEMM. A E4M3 row-major [M,K], B E4M3 row-major [N,K], C fp32 row-major [M,N].
A_scale fp32 [M], B_scale fp32 [N]. C[m,n] = A_scale[m]*B_scale[n]*sum_k float(A[m,k])*float(B[n,k]).
Accumulate in fp32. The oracle uses the actual quantized operands and scales, not original unquantized inputs.''',
'FusedRotary':'''Split-half rotary embedding, BF16 inputs/output, fp32 intermediate arithmetic.
x/out contiguous [B=1,H=72,N=16384,D=128]; sin_table/cos_table are BF16 [N,D/2], shared across batch/head.
For d<D/2: out[...,d]=bf16(x[...,d]*cos[pos,d] - x[...,d+D/2]*sin[pos,d]);
out[...,d+D/2]=bf16(x[...,d+D/2]*cos[pos,d] + x[...,d]*sin[pos,d]).
Position tables encode position IDs 8192..24575. Do not pair adjacent elements. All heads have independent inputs.''',
'BasedLinearAttention':'''Based Taylor linear attention (unnormalized), BF16 contiguous inputs/output.
Q,K [B=8,H=32,N=8192,16], V/O [B,H,N,64]. For j<=i let x=dot(Q_i,K_j)/sqrt(16).
O_i=sum_{j<=i}(1+x+x*x/2)*V_j. There is no denominator or softmax; include the constant, linear and quadratic terms.
Accumulate in fp32 and round output to BF16. Inputs are generated independently for every batch/head.''',
'HedgehogAttention':'''Hedgehog hybrid normalized attention, BF16 Q,K,V,O[B=8,H=32,N=8192,128].
Q_map,K_map are BF16 [H,128,64]. phi(x,map)=concat(softmax(x@map),softmax(-(x@map))).
For query i, window keys are max(0,64*floor(i/64)-64)..i (previous and current 64-token blocks).
Window weights = beta[h]*exp(dot(Q_i,K_j)/sqrt(128)-max_window_logit_i).
For older keys use alpha[h]*dot(phi(Q_i,Q_map),phi(K_j,K_map)); alpha/beta are positive fp32 [H].
Normalize by the sum of all window and linear weights (+1e-6), multiply V and round O to BF16.
Accumulate in fp32. The reference uses BF16 tensor-core intermediates; the published tolerance accounts for their rounding.''',
'LinearAttention':'''Exponentially decayed causal linear attention, BF16 Q,K,V,O[B=4,H=32,N=16384,D=128], slopes positive fp32 [H].
O_i=sum_{j<=i} exp(-slopes[h]*(i-j))*dot(Q_i,K_j)*V_j. No softmax, normalization or 1/sqrt(D) scale.
Accumulate in fp32, round output to BF16. O is output-only even though the reference internally zeros its TMA-reduction destination.''',
'Mamba2Forward':'''Mamba2 SSD state-128, single shared Q/K group variant.
Q,K BF16 [B=8,1,N=8192,128], V/O BF16 [B,H=80,N,64], A fp32 negative log-decays [B,H,N].
State_(-1)=0; State_t=exp(A_t)*State_(t-1)+outer(K_t,V_t); O_t=Q_t@State_t.
Thus O_t=sum_{j<=t} dot(Q_t,K_j)*exp(sum_{u=j+1..t} A_u)*V_j. No additional softmax or scale.
Accumulate in fp32, round output to BF16. Q/K are shared across heads; A/V are not.''',
}
API_RE=r'(?m)^(?:cudaError_t|void)\s+(Hopper\w+|Bf16Gemm|Fp8GemmScaled|Fp8Gemm|FusedRotary|solution_mqa_logits|BasedLinearAttention|HedgehogAttention|LinearAttention|Mamba2Forward)\s*\([^)]*\)\s*\{'


def render(task):
    sol=(task/'solution.h').read_text();match=re.search(API_RE,sol)
    if not match:raise ValueError(task)
    api=match[1];sig=match[0].rstrip()[:-1].strip()+';'
    key=next((k for k in CONTRACTS if task.name.startswith(k)),api)
    if task.name=='deepgemm_mqa_logits':key=task.name
    desc='Implement a CUDA kernel for Hopper SM90 (H200).\n\n'+CONTRACTS[key]+'\n\nRequired public API:\n'+sig+'\n'
    make=(task/'Makefile').read_text()
    active=re.search(r'^ref:\s*([^\s]+\.cu)',make,re.M)[1]
    code=(task/active).read_text()
    cases=re.findall(r'\{\s*"[^"\n]+"\s*,\s*\d+[^}]+\}',code)
    calls=re.findall(r'run_benchmark<[^;\n]+;',code)
    if cases:
        struct=re.search(r'struct TestConfig\s*\{([^}]+)\}',code,re.S)
        fields=re.findall(r'\bint\s+([^;]+);',struct[1]) if struct else []
        desc+='\nDefault cases ('+', '.join(fields)+'):\n'+'\n'.join(cases)+'\n'
    elif calls:
        desc+='\nDefault cases (label, M, N, K):\n'+'\n'.join(calls)+'\n'
    desc+='\nThe harness generates fractional random inputs, poisons output-only buffers, and rejects nonfinite results.\n'
    desc+='Reference and solution use the same cold-L2 timing policy and per-case minimum; every reported case contributes to the geometric-mean speedup.\n'
    cfg=yaml.safe_load((task/'task.yaml').read_text())
    cfg['hardware']['required_features']=['sm_90a']
    previous=[p for p in (cfg['anti_cheat'].get('blocked_patterns') or []) if 'Cutlass\\w*|RunCutlassReference' not in p]
    cfg['anti_cheat']['blocked_patterns']=list(dict.fromkeys(previous+[
        r'#include\s*[<"][^">]*(?:reference|harness|common\.cuh|hopper_validation|hopper_gemm_oracle|solution_api|\.\.)',
        r'\b(?:Cutlass(?:Gemm|Gett)\w*|RunCutlassReference|reference_gemm\w*|kh_\w+|launch_tk_rotary|inner_run|run_ref_kernel|based_init|la_init|hedgehog_init|la_kernel|based_linear_attention|hedgehog_linear_attention_smd)\b',
    ]))
    cfg['anti_cheat']['blocked_patterns'] += [r'\b(?:__nv_fp4\w*|float_e2m1_t)\b',r'cvt\.[^;\n]*\.e2m1']
    if key in ['48_','49_','51_','55_','56_','61_','62_','88_','Bf16Gemm','FusedRotary','BasedLinearAttention','HedgehogAttention','LinearAttention','Mamba2Forward']:
        cfg['anti_cheat']['blocked_patterns'] += [r'\b(?:__nv_fp8\w*|float_e[45]m[23]_t)\b',r'cvt\.[^;\n]*\.(?:e4m3|e5m2)']
    if key=='48_':
        cfg['anti_cheat']['blocked_patterns'] += [r'\b(?:__nv_bfloat16|__float2bfloat16\w*)\b',r'mma[^;\n]*\.bf16']
    cfg['anti_cheat']['blocked_patterns']=list(dict.fromkeys(cfg['anti_cheat']['blocked_patterns']))
    return desc,yaml.safe_dump(cfg,sort_keys=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    stale=[]
    for cls in ['class1','class2']:
        for make in sorted((ROOT/'tasks'/cls/'hopper').glob('*/Makefile')):
            task=make.parent;desc,config=render(task)
            for name,text in [('description.txt',desc),('task.yaml',config)]:
                if args.check:
                    if (task/name).read_text()!=text:stale.append(str(task/name))
                elif (task/name).read_text()!=text:(task/name).write_text(text)
    if stale:raise SystemExit('Stale prompts:\n'+'\n'.join(stale))

if __name__=='__main__':main()
