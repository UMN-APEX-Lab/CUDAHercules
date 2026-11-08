#!/usr/bin/env python3
"""Isolated task-57 experiment using a pinned Tencent HPC-Ops kernel.

Canonical task sources and the reference cache are never modified. Compilation
can run while KDA is active; GPU work waits for the shared exclusive GPU lock.
"""
from pathlib import Path
import argparse
import difflib
import fcntl
import hashlib
import json
import math
import os
import posixpath
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "2a2e26562433a8ba4b504858f1c938eb7612c901"
TASK = ROOT / "tasks/class1/hopper/57_hopper_grouped_gemm"
SOURCE = "src/group_gemm/sm90/cp_async/group_gemm_fp8.cu"
CONFIG = "src/group_gemm/sm90/cp_async/config.h"


def source_fingerprint(work):
    digest=hashlib.sha256()
    for path in sorted(work.rglob('*')):
        if path.is_file() and (path.suffix in ['.cu','.cuh','.h'] or path.name=='Makefile'):
            digest.update(str(path.relative_to(work)).encode()+b'\0'+path.read_bytes())
    return digest.hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected exactly one occurrence: {before[:100]!r}")
    return text.replace(before, after, 1)


def adapt(upstream):
    config = (upstream / CONFIG).read_text()
    config = config.replace('#include "src/utils/utils.h"', '')
    # The upstream kernel already swaps operands. Its first MMA operand is B.
    config = config.replace('F32E4M3E4M3_SS_TN', 'F32E5M2E4M3_SS_TN')
    source = (upstream / SOURCE).read_text()
    kernel = source[source.index('namespace hpc {'):source.index('}  // namespace kernels')]
    kernel = replace_once(kernel, '''const void *Cptr, const void *Aptr, const void *Bptr,
                                     const float *y_scale_ptr, const int *seqlens_ptr,
                                     const int *cu_seqlens_ptr, const int *tiles_ptr,
                                     const int *cu_tiles_ptr, const int4 *task_map_ptr,
                                     int task_map_len, int m, int n, int k, int num_group,
                                     cutlass::FastDivmod flat_divider''', '''int num_group, const int *Ms, const int *Ns, const int *Ks,
      const __nv_fp8_e4m3 *const *As, const int *ldas,
      const __nv_fp8_e5m2 *const *Bs, const int *ldbs,
      __half *const *Ds, const int *ldds, float alpha''')
    begin = kernel.index('  if constexpr (!kUseTaskMap)')
    end = kernel.index('    // Global tensors', begin)
    kernel = kernel[:begin] + '''  int total_tiles = 0;
  for (int g = 0; g < num_group; ++g)
    total_tiles += ((Ms[g] + kTileM - 1) / kTileM) * ((Ns[g] + kTileN - 1) / kTileN);
  while (iblock < total_tiles) {
    int igroup = 0, rem = iblock;
    for (; igroup < num_group; ++igroup) {
      int count = ((Ms[igroup] + kTileM - 1) / kTileM) * ((Ns[igroup] + kTileN - 1) / kTileN);
      if (rem < count) break;
      rem -= count;
    }
    int m = Ms[igroup], n = Ns[igroup], k = Ks[igroup];
    int num_tile_n = (n + kTileN - 1) / kTileN;
    int itile_m = rem / num_tile_n, itile_n = rem % num_tile_n;
    int ntile = k / kTileK;
    iblock += gridDim.x;
''' + kernel[end:]
    kernel = kernel.replace('(Tin *)Aptr + uint64_t(start_token) * k', '(Tin *)As[igroup]')
    kernel = kernel.replace('(Tin *)Bptr + uint64_t(igroup) * n * k', '(Tin *)Bs[igroup]')
    kernel = kernel.replace('make_stride(k, Int<1>{}));', 'make_stride(ldas[igroup], Int<1>{}));', 1)
    kernel = kernel.replace('make_stride(k, Int<1>{}));', 'make_stride(ldbs[igroup], Int<1>{}));', 1)
    kernel = replace_once(kernel, '''(Tout *)Cptr + uint64_t(start_token) * n''', '(Tout *)Ds[igroup]')
    kernel = replace_once(kernel, 'make_shape(n, m), make_stride(Int<1>{}, n)',
                          'make_shape(n, m), make_stride(ldds[igroup], Int<1>{})')
    kernel = replace_once(kernel, 'float scale = y_scale_ptr[igroup];', 'float scale = alpha;')
    begin = kernel.index('    // Epilogue\n')
    end = kernel.index('\n  // PDL release', begin)
    # Retain the upstream cp.async/WGMMA pipeline and FP32 promotion. Only its
    # BF16 row-major epilogue is replaced to satisfy the existing task ABI.
    kernel = kernel[:begin] + '''    // Task ABI requires FP16 column-major output, without BF16 rounding.
    auto tCg = thr_mma.partition_C(gCT);
    auto coords = thr_mma.partition_C(make_identity_tensor(make_shape(Int<kTileN>{}, Int<kTileM>{})));
#pragma unroll
    for (int i = 0; i < size(tDr); ++i) {
      if (itile_n * kTileN + get<0>(coords(i)) < n &&
          itile_m * kTileM + get<1>(coords(i)) < m)
        tCg(i) = Tout(tDr(i));
    }
    __syncthreads();
  }
''' + kernel[end:]
    header = '''// Copyright (C) 2026 Tencent. MIT license: see LICENSE.txt.
// Adapted from Tencent/hpc-ops at ''' + COMMIT + '''.
// ABI changes: mixed E4M3/E5M2, device pointer arrays, FP16 column-major D.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
#include <cstdlib>
''' + config + '\n' + kernel + '\n} } }\n'
    header += r'''
namespace hpc_experiment {
template<int TM,int TK,int ST,int MIN>
cudaError_t launch(int grid_multiplier, int count, const int* Ms,const int* Ns,const int* Ks,
 const __nv_fp8_e4m3*const* As,const int* ldas,const __nv_fp8_e5m2*const* Bs,const int* ldbs,
 __half*const* Ds,const int* ldds,float alpha) {
 using C=hpc::group_gemm_cp_async::config::FP8GemmConfig<cutlass::float_e4m3_t,cutlass::half_t,TM,64,TK,ST>;
 auto kernel=hpc::group_gemm_cp_async::kernels::group_gemm_fp8_multistage_kernel<C,false,false,MIN>;
 static bool configured=false;
 static int sms=0;
 if(!configured) {
   auto e=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,C::kShmSize);
   if(e!=cudaSuccess)return e;
   int device=0;cudaGetDevice(&device);cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);
   configured=true;
 }
 kernel<<<sms*grid_multiplier,128,C::kShmSize>>>(count,Ms,Ns,Ks,As,ldas,Bs,ldbs,Ds,ldds,alpha);
 return cudaGetLastError();
}
}
inline cudaError_t HopperGroupedGemm(int count,const int* Ms,const int* Ns,const int* Ks,
 float alpha,const __nv_fp8_e4m3** As,const int* ldas,const __nv_fp8_e5m2** Bs,const int* ldbs,
 float beta,__half** Ds,const int* ldds) {
 (void)beta;
 static int config=[](){const char* s=std::getenv("HPC_CONFIG");return s?std::atoi(s):0;}();
 #define RUN(TM,TK,ST,MIN,GRID) return hpc_experiment::launch<TM,TK,ST,MIN>(GRID,count,Ms,Ns,Ks,As,ldas,Bs,ldbs,Ds,ldds,alpha)
 switch(config) {
   case 1: RUN(32,128,2,5,8);
   case 2: RUN(48,128,2,7,7);
   case 3: RUN(16,128,2,5,9);
   case 4: RUN(64,128,3,3,3);
   case 5: RUN(64,128,4,2,2);
   case 6: RUN(64,128,6,1,1);
   case 7: RUN(64,64,3,5,5);
   case 8: RUN(32,128,4,3,3);
   case 9: RUN(64,128,2,5,2);
   case 10: RUN(64,128,2,5,1);
   default: RUN(64,128,2,5,5);
 }
 #undef RUN
}
'''
    return header


def adapt_tma(upstream, work):
    files=['src/group_gemm/sm90/config.h','src/group_gemm/sm90/kernels.cuh',
           'src/utils/tma.cuh','src/utils/utils.cuh','src/utils/utils.h']
    for name in files:
        text=(upstream/name).read_text()
        text=text.replace('F32E4M3E4M3_SS_TN','F32E5M2E4M3_SS_TN')
        if name.endswith('kernels.cuh'):
            text=text.replace('make_tensor_like<cute::bfloat16_t>(tCr)',
                              'make_tensor_like<cutlass::half_t>(tCr)')
        dest=work/'tma_upstream'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
    return r'''
// Copyright (C) 2026 Tencent. MIT license: see LICENSE.txt.
// Prepared TMA kernel from Tencent/hpc-ops 2a2e26562433a8ba4b504858f1c938eb7612c901.
// Original row-major output is transposed on EVERY measured invocation.
#pragma once
#define HPC_TARGET_ARCH 90
#include <functional>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
#include "src/group_gemm/sm90/config.h"
#include "src/group_gemm/sm90/kernels.cuh"
namespace hpc_tma_experiment {
using namespace cute;
static std::function<cudaError_t(const int*,__half*const*)> operation;
static std::vector<void*> allocations;
template<class T> T* allocate(size_t n) {
 T* p=nullptr;auto e=cudaMalloc(&p,n*sizeof(T));
 if(e!=cudaSuccess){fprintf(stderr,"TMA allocation: %s\n",cudaGetErrorString(e));exit(1);}
 allocations.push_back(p);return p;
}
__global__ void transpose_groups(const cutlass::half_t* input,__half*const* output,
 const int* Ms,const int* prefix,int groups,int N) {
 __shared__ __half tile[32][33];
 int total=0;
 for(int g=0;g<groups;++g)total+=((Ms[g]+31)/32)*((N+31)/32);
 for(int index=blockIdx.x;index<total;index+=gridDim.x) {
  int group=0,rem=index;
  for(;group<groups;++group){int count=((Ms[group]+31)/32)*((N+31)/32);if(rem<count)break;rem-=count;}
  int M=Ms[group],bm=rem/((N+31)/32),bn=rem%((N+31)/32);
  int row=bm*32+threadIdx.y,col=bn*32+threadIdx.x;
  for(int i=0;i<32;i+=8)if(row+i<M&&col<N)
   tile[threadIdx.y+i][threadIdx.x]=reinterpret_cast<const __half*>(input)[(size_t(prefix[group])+row+i)*N+col];
  __syncthreads();
  row=bm*32+threadIdx.x;col=bn*32+threadIdx.y;
  for(int i=0;i<32;i+=8)if(row<M&&col+i<N)
   output[group][row+size_t(col+i)*M]=tile[threadIdx.x][threadIdx.y+i];
  __syncthreads();
 }
}
template<int TM,int ST,int POLICY>
void prepare(int groups,const int* hMs,const int* hNs,const int* hKs,
 const __nv_fp8_e4m3*const* hAs,const __nv_fp8_e5m2*const* hBs,float alpha) {
 using Tin=cutlass::float_e4m3_t;using Tout=cutlass::half_t;
 int N=hNs[0],K=hKs[0];
 std::vector<int> prefix(groups+1,0);
 for(int g=0;g<groups;++g){
  if(hNs[g]!=N||hKs[g]!=K||K%128||N%128){fputs("Unsupported TMA experiment geometry\n",stderr);exit(1);}
  prefix[g+1]=prefix[g]+hMs[g];
  if(reinterpret_cast<const char*>(hAs[g])!=reinterpret_cast<const char*>(hAs[0])+size_t(prefix[g])*K||
     reinterpret_cast<const char*>(hBs[g])!=reinterpret_cast<const char*>(hBs[0])+size_t(g)*N*K){
   fputs("TMA experiment needs contiguous group buffers\n",stderr);exit(1);}
 }
 int M=prefix.back();
 auto scratch=allocate<Tout>(size_t(M)*N);
 auto cu=allocate<int>(groups+1);cudaMemcpy(cu,prefix.data(),(groups+1)*sizeof(int),cudaMemcpyHostToDevice);
 std::vector<float> scales(groups,alpha);auto scale=allocate<float>(groups);
 cudaMemcpy(scale,scales.data(),groups*sizeof(float),cudaMemcpyHostToDevice);
 auto descriptors=allocate<TmaDescriptor>(groups*2);
 auto tiles=allocate<int>(groups+1),cu_tiles=allocate<int>(groups+1);
 using C=hpc::group_gemm::GroupGEMMFp8Config<Tin,Tout,TM,128,128,ST,2,1,128,128,64>;
 C config;
 auto X=make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(hAs[0])),make_shape(M,K),make_stride(K,Int<1>{}));
 auto W=make_tensor(make_gmem_ptr(reinterpret_cast<const Tin*>(hBs[0])),make_shape(N,K,groups),make_stride(K,Int<1>{},N*K));
 auto Y=make_tensor(make_gmem_ptr(scratch),make_shape(N,M),make_stride(Int<1>{},N));
 auto [tx,tw,ty]=config.get_tma(X,W,Y);
 hpc::vec_t<TmaDescriptor,2> td{*tx.get_tma_descriptor(),*ty.get_tma_descriptor()};
 auto kernel=hpc::group_gemm::kernels::group_gemm_fp8_kernel<C,decltype(tx),decltype(tw),decltype(ty),POLICY,false>;
 int device=0,sms=0;cudaGetDevice(&device);cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);
 int shared=config.get_shm_size()+sizeof(int)*(groups+1);
 auto e=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);
 if(e!=cudaSuccess){fprintf(stderr,"TMA configure: %s\n",cudaGetErrorString(e));exit(1);}
 auto source=reinterpret_cast<const Tin*>(hAs[0]);
 cutlass::FastDivmod divider(N/128);
 operation=[=](const int* Ms,__half*const* Ds){
  hpc::group_gemm::kernels::update_grouped_tma<Tin,Tout,decltype(tx),decltype(ty),TM,8,32,false,false>
   <<<groups+1,32>>>(td,descriptors,source,scratch,Ms,cu,tiles,cu_tiles,nullptr,groups,M,N,K,sms);
  kernel<<<sms,384,shared>>>(tw,descriptors,const_cast<int*>(Ms),scale,tiles,cu_tiles,nullptr,groups,M,N,K,divider);
  transpose_groups<<<sms*4,dim3(32,8)>>>(scratch,Ds,Ms,cu,groups,N);
  return cudaGetLastError();
 };
}
}
inline void HpcTmaPrepare(int groups,const int* Ms,const int* Ns,const int* Ks,
 const __nv_fp8_e4m3*const* As,const __nv_fp8_e5m2*const* Bs,float alpha) {
 using namespace hpc_tma_experiment;
 operation={};for(auto p:allocations)cudaFree(p);allocations.clear();
 const char* text=std::getenv("HPC_CONFIG");int cfg=text?std::atoi(text):0;
 switch(cfg){
  case 1:prepare<32,8,2>(groups,Ms,Ns,Ks,As,Bs,alpha);break;
  case 2:prepare<48,8,2>(groups,Ms,Ns,Ks,As,Bs,alpha);break;
  case 3:prepare<64,6,2>(groups,Ms,Ns,Ks,As,Bs,alpha);break;
  case 4:prepare<64,8,1>(groups,Ms,Ns,Ks,As,Bs,alpha);break;
  case 5:prepare<32,8,1>(groups,Ms,Ns,Ks,As,Bs,alpha);break;
  default:prepare<64,8,2>(groups,Ms,Ns,Ks,As,Bs,alpha);break;
 }
}
inline cudaError_t HopperGroupedGemm(int count,const int* Ms,const int* Ns,const int* Ks,
 float alpha,const __nv_fp8_e4m3** As,const int* ldas,const __nv_fp8_e5m2** Bs,const int* ldbs,
 float beta,__half** Ds,const int* ldds) {
 return hpc_tma_experiment::operation(Ms,Ds);
}
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--upstream',type=Path,default=ROOT/'.cache/production_gemm/hpc-ops')
    ap.add_argument('--output',type=Path,default=ROOT/'results/hopper57_hpc_ops')
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--skip-build',action='store_true',help='Reuse the already prepared binary')
    ap.add_argument('--configs',default='')
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--backend',choices=['cp_async','tma'],default='cp_async')
    ap.add_argument('--base-ref',default='456e250325756eb59af64eb3bf54338a0c155091',
                    help='Original CUTLASS baseline revision, preserved after reference adoption')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    configs=list(map(int,args.configs.split(','))) if args.configs else list(range(11 if args.backend=='cp_async' else 6))
    if any(c not in range(11 if args.backend=='cp_async' else 6) for c in configs):
        raise SystemExit('Unknown configuration for selected backend')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.upstream,text=True).strip()
    if revision!=COMMIT:raise SystemExit(f'Expected upstream {COMMIT}; got {revision}')
    work=out/'work';work.mkdir(exist_ok=True)
    relative=TASK.relative_to(ROOT).as_posix()
    tree=subprocess.check_output(['git','ls-tree','-rz',args.base_ref,'--',relative],cwd=ROOT)
    for entry in tree.split(b'\0'):
        if not entry:continue
        metadata,path=entry.split(b'\t');path=path.decode()
        if Path(path).name=='solution.h':continue
        data=subprocess.check_output(['git','show',f'{args.base_ref}:{path}'],cwd=ROOT)
        if metadata.startswith(b'120000'):
            target=posixpath.normpath(posixpath.join(posixpath.dirname(path),data.decode()))
            data=subprocess.check_output(['git','show',f'{args.base_ref}:{target}'],cwd=ROOT)
        dest=work/Path(path).relative_to(relative);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    makefile=work/'Makefile'
    flags='-DNDEBUG -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED'+(' -Itma_upstream' if args.backend=='tma' else '')
    makefile.write_text(makefile.read_text().replace('-DNDEBUG',flags))
    (work/'solution.h').write_text(adapt(args.upstream) if args.backend=='cp_async' else adapt_tma(args.upstream,work))
    if args.backend=='tma':
        cu=work/'57_hopper_grouped_gemm.cu'
        cu.write_text(replace_once(cu.read_text(),'      // Run solution once for correctness check',
          '''      HpcTmaPrepare(options.groups,h_Ms.data(),h_Ns.data(),h_Ks.data(),h_A_ptrs.data(),h_B_ptrs.data(),options.alpha);
      // Run solution once for correctness check'''))
    shutil.copy2(args.upstream/'LICENSE.txt',work/'LICENSE.txt')
    (out/'adapter.diff').write_text(''.join(difflib.unified_diff(
        (args.upstream/SOURCE).read_text().splitlines(keepends=True),
        (work/'solution.h').read_text().splitlines(keepends=True),fromfile=SOURCE,tofile='adapted_solution.h')))
    provenance=dict(upstream='https://github.com/Tencent/hpc-ops',commit=revision,
        backend=args.backend,
        baseline_revision=args.base_ref,
        source=SOURCE,config=CONFIG,canonical_task_unchanged=True,
        adapter_sha256=hashlib.sha256((work/'solution.h').read_bytes()).hexdigest(),
        adaptations=['E5M2 x E4M3 WGMMA after operand swap, no input conversion',
          'Device Ms/Ns/Ks and pointer/stride arrays replace contiguous grouped buffers',
          'FP16 column-major epilogue replaces BF16 row-major epilogue',
          'Keep upstream cp.async pipeline and per-K-tile FP32 accumulation promotion'],
        limitations=['Experiment requires K multiple of tile K and N multiple of 64, true for all default task-57 cases'],
        timing='Identical cold-L2 CUDA-event policy, complete API calls, min over 20 trials; repeat whole process')
    provenance['upstream_source_sha256']={name:hashlib.sha256((args.upstream/name).read_bytes()).hexdigest()
        for name in [SOURCE,CONFIG,'src/group_gemm/sm90/kernels.cuh','src/group_gemm/sm90/config.h']}
    if args.backend=='tma':
        provenance.update(source='src/group_gemm/sm90/kernels.cuh',
          adaptations=['E5M2 x E4M3 MMA after swap; FP16 output without BF16 rounding',
            'Prepared contiguous input descriptors and buffers outside timing, as with the existing CUTLASS reference',
            'Include descriptor update, GEMM and row-major-to-column-major transpose on EVERY measured call'],
          limitations=['Prepared contiguous grouped input buffers, uniform N/K; true for default task-57 inputs'])
    fingerprint=source_fingerprint(work)
    provenance['prepared_source_sha256']=fingerprint
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    env={**os.environ,'CUTLASS_DIR':str(ROOT/'reference_sources/cutlass'),'KH_BENCHMARK':'20'}
    if not args.skip_build:
        build=subprocess.run(['make','-B','test'],cwd=work,env=env,text=True,capture_output=True)
        (out/'build.log').write_text(build.stdout+build.stderr)
        if build.returncode:raise SystemExit('Compilation failed; see '+str(out/'build.log'))
        stamp=dict(source_sha256=fingerprint,binary_sha256=hashlib.sha256((work/'test').read_bytes()).hexdigest())
        (out/'build_manifest.json').write_text(json.dumps(stamp,indent=2)+'\n')
        print('Compiled HPC-Ops adapter',flush=True)
    else:
        stamp_path=out/'build_manifest.json'
        if not stamp_path.exists() or not (work/'test').exists():raise SystemExit('No verified build; omit --skip-build')
        stamp=json.loads(stamp_path.read_text())
        if stamp['source_sha256']!=fingerprint or stamp['binary_sha256']!=hashlib.sha256((work/'test').read_bytes()).hexdigest():
            raise SystemExit('Prepared sources or binary changed; omit --skip-build')
    if args.prepare_only:return
    reports=[]
    print('Waiting for exclusive GPU lock',flush=True)
    with (ROOT/'.cache/hopper_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        print('GPU lock acquired',flush=True)
        for repeat in range(args.repeats):
            for config in configs:
                result=subprocess.run(['./test'],cwd=work,env={**env,'HPC_CONFIG':str(config)},text=True,capture_output=True,timeout=600)
                log=out/f'config{config}_repeat{repeat}.log';log.write_text(result.stdout+result.stderr)
                cases=[]
                blocks=re.split(r'===\s*(.*?)\s*===',result.stdout)
                for label,body in zip(blocks[1::2],blocks[2::2]):
                    ref=re.search(r'Ref time:[^\n]*min:\s*([\d.]+)\s*ms',body)
                    sol=re.search(r'Kernel time:[^\n]*min:\s*([\d.]+)\s*ms',body)
                    if ref and sol:cases.append(dict(name=label,ref_ms=float(ref[1]),hpc_ms=float(sol[1]),speedup=float(ref[1])/float(sol[1])))
                row=dict(config=config,repeat=repeat,passed=result.returncode==0 and len(cases)==3,cases=cases,log=str(log.relative_to(out)))
                if row['passed']:row['geomean_speedup']=math.exp(sum(math.log(c['speedup']) for c in cases)/len(cases))
                reports.append(row);(out/'measurements.json').write_text(json.dumps(reports,indent=2)+'\n')
                print(json.dumps(row),flush=True)
                if result.returncode and not cases:print(result.stderr[-1200:],flush=True)
    if not any(r['passed'] for r in reports):raise SystemExit('No passing HPC-Ops candidate')


if __name__=='__main__':main()
