#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

// Slow mathematical implementation; optimization is the benchmark challenge.
__global__ void hh_features(const __nv_bfloat16* x,const __nv_bfloat16* map,
                            float* phi,int H,int N) {
  int row=blockIdx.x,h=(row/N)%H;
  int f=threadIdx.x;
  __shared__ float logits[64];
  float value=0;
  for(int d=0;d<128;++d) value+=float(x[size_t(row)*128+d])*float(map[(h*128+d)*64+f]);
  logits[f]=value;__syncthreads();
  float hi=-INFINITY,lo=-INFINITY;
  for(int j=0;j<64;++j) {hi=fmaxf(hi,logits[j]);lo=fmaxf(lo,-logits[j]);}
  float plus=0,minus=0;
  for(int j=0;j<64;++j) {plus+=expf(logits[j]-hi);minus+=expf(-logits[j]-lo);}
  phi[size_t(row)*128+f]=expf(value-hi)/plus;
  phi[size_t(row)*128+64+f]=expf(-value-lo)/minus;
}
__global__ void hh_output(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,
    __nv_bfloat16* o,const float* qp,const float* kp,const float* alpha,const float* beta,int H,int N) {
  int row=blockIdx.x,h=(row/N)%H,i=row%N,d=threadIdx.x;
  size_t base=size_t(row/N)*N*128;
  int start=max(0,(i/64-1)*64);
  float largest=-INFINITY;
  for(int j=start;j<=i;++j) {
    float dot=0;for(int f=0;f<128;++f) dot+=float(q[size_t(row)*128+f])*float(k[base+size_t(j)*128+f]);
    largest=fmaxf(largest,dot*0.0883883476483f);
  }
  float total=0,norm=0;
  for(int j=0;j<=i;++j) {
    float dot=0;
    for(int f=0;f<128;++f)
      dot+=j<start ? qp[size_t(row)*128+f]*kp[base+size_t(j)*128+f]
                   : float(q[size_t(row)*128+f])*float(k[base+size_t(j)*128+f]);
    float weight=j<start?alpha[h]*dot:beta[h]*expf(dot*0.0883883476483f-largest);
    norm+=weight;total+=weight*float(v[base+size_t(j)*128+d]);
  }
  o[size_t(row)*128+d]=__float2bfloat16(total/(norm+1e-6f));
}
void HedgehogAttention(__nv_bfloat16* Q,__nv_bfloat16* K,__nv_bfloat16* V,__nv_bfloat16* O,
    __nv_bfloat16* Q_map,__nv_bfloat16* K_map,float* alphas,float* betas,int B,int H,int N) {
  float *qp,*kp;size_t count=size_t(B)*H*N*128;
  cudaMalloc(&qp,count*sizeof(float));cudaMalloc(&kp,count*sizeof(float));
  hh_features<<<B*H*N,64>>>(Q,Q_map,qp,H,N);hh_features<<<B*H*N,64>>>(K,K_map,kp,H,N);
  hh_output<<<B*H*N,128>>>(Q,K,V,O,qp,kp,alphas,betas,H,N);
  cudaFree(qp);cudaFree(kp);
}
