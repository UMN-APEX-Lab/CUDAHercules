/**
 * GPU Molecular Dynamics Kernels — Three Material Systems
 * Adapted from GPUMD (brucefan1983/GPUMD, GPLv3).
 *
 * Implements three complete force pipelines:
 *   1. Lennard-Jones (noble gas, pairwise)
 *   2. Tersoff (silicon, three-body bond-order)
 *   3. Coulomb/Ewald (NaCl, real-space + k-space)
 *
 * Plus shared kernels:
 *   - Cell-list neighbor list construction
 *   - Velocity Verlet time integration
 *   - Kinetic energy computation
 */
#pragma once
#include <cuda_runtime.h>
#include <cmath>

#define MD_BLOCK_SIZE 128
#define MAX_NEIGHBORS_LJ 200
#define MAX_NEIGHBORS_TERSOFF 50
#define MAX_NEIGHBORS_COULOMB 100
#define EPSILON_ZERO 1.0e-16
#define K_COULOMB 14.3996f

// ═══════════════════════════════════════════════════════════════════════
// Data structures
// ═══════════════════════════════════════════════════════════════════════

struct Box {
    double Lx, Ly, Lz;
    double half_Lx, half_Ly, half_Lz;
};

struct LJParams {
    double sigma, epsilon, cutoff, cutoff_sq;
    double sigma6_x4eps, sigma12_x4eps;
};

struct TersoffParams {
    double A, B, lambda, mu, beta, n, c, d, h;
    double R1, R2, m, alpha, gamma;
    double c2, d2, one_plus_c2_over_d2, pi_factor, minus_half_over_n;
};

inline TersoffParams si_tersoff_params() {
    TersoffParams p;
    p.A=1830.8; p.B=471.18; p.lambda=2.4799; p.mu=1.7322;
    p.beta=1.1e-6; p.n=0.78734; p.c=1.0039e5; p.d=16.217; p.h=-0.59825;
    p.R1=2.7; p.R2=3.0; p.m=3.0; p.alpha=0.0; p.gamma=1.0;
    p.c2=p.c*p.c; p.d2=p.d*p.d; p.one_plus_c2_over_d2=1.0+p.c2/p.d2;
    p.pi_factor=M_PI/(p.R2-p.R1); p.minus_half_over_n=-0.5/p.n;
    return p;
}

// ═══════════════════════════════════════════════════════════════════════
// Shared device helpers
// ═══════════════════════════════════════════════════════════════════════

__device__ __forceinline__ void apply_mic(const Box& box, double& dx, double& dy, double& dz) {
    if (dx > box.half_Lx) dx -= box.Lx; else if (dx < -box.half_Lx) dx += box.Lx;
    if (dy > box.half_Ly) dy -= box.Ly; else if (dy < -box.half_Ly) dy += box.Ly;
    if (dz > box.half_Lz) dz -= box.Lz; else if (dz < -box.half_Lz) dz += box.Lz;
}

// ═══════════════════════════════════════════════════════════════════════
// Shared kernels: Neighbor list
// ═══════════════════════════════════════════════════════════════════════

__global__ void kernel_assign_cells(
    int N, const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
    const Box box, int cx, int cy, int cz, double cell_size,
    int* cell_count, int* cell_contents, int max_per_cell)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int ix = ((int)(x[i]/cell_size)%cx+cx)%cx;
    int iy = ((int)(y[i]/cell_size)%cy+cy)%cy;
    int iz = ((int)(z[i]/cell_size)%cz+cz)%cz;
    int cid = ix*cy*cz + iy*cz + iz;
    int pos = atomicAdd(&cell_count[cid], 1);
    if (pos < max_per_cell) cell_contents[cid*max_per_cell + pos] = i;
}

__global__ void kernel_build_neighbor_list(
    int N, const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
    const Box box, double cutoff_sq, int cx, int cy, int cz, double cell_size,
    const int* cell_count, const int* cell_contents, int max_per_cell,
    int* nn, int* nl, int max_neighbors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double xi=x[i], yi=y[i], zi=z[i];
    int ix=((int)(xi/cell_size)%cx+cx)%cx;
    int iy=((int)(yi/cell_size)%cy+cy)%cy;
    int iz=((int)(zi/cell_size)%cz+cz)%cz;
    int count=0;
    for(int dx=-1;dx<=1;dx++) for(int dy=-1;dy<=1;dy++) for(int dz=-1;dz<=1;dz++){
        int jx=(ix+dx+cx)%cx, jy=(iy+dy+cy)%cy, jz=(iz+dz+cz)%cz;
        int cid=jx*cy*cz+jy*cz+jz;
        int nc=min(cell_count[cid],max_per_cell);
        for(int k=0;k<nc;k++){
            int j=cell_contents[cid*max_per_cell+k];
            if(j==i) continue;
            double rx=x[j]-xi, ry=y[j]-yi, rz=z[j]-zi;
            apply_mic(box,rx,ry,rz);
            if(rx*rx+ry*ry+rz*rz<cutoff_sq && count<max_neighbors){
                nl[i*max_neighbors+count]=j; count++;
            }
        }
    }
    nn[i]=count;
}

// ═══════════════════════════════════════════════════════════════════════
// Shared kernels: Integration + energy
// ═══════════════════════════════════════════════════════════════════════

__global__ void kernel_verlet_step1(
    int N, double dt, double mass_inv, const Box box,
    const double* __restrict__ fx, const double* __restrict__ fy, const double* __restrict__ fz,
    double* x, double* y, double* z, double* vx, double* vy, double* vz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double h=0.5*dt*mass_inv;
    vx[i]+=h*fx[i]; vy[i]+=h*fy[i]; vz[i]+=h*fz[i];
    x[i]+=dt*vx[i]; y[i]+=dt*vy[i]; z[i]+=dt*vz[i];
    if(x[i]<0)x[i]+=box.Lx; else if(x[i]>=box.Lx)x[i]-=box.Lx;
    if(y[i]<0)y[i]+=box.Ly; else if(y[i]>=box.Ly)y[i]-=box.Ly;
    if(z[i]<0)z[i]+=box.Lz; else if(z[i]>=box.Lz)z[i]-=box.Lz;
}

__global__ void kernel_verlet_step2(
    int N, double dt, double mass_inv,
    const double* __restrict__ fx, const double* __restrict__ fy, const double* __restrict__ fz,
    double* vx, double* vy, double* vz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double h=0.5*dt*mass_inv;
    vx[i]+=h*fx[i]; vy[i]+=h*fy[i]; vz[i]+=h*fz[i];
}

__global__ void kernel_kinetic_energy(
    int N, double mass,
    const double* __restrict__ vx, const double* __restrict__ vy, const double* __restrict__ vz,
    double* ke)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    ke[i] = 0.5*mass*(vx[i]*vx[i]+vy[i]*vy[i]+vz[i]*vz[i]);
}

// ═══════════════════════════════════════════════════════════════════════
// Force 1: Lennard-Jones (pairwise)
// ═══════════════════════════════════════════════════════════════════════

__global__ void kernel_lj_force(
    int N, const LJParams lj, const Box box,
    const int* __restrict__ nn, const int* __restrict__ nl, int max_nb,
    const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
    double* fx, double* fy, double* fz, double* pe)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double xi=x[i],yi=y[i],zi=z[i];
    double sfx=0,sfy=0,sfz=0,spe=0;
    int n_nb=nn[i];
    for(int k=0;k<n_nb;k++){
        int j=nl[i*max_nb+k];
        double dx=x[j]-xi,dy=y[j]-yi,dz=z[j]-zi;
        apply_mic(box,dx,dy,dz);
        double r2=dx*dx+dy*dy+dz*dz;
        if(r2<lj.cutoff_sq){
            double r2inv=1.0/r2, r6inv=r2inv*r2inv*r2inv;
            double f2=6.0*(lj.sigma6_x4eps*r6inv-2.0*lj.sigma12_x4eps*r6inv*r6inv)*r2inv;
            sfx+=f2*dx; sfy+=f2*dy; sfz+=f2*dz;
            spe+=(lj.sigma12_x4eps*r6inv*r6inv-lj.sigma6_x4eps*r6inv)*0.5;
        }
    }
    fx[i]=sfx; fy[i]=sfy; fz[i]=sfz; pe[i]=spe;
}

// ═══════════════════════════════════════════════════════════════════════
// Force 2: Tersoff (three-body)
// ═══════════════════════════════════════════════════════════════════════

__device__ __forceinline__ double tersoff_fc(const TersoffParams& p, double d) {
    if(d<p.R1) return 1.0;
    if(d<p.R2) return cos(p.pi_factor*(d-p.R1))*0.5+0.5;
    return 0.0;
}

__device__ __forceinline__ void tersoff_fc_fcp(const TersoffParams& p, double d, double& fc, double& fcp) {
    if(d<p.R1){fc=1.0;fcp=0.0;}
    else if(d<p.R2){double a=p.pi_factor*(d-p.R1);fc=cos(a)*0.5+0.5;fcp=-sin(a)*p.pi_factor*0.5;}
    else{fc=0.0;fcp=0.0;}
}

__device__ __forceinline__ double tersoff_g(const TersoffParams& p, double c) {
    double diff=c-p.h, t=p.d2+diff*diff;
    return p.gamma*(p.one_plus_c2_over_d2-p.c2/t);
}

__device__ __forceinline__ void tersoff_g_gp(const TersoffParams& p, double c, double& g, double& gp) {
    double diff=c-p.h, t=p.d2+diff*diff;
    g=p.gamma*(p.one_plus_c2_over_d2-p.c2/t);
    gp=p.gamma*(2.0*p.c2*diff/(t*t));
}

__global__ void kernel_tersoff_step1(
    int N, const TersoffParams p, const Box box,
    const int* __restrict__ nn, const int* __restrict__ nl, int max_nb,
    const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
    double* g_b, double* g_bp)
{
    int n1=blockIdx.x*blockDim.x+threadIdx.x;
    if(n1>=N) return;
    int num_nb=nn[n1]; double x1=x[n1],y1=y[n1],z1=z[n1];
    for(int i1=0;i1<num_nb;i1++){
        int n2=nl[n1*max_nb+i1];
        double x12=x[n2]-x1,y12=y[n2]-y1,z12=z[n2]-z1;
        apply_mic(box,x12,y12,z12);
        double d12=sqrt(x12*x12+y12*y12+z12*z12);
        double zeta=0.0;
        for(int i2=0;i2<num_nb;i2++){
            int n3=nl[n1*max_nb+i2]; if(n3==n2) continue;
            double x13=x[n3]-x1,y13=y[n3]-y1,z13=z[n3]-z1;
            apply_mic(box,x13,y13,z13);
            double d13=sqrt(x13*x13+y13*y13+z13*z13);
            if(d13>p.R2) continue;
            zeta+=tersoff_fc(p,d13)*tersoff_g(p,(x12*x13+y12*y13+z12*z13)/(d12*d13));
        }
        double bzn=pow(p.beta*zeta,p.n), b_ij=pow(1.0+bzn,p.minus_half_over_n);
        int idx=i1*N+n1;
        if(zeta<EPSILON_ZERO){g_b[idx]=1.0;g_bp[idx]=0.0;}
        else{g_b[idx]=b_ij; g_bp[idx]=-b_ij*bzn*0.5/((1.0+bzn)*zeta);}
    }
}

__global__ void kernel_tersoff_step2(
    int N, const TersoffParams p, const Box box,
    const int* __restrict__ nn, const int* __restrict__ nl, int max_nb,
    const double* __restrict__ g_b, const double* __restrict__ g_bp,
    const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
    double* g_pe, double* g_f12x, double* g_f12y, double* g_f12z)
{
    int n1=blockIdx.x*blockDim.x+threadIdx.x;
    if(n1>=N) return;
    int num_nb=nn[n1]; double x1=x[n1],y1=y[n1],z1=z[n1]; double pot=0.0;
    for(int i1=0;i1<num_nb;i1++){
        int idx=i1*N+n1, n2=nl[n1*max_nb+i1];
        double x12=x[n2]-x1,y12=y[n2]-y1,z12=z[n2]-z1;
        apply_mic(box,x12,y12,z12);
        double d12=sqrt(x12*x12+y12*y12+z12*z12), d12inv=1.0/d12;
        double fc12,fcp12; tersoff_fc_fcp(p,d12,fc12,fcp12);
        double fr12=p.A*exp(-p.lambda*d12), fa12=p.B*exp(-p.mu*d12);
        double frp12=-p.lambda*fr12, fap12=-p.mu*fa12;
        double b12=g_b[idx];
        double factor=(fcp12*(fr12-b12*fa12)+fc12*(frp12-b12*fap12))*d12inv;
        double f12x=x12*factor*0.5, f12y=y12*factor*0.5, f12z=z12*factor*0.5;
        pot+=fc12*(fr12-b12*fa12)*0.5;
        double bp12=g_bp[idx];
        for(int i2=0;i2<num_nb;i2++){
            int n3=nl[n1*max_nb+i2]; if(n3==n2) continue;
            double x13=x[n3]-x1,y13=y[n3]-y1,z13=z[n3]-z1;
            apply_mic(box,x13,y13,z13);
            double d13=sqrt(x13*x13+y13*y13+z13*z13);
            if(d13>p.R2) continue;
            double fc13=tersoff_fc(p,d13);
            double ood=1.0/(d12*d13), cos123=(x12*x13+y12*y13+z12*z13)*ood;
            double g_val,gp_val; tersoff_g_gp(p,cos123,g_val,gp_val);
            double dc=-fc12*bp12*fa12*fc13*gp_val;
            double cod=cos123*d12inv*d12inv;
            f12x+=(dc*(x13*ood-x12*cod))*0.5;
            f12y+=(dc*(y13*ood-y12*cod))*0.5;
            f12z+=(dc*(z13*ood-z12*cod))*0.5;
        }
        g_f12x[idx]=f12x; g_f12y[idx]=f12y; g_f12z[idx]=f12z;
    }
    g_pe[n1]+=pot;
}

__global__ void kernel_accumulate_forces(
    int N, const int* __restrict__ nn, const int* __restrict__ nl, int max_nb,
    const double* __restrict__ g_f12x, const double* __restrict__ g_f12y, const double* __restrict__ g_f12z,
    double* fx, double* fy, double* fz)
{
    int n1=blockIdx.x*blockDim.x+threadIdx.x;
    if(n1>=N) return;
    double sfx=0,sfy=0,sfz=0; int num_nb=nn[n1];
    for(int i1=0;i1<num_nb;i1++){int idx=i1*N+n1; sfx+=g_f12x[idx];sfy+=g_f12y[idx];sfz+=g_f12z[idx];}
    for(int i1=0;i1<num_nb;i1++){
        int n2=nl[n1*max_nb+i1], num_nb2=nn[n2];
        for(int i2=0;i2<num_nb2;i2++){
            if(nl[n2*max_nb+i2]==n1){int idx=i2*N+n2; sfx-=g_f12x[idx];sfy-=g_f12y[idx];sfz-=g_f12z[idx];break;}
        }
    }
    fx[n1]=sfx; fy[n1]=sfy; fz[n1]=sfz;
}

// ═══════════════════════════════════════════════════════════════════════
// Force 3: Coulomb / Ewald (real-space + k-space)
// ═══════════════════════════════════════════════════════════════════════

__global__ void kernel_coulomb_real(
    int N, float alpha, const Box box,
    const int* __restrict__ nn, const int* __restrict__ nl, int max_nb,
    const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
    const float* __restrict__ charge,
    double* fx, double* fy, double* fz, double* pe)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    double xi=x[i],yi=y[i],zi=z[i]; float qi=charge[i];
    double sfx=0,sfy=0,sfz=0,spe=0;
    for(int k=0;k<nn[i];k++){
        int j=nl[i*max_nb+k];
        double dx=x[j]-xi,dy=y[j]-yi,dz=z[j]-zi;
        apply_mic(box,dx,dy,dz);
        double r=sqrt(dx*dx+dy*dy+dz*dz), rinv=1.0/r;
        float ar=alpha*(float)r, erfc_ar=erfcf(ar), exp_ar2=expf(-ar*ar);
        double f_r=K_COULOMB*qi*charge[j]*(erfc_ar*rinv*rinv+2.0f*alpha*0.5641895835f*exp_ar2*rinv)*rinv;
        sfx+=f_r*dx; sfy+=f_r*dy; sfz+=f_r*dz;
        spe+=K_COULOMB*qi*charge[j]*erfc_ar*rinv*0.5;
    }
    fx[i]=sfx; fy[i]=sfy; fz[i]=sfz; pe[i]=spe;
}

__global__ void kernel_structure_factor(
    int num_kpoints, int N, const float* __restrict__ charge,
    const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
    const float* __restrict__ kx_a, const float* __restrict__ ky_a, const float* __restrict__ kz_a,
    float* S_real, float* S_imag)
{
    int nk=blockIdx.x*blockDim.x+threadIdx.x;
    if(nk>=num_kpoints) return;
    float sr=0,si=0,kx=kx_a[nk],ky=ky_a[nk],kz=kz_a[nk];
    for(int n=0;n<N;n++){
        float kr=kx*(float)x[n]+ky*(float)y[n]+kz*(float)z[n], s,c;
        sincosf(kr,&s,&c); float q=charge[n]; sr+=q*c; si-=q*s;
    }
    S_real[nk]=sr; S_imag[nk]=si;
}

__global__ void kernel_kspace_force(
    int N, int num_kpoints, const float* __restrict__ charge,
    const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
    const float* __restrict__ kx_a, const float* __restrict__ ky_a, const float* __restrict__ kz_a,
    const float* __restrict__ G_a, const float* __restrict__ S_real, const float* __restrict__ S_imag,
    double* fx, double* fy, double* fz, double* pe)
{
    int n=blockIdx.x*blockDim.x+threadIdx.x;
    if(n>=N) return;
    float q=charge[n]; double sfx=0,sfy=0,sfz=0,spe=0;
    for(int nk=0;nk<num_kpoints;nk++){
        float kx=kx_a[nk],ky=ky_a[nk],kz=kz_a[nk];
        float kr=kx*(float)x[n]+ky*(float)y[n]+kz*(float)z[n], s,c;
        sincosf(kr,&s,&c); float G=G_a[nk];
        float imag=G*(S_real[nk]*s+S_imag[nk]*c);
        spe+=q*G*(S_real[nk]*c-S_imag[nk]*s);
        sfx+=2.0*q*kx*imag; sfy+=2.0*q*ky*imag; sfz+=2.0*q*kz*imag;
    }
    fx[n]+=K_COULOMB*sfx; fy[n]+=K_COULOMB*sfy; fz[n]+=K_COULOMB*sfz; pe[n]+=K_COULOMB*spe;
}

__global__ void kernel_self_energy(int N, float alpha, const float* __restrict__ charge, double* pe) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    float q=charge[i]; pe[i]-=K_COULOMB*alpha*0.5641895835f*q*q;
}
