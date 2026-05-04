/**
 * Unified GPU Molecular Dynamics Benchmark — CUDA-Hercules Class 3
 *
 * Reads real material structures from GPUMD-format model.xyz files.
 * Runs three material systems with different force pipelines:
 *   System 1: Argon — Lennard-Jones pairwise potential
 *   System 2: Silicon — Tersoff three-body potential
 *   System 3: NaCl — Coulomb/Ewald (real-space + k-space)
 *
 * Correctness: per-atom force comparison against pre-computed reference +
 *              energy conservation during MD run.
 * Performance: total kernel time across all 6 test cases.
 *
 * Usage:
 *   ./md_bench <data_dir>              # normal evaluation
 *   ./md_bench <data_dir> --save-ref   # generate reference force files
 */
#include "md_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

// ═══════════════════════════════════════════════════════════════════════
// Extended XYZ (model.xyz) parser
// ═══════════════════════════════════════════════════════════════════════

struct AtomData {
    int N;
    Box box;
    std::vector<double> x, y, z;
    std::vector<std::string> species;
    std::vector<float> charges;
};

bool read_model_xyz(const char* filename, AtomData& data) {
    std::ifstream f(filename);
    if (!f.is_open()) { printf("ERROR: cannot open %s\n", filename); return false; }
    std::string line;
    std::getline(f, line); data.N = std::stoi(line);
    std::getline(f, line);
    size_t lpos = line.find("Lattice=\"");
    if (lpos == std::string::npos) { printf("ERROR: no Lattice in %s\n", filename); return false; }
    size_t lstart = lpos + 9, lend = line.find("\"", lstart);
    std::string lat_str = line.substr(lstart, lend - lstart);
    double lat[9]; std::istringstream ls(lat_str);
    for (int i = 0; i < 9; i++) ls >> lat[i];
    data.box.Lx = lat[0]; data.box.Ly = lat[4]; data.box.Lz = lat[8];
    data.box.half_Lx = data.box.Lx / 2;
    data.box.half_Ly = data.box.Ly / 2;
    data.box.half_Lz = data.box.Lz / 2;
    data.x.resize(data.N); data.y.resize(data.N); data.z.resize(data.N);
    data.species.resize(data.N); data.charges.resize(data.N, 0.0f);
    for (int i = 0; i < data.N; i++) {
        std::getline(f, line); std::istringstream ss(line);
        ss >> data.species[i] >> data.x[i] >> data.y[i] >> data.z[i];
        if (data.species[i] == "Na") data.charges[i] = 1.0f;
        else if (data.species[i] == "Cl") data.charges[i] = -1.0f;
    }

    // Add deterministic perturbation to break perfect crystal symmetry.
    // Without this, net forces ≈ 0 due to cancellation, making force
    // comparison meaningless (GPU FP summation noise >> net force).
    // Perturbation: 1% of minimum lattice spacing, seeded RNG.
    double min_L = fmin(data.box.Lx, fmin(data.box.Ly, data.box.Lz));
    double perturb = 0.01 * min_L / cbrt((double)data.N); // ~0.01 * nearest-neighbor dist
    srand(54321 + data.N); // deterministic, unique per system size
    for (int i = 0; i < data.N; i++) {
        data.x[i] += perturb * (rand() / (double)RAND_MAX - 0.5);
        data.y[i] += perturb * (rand() / (double)RAND_MAX - 0.5);
        data.z[i] += perturb * (rand() / (double)RAND_MAX - 0.5);
        // Wrap into box
        if (data.x[i] < 0) data.x[i] += data.box.Lx;
        else if (data.x[i] >= data.box.Lx) data.x[i] -= data.box.Lx;
        if (data.y[i] < 0) data.y[i] += data.box.Ly;
        else if (data.y[i] >= data.box.Ly) data.y[i] -= data.box.Ly;
        if (data.z[i] < 0) data.z[i] += data.box.Lz;
        else if (data.z[i] >= data.box.Lz) data.z[i] -= data.box.Lz;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════

void init_velocities(int N, double T, double mass,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz) {
    vx.resize(N); vy.resize(N); vz.resize(N);
    srand(42);
    double scale = sqrt(T / mass), svx = 0, svy = 0, svz = 0;
    for (int i = 0; i < N; i++) {
        double u1=(rand()+1.0)/(RAND_MAX+2.0), u2=(rand()+1.0)/(RAND_MAX+2.0);
        double u3=(rand()+1.0)/(RAND_MAX+2.0), u4=(rand()+1.0)/(RAND_MAX+2.0);
        double u5=(rand()+1.0)/(RAND_MAX+2.0), u6=(rand()+1.0)/(RAND_MAX+2.0);
        vx[i]=scale*sqrt(-2*log(u1))*cos(2*M_PI*u2);
        vy[i]=scale*sqrt(-2*log(u3))*cos(2*M_PI*u4);
        vz[i]=scale*sqrt(-2*log(u5))*cos(2*M_PI*u6);
        svx+=vx[i]; svy+=vy[i]; svz+=vz[i];
    }
    svx/=N; svy/=N; svz/=N;
    for (int i = 0; i < N; i++) { vx[i]-=svx; vy[i]-=svy; vz[i]-=svz; }
}

double gpu_sum(const double* d, int N) {
    std::vector<double> h(N);
    cudaMemcpy(h.data(), d, N*sizeof(double), cudaMemcpyDeviceToHost);
    double s = 0; for (int i = 0; i < N; i++) s += h[i]; return s;
}

void generate_kvectors(double Lx, double Ly, double Lz, float alpha,
    std::vector<float>& kx, std::vector<float>& ky, std::vector<float>& kz, std::vector<float>& G_arr) {
    float twopi=2*M_PI, bx=twopi/Lx, by=twopi/Ly, bz=twopi/Lz;
    float ksq_max=twopi*twopi*alpha*alpha, alpha_factor=0.25f/(alpha*alpha);
    float volume=Lx*Ly*Lz;
    int nmax=(int)(alpha*fmax(Lx,fmax(Ly,Lz))/twopi)+1;
    for(int n1=0;n1<=nmax;n1++) for(int n2=-nmax;n2<=nmax;n2++) for(int n3=-nmax;n3<=nmax;n3++){
        if(n1==0&&n2==0&&n3==0) continue;
        if(n1==0&&n2<0) continue; if(n1==0&&n2==0&&n3<0) continue;
        float kvx=n1*bx,kvy=n2*by,kvz=n3*bz, ksq=kvx*kvx+kvy*kvy+kvz*kvz;
        if(ksq<ksq_max){kx.push_back(kvx);ky.push_back(kvy);kz.push_back(kvz);
            G_arr.push_back(2.0f*(twopi/volume)/ksq*exp(-ksq*alpha_factor));}
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Reference force I/O and comparison
// ═══════════════════════════════════════════════════════════════════════

bool save_ref_forces(const char* filename, int N,
    const double* d_fx, const double* d_fy, const double* d_fz, const double* d_pe) {
    std::vector<double> fx(N), fy(N), fz(N), pe(N);
    cudaMemcpy(fx.data(), d_fx, N*8, cudaMemcpyDeviceToHost);
    cudaMemcpy(fy.data(), d_fy, N*8, cudaMemcpyDeviceToHost);
    cudaMemcpy(fz.data(), d_fz, N*8, cudaMemcpyDeviceToHost);
    cudaMemcpy(pe.data(), d_pe, N*8, cudaMemcpyDeviceToHost);
    FILE* f = fopen(filename, "wb");
    if (!f) { printf("ERROR: cannot write %s\n", filename); return false; }
    fwrite(&N, 4, 1, f);
    fwrite(fx.data(), 8, N, f);
    fwrite(fy.data(), 8, N, f);
    fwrite(fz.data(), 8, N, f);
    fwrite(pe.data(), 8, N, f);
    fclose(f);
    printf("  Saved reference forces: %s (%d atoms)\n", filename, N);
    return true;
}

struct ForceRef {
    int N;
    std::vector<double> fx, fy, fz, pe;
    bool loaded;
};

bool load_ref_forces(const char* filename, ForceRef& ref) {
    FILE* f = fopen(filename, "rb");
    if (!f) { ref.loaded = false; return false; }
    fread(&ref.N, 4, 1, f);
    ref.fx.resize(ref.N); ref.fy.resize(ref.N); ref.fz.resize(ref.N); ref.pe.resize(ref.N);
    fread(ref.fx.data(), 8, ref.N, f);
    fread(ref.fy.data(), 8, ref.N, f);
    fread(ref.fz.data(), 8, ref.N, f);
    fread(ref.pe.data(), 8, ref.N, f);
    fclose(f);
    ref.loaded = true;
    return true;
}

struct ForceCompareResult {
    bool checked;
    bool correct;
    double max_abs_err;   // max |F_gpu - F_ref| across all atoms/components
    double rms_err;       // RMS of force differences
    double max_rel_err;   // max |F_gpu - F_ref| / |F_ref| per atom
    double max_pe_err;    // max |PE_gpu - PE_ref|
};

ForceCompareResult compare_forces(int N, const double* d_fx, const double* d_fy,
    const double* d_fz, const double* d_pe, const ForceRef& ref, double tol) {
    ForceCompareResult c = {true, true, 0, 0, 0, 0};
    if (!ref.loaded || ref.N != N) { c.checked = false; c.correct = false; return c; }

    std::vector<double> fx(N), fy(N), fz(N), pe(N);
    cudaMemcpy(fx.data(), d_fx, N*8, cudaMemcpyDeviceToHost);
    cudaMemcpy(fy.data(), d_fy, N*8, cudaMemcpyDeviceToHost);
    cudaMemcpy(fz.data(), d_fz, N*8, cudaMemcpyDeviceToHost);
    cudaMemcpy(pe.data(), d_pe, N*8, cudaMemcpyDeviceToHost);

    // First pass: compute average force magnitude for threshold
    double sum_fmag = 0;
    for (int i = 0; i < N; i++) {
        sum_fmag += sqrt(ref.fx[i]*ref.fx[i] + ref.fy[i]*ref.fy[i] + ref.fz[i]*ref.fz[i]);
    }
    double avg_fmag = sum_fmag / N;
    // Skip relative error for atoms with |F| < 1e-6 * avg (e.g. symmetric crystal sites)
    double fmag_floor = avg_fmag * 1e-6;

    double sum_sq = 0;
    int n_rel_checked = 0;
    for (int i = 0; i < N; i++) {
        double efx = fabs(fx[i] - ref.fx[i]);
        double efy = fabs(fy[i] - ref.fy[i]);
        double efz = fabs(fz[i] - ref.fz[i]);
        double epe = fabs(pe[i] - ref.pe[i]);
        double e = fmax(efx, fmax(efy, efz));
        double fmag = sqrt(ref.fx[i]*ref.fx[i] + ref.fy[i]*ref.fy[i] + ref.fz[i]*ref.fz[i]);

        if (e > c.max_abs_err) c.max_abs_err = e;
        if (epe > c.max_pe_err) c.max_pe_err = epe;
        sum_sq += efx*efx + efy*efy + efz*efz;

        // Relative error only for atoms with non-negligible force
        if (fmag > fmag_floor) {
            double rel = e / fmag;
            if (rel > c.max_rel_err) c.max_rel_err = rel;
            n_rel_checked++;
        }
    }
    c.rms_err = sqrt(sum_sq / (3.0 * N));
    c.correct = (c.max_rel_err < tol);
    return c;
}

// ═══════════════════════════════════════════════════════════════════════
// GPU state and helpers
// ═══════════════════════════════════════════════════════════════════════

struct GPUState {
    int N;
    double *d_x,*d_y,*d_z,*d_vx,*d_vy,*d_vz,*d_fx,*d_fy,*d_fz,*d_pe,*d_ke;
    int *d_nn,*d_nl,*d_cc,*d_ci;
    int max_nb, cx, cy, cz, nc, mpc, grid;
    double cell_size;
    Box box;

    void alloc(const AtomData& data, int max_neighbors, double cs, int max_per_cell) {
        N = data.N; box = data.box; max_nb = max_neighbors; cell_size = cs; mpc = max_per_cell;
        grid = (N + MD_BLOCK_SIZE - 1) / MD_BLOCK_SIZE;
        cx = fmax(3, (int)(box.Lx/cs)); cy = fmax(3, (int)(box.Ly/cs)); cz = fmax(3, (int)(box.Lz/cs));
        nc = cx*cy*cz;
        cudaMalloc(&d_x,N*8);cudaMalloc(&d_y,N*8);cudaMalloc(&d_z,N*8);
        cudaMalloc(&d_vx,N*8);cudaMalloc(&d_vy,N*8);cudaMalloc(&d_vz,N*8);
        cudaMalloc(&d_fx,N*8);cudaMalloc(&d_fy,N*8);cudaMalloc(&d_fz,N*8);
        cudaMalloc(&d_pe,N*8);cudaMalloc(&d_ke,N*8);
        cudaMalloc(&d_nn,N*4);cudaMalloc(&d_nl,N*max_nb*4);
        cudaMalloc(&d_cc,nc*4);cudaMalloc(&d_ci,nc*mpc*4);
    }
    void upload(const AtomData& data, const std::vector<double>& vx,
                const std::vector<double>& vy, const std::vector<double>& vz) {
        cudaMemcpy(d_x,data.x.data(),N*8,cudaMemcpyHostToDevice);
        cudaMemcpy(d_y,data.y.data(),N*8,cudaMemcpyHostToDevice);
        cudaMemcpy(d_z,data.z.data(),N*8,cudaMemcpyHostToDevice);
        cudaMemcpy(d_vx,vx.data(),N*8,cudaMemcpyHostToDevice);
        cudaMemcpy(d_vy,vy.data(),N*8,cudaMemcpyHostToDevice);
        cudaMemcpy(d_vz,vz.data(),N*8,cudaMemcpyHostToDevice);
    }
    // GPU neighbor list build (fast, for MD steps — non-deterministic order is OK)
    void build_nb() {
        cudaMemset(d_cc,0,nc*4);
        kernel_assign_cells<<<grid,MD_BLOCK_SIZE>>>(N,d_x,d_y,d_z,box,cx,cy,cz,cell_size,d_cc,d_ci,mpc);
        kernel_build_neighbor_list<<<grid,MD_BLOCK_SIZE>>>(N,d_x,d_y,d_z,box,cell_size*cell_size,cx,cy,cz,cell_size,d_cc,d_ci,mpc,d_nn,d_nl,max_nb);
    }

    // CPU neighbor list build (deterministic, for reference force comparison)
    void build_nb_deterministic() {
        std::vector<double> hx(N), hy(N), hz(N);
        cudaMemcpy(hx.data(), d_x, N*8, cudaMemcpyDeviceToHost);
        cudaMemcpy(hy.data(), d_y, N*8, cudaMemcpyDeviceToHost);
        cudaMemcpy(hz.data(), d_z, N*8, cudaMemcpyDeviceToHost);
        double csq = cell_size * cell_size;
        // Cell list on CPU
        std::vector<std::vector<int>> cells(nc);
        for (int i = 0; i < N; i++) {
            int ix=((int)(hx[i]/cell_size)%cx+cx)%cx;
            int iy=((int)(hy[i]/cell_size)%cy+cy)%cy;
            int iz=((int)(hz[i]/cell_size)%cz+cz)%cz;
            cells[ix*cy*cz+iy*cz+iz].push_back(i);
        }
        // Neighbor list on CPU (sorted by atom index)
        std::vector<int> h_nn(N, 0), h_nl(N * max_nb, 0);
        for (int i = 0; i < N; i++) {
            double xi=hx[i], yi=hy[i], zi=hz[i];
            int iix=((int)(xi/cell_size)%cx+cx)%cx;
            int iiy=((int)(yi/cell_size)%cy+cy)%cy;
            int iiz=((int)(zi/cell_size)%cz+cz)%cz;
            int cnt = 0;
            for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) for (int dz=-1;dz<=1;dz++) {
                int jx=(iix+dx+cx)%cx, jy=(iiy+dy+cy)%cy, jz=(iiz+dz+cz)%cz;
                for (int j : cells[jx*cy*cz+jy*cz+jz]) {
                    if (j==i) continue;
                    double rx=hx[j]-xi, ry=hy[j]-yi, rz=hz[j]-zi;
                    if(rx>box.half_Lx)rx-=box.Lx; else if(rx<-box.half_Lx)rx+=box.Lx;
                    if(ry>box.half_Ly)ry-=box.Ly; else if(ry<-box.half_Ly)ry+=box.Ly;
                    if(rz>box.half_Lz)rz-=box.Lz; else if(rz<-box.half_Lz)rz+=box.Lz;
                    if(rx*rx+ry*ry+rz*rz < csq && cnt < max_nb) {
                        h_nl[i*max_nb+cnt]=j; cnt++;
                    }
                }
            }
            std::sort(&h_nl[i*max_nb], &h_nl[i*max_nb]+cnt);
            h_nn[i] = cnt;
        }
        cudaMemcpy(d_nn, h_nn.data(), N*4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_nl, h_nl.data(), N*max_nb*4, cudaMemcpyHostToDevice);
        // Also fill cell arrays for GPU (needed if GPU build_nb is called later)
        cudaMemset(d_cc,0,nc*4);
        kernel_assign_cells<<<grid,MD_BLOCK_SIZE>>>(N,d_x,d_y,d_z,box,cx,cy,cz,cell_size,d_cc,d_ci,mpc);
        cudaDeviceSynchronize();
    }
    double total_energy(double mass) {
        kernel_kinetic_energy<<<grid,MD_BLOCK_SIZE>>>(N,mass,d_vx,d_vy,d_vz,d_ke);
        cudaDeviceSynchronize();
        return gpu_sum(d_pe,N) + gpu_sum(d_ke,N);
    }
    void cleanup() {
        cudaFree(d_x);cudaFree(d_y);cudaFree(d_z);cudaFree(d_vx);cudaFree(d_vy);cudaFree(d_vz);
        cudaFree(d_fx);cudaFree(d_fy);cudaFree(d_fz);cudaFree(d_pe);cudaFree(d_ke);
        cudaFree(d_nn);cudaFree(d_nl);cudaFree(d_cc);cudaFree(d_ci);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Run result
// ═══════════════════════════════════════════════════════════════════════

struct RunResult {
    float time_ms;
    double energy_drift;
    bool energy_passed;
    ForceCompareResult force;
};

// ═══════════════════════════════════════════════════════════════════════
// Simulation runners
// ═══════════════════════════════════════════════════════════════════════

RunResult run_lj(const char* xyz_file, int steps, double dt, double T, double mass,
                 const char* ref_file, bool save_ref, double force_tol) {
    RunResult res = {}; res.force.checked = false;
    AtomData data; if(!read_model_xyz(xyz_file, data)) return res;
    std::vector<double> vx,vy,vz; init_velocities(data.N,T,mass,vx,vy,vz);
    LJParams lj; lj.cutoff = 8.5; lj.cutoff_sq = lj.cutoff * lj.cutoff;
    lj.sigma6_x4eps = 4.0*0.0104*pow(3.4,6); lj.sigma12_x4eps = 4.0*0.0104*pow(3.4,12);

    GPUState g; g.alloc(data, MAX_NEIGHBORS_LJ, lj.cutoff, 64); g.upload(data, vx, vy, vz);
    auto compute=[&](){ cudaMemset(g.d_pe,0,g.N*8);
        kernel_lj_force<<<g.grid,MD_BLOCK_SIZE>>>(g.N,lj,g.box,g.d_nn,g.d_nl,g.max_nb,g.d_x,g.d_y,g.d_z,g.d_fx,g.d_fy,g.d_fz,g.d_pe); };
    g.build_nb_deterministic(); compute(); cudaDeviceSynchronize();

    // Force reference check
    if (save_ref) {
        save_ref_forces(ref_file, g.N, g.d_fx, g.d_fy, g.d_fz, g.d_pe);
    } else {
        ForceRef ref;
        if (load_ref_forces(ref_file, ref))
            res.force = compare_forces(g.N, g.d_fx, g.d_fy, g.d_fz, g.d_pe, ref, force_tol);
    }

    // MD run
    double E0 = g.total_energy(mass);
    cudaEvent_t ts,te; cudaEventCreate(&ts); cudaEventCreate(&te); cudaEventRecord(ts);
    double mi = 1.0/mass;
    for(int s=0;s<steps;s++){
        kernel_verlet_step1<<<g.grid,MD_BLOCK_SIZE>>>(g.N,dt,mi,g.box,g.d_fx,g.d_fy,g.d_fz,g.d_x,g.d_y,g.d_z,g.d_vx,g.d_vy,g.d_vz);
        if(s%10==0) g.build_nb(); compute();
        kernel_verlet_step2<<<g.grid,MD_BLOCK_SIZE>>>(g.N,dt,mi,g.d_fx,g.d_fy,g.d_fz,g.d_vx,g.d_vy,g.d_vz);
    }
    cudaEventRecord(te); cudaEventSynchronize(te); cudaEventElapsedTime(&res.time_ms,ts,te);
    double E1 = g.total_energy(mass);
    double es=fmax(fabs(E0),fabs(E1))/g.N;
    res.energy_drift=(es>1e-10)?fabs((E1-E0)/g.N)/es:0;
    res.energy_passed = res.energy_drift < 0.01;
    g.cleanup(); cudaEventDestroy(ts); cudaEventDestroy(te);
    return res;
}

RunResult run_tersoff(const char* xyz_file, int steps, double dt, double T, double mass,
                      const char* ref_file, bool save_ref, double force_tol) {
    RunResult res = {}; res.force.checked = false;
    AtomData data; if(!read_model_xyz(xyz_file, data)) return res;
    std::vector<double> vx,vy,vz; init_velocities(data.N,T,mass,vx,vy,vz);
    TersoffParams tp = si_tersoff_params();
    GPUState g; g.alloc(data, MAX_NEIGHBORS_TERSOFF, tp.R2, 32); g.upload(data, vx, vy, vz);
    double *d_b,*d_bp,*d_f12x,*d_f12y,*d_f12z; int mp=g.N*MAX_NEIGHBORS_TERSOFF;
    cudaMalloc(&d_b,mp*8);cudaMalloc(&d_bp,mp*8);
    cudaMalloc(&d_f12x,mp*8);cudaMalloc(&d_f12y,mp*8);cudaMalloc(&d_f12z,mp*8);
    auto compute=[&](){ cudaMemset(g.d_pe,0,g.N*8);
        kernel_tersoff_step1<<<g.grid,MD_BLOCK_SIZE>>>(g.N,tp,g.box,g.d_nn,g.d_nl,g.max_nb,g.d_x,g.d_y,g.d_z,d_b,d_bp);
        kernel_tersoff_step2<<<g.grid,MD_BLOCK_SIZE>>>(g.N,tp,g.box,g.d_nn,g.d_nl,g.max_nb,d_b,d_bp,g.d_x,g.d_y,g.d_z,g.d_pe,d_f12x,d_f12y,d_f12z);
        kernel_accumulate_forces<<<g.grid,MD_BLOCK_SIZE>>>(g.N,g.d_nn,g.d_nl,g.max_nb,d_f12x,d_f12y,d_f12z,g.d_fx,g.d_fy,g.d_fz); };
    g.build_nb_deterministic(); compute(); cudaDeviceSynchronize();

    if (save_ref) {
        save_ref_forces(ref_file, g.N, g.d_fx, g.d_fy, g.d_fz, g.d_pe);
    } else {
        ForceRef ref;
        if (load_ref_forces(ref_file, ref))
            res.force = compare_forces(g.N, g.d_fx, g.d_fy, g.d_fz, g.d_pe, ref, force_tol);
    }

    double E0 = g.total_energy(mass);
    cudaEvent_t ts,te; cudaEventCreate(&ts); cudaEventCreate(&te); cudaEventRecord(ts);
    double mi=1.0/mass;
    for(int s=0;s<steps;s++){
        kernel_verlet_step1<<<g.grid,MD_BLOCK_SIZE>>>(g.N,dt,mi,g.box,g.d_fx,g.d_fy,g.d_fz,g.d_x,g.d_y,g.d_z,g.d_vx,g.d_vy,g.d_vz);
        if(s%5==0) g.build_nb(); compute();
        kernel_verlet_step2<<<g.grid,MD_BLOCK_SIZE>>>(g.N,dt,mi,g.d_fx,g.d_fy,g.d_fz,g.d_vx,g.d_vy,g.d_vz);
    }
    cudaEventRecord(te); cudaEventSynchronize(te); cudaEventElapsedTime(&res.time_ms,ts,te);
    double E1 = g.total_energy(mass);
    double es=fmax(fabs(E0),fabs(E1))/g.N;
    res.energy_drift=(es>1e-10)?fabs((E1-E0)/g.N)/es:0;
    res.energy_passed = res.energy_drift < 0.01;
    cudaFree(d_b);cudaFree(d_bp);cudaFree(d_f12x);cudaFree(d_f12y);cudaFree(d_f12z);
    g.cleanup(); cudaEventDestroy(ts); cudaEventDestroy(te);
    return res;
}

RunResult run_coulomb(const char* xyz_file, int steps, double dt, double T, double mass,
                      const char* ref_file, bool save_ref, double force_tol) {
    RunResult res = {}; res.force.checked = false;
    AtomData data; if(!read_model_xyz(xyz_file, data)) return res;
    std::vector<double> vx,vy,vz; init_velocities(data.N,T,mass,vx,vy,vz);
    float alpha=0.3; double cutoff=7.0;
    std::vector<float> h_kx,h_ky,h_kz,h_G;
    generate_kvectors(data.box.Lx,data.box.Ly,data.box.Lz,alpha,h_kx,h_ky,h_kz,h_G);
    int nk=h_kx.size();
    GPUState g; g.alloc(data, MAX_NEIGHBORS_COULOMB, cutoff, 64); g.upload(data, vx, vy, vz);
    float *d_charge,*d_kx,*d_ky,*d_kz,*d_G,*d_Sr,*d_Si;
    cudaMalloc(&d_charge,g.N*4); cudaMemcpy(d_charge,data.charges.data(),g.N*4,cudaMemcpyHostToDevice);
    cudaMalloc(&d_kx,nk*4);cudaMalloc(&d_ky,nk*4);cudaMalloc(&d_kz,nk*4);
    cudaMalloc(&d_G,nk*4);cudaMalloc(&d_Sr,nk*4);cudaMalloc(&d_Si,nk*4);
    cudaMemcpy(d_kx,h_kx.data(),nk*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ky,h_ky.data(),nk*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_kz,h_kz.data(),nk*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_G,h_G.data(),nk*4,cudaMemcpyHostToDevice);
    int grid_k=(nk+MD_BLOCK_SIZE-1)/MD_BLOCK_SIZE;
    auto compute=[&](){
        cudaMemset(g.d_fx,0,g.N*8);cudaMemset(g.d_fy,0,g.N*8);cudaMemset(g.d_fz,0,g.N*8);cudaMemset(g.d_pe,0,g.N*8);
        kernel_coulomb_real<<<g.grid,MD_BLOCK_SIZE>>>(g.N,alpha,g.box,g.d_nn,g.d_nl,g.max_nb,g.d_x,g.d_y,g.d_z,d_charge,g.d_fx,g.d_fy,g.d_fz,g.d_pe);
        kernel_structure_factor<<<grid_k,MD_BLOCK_SIZE>>>(nk,g.N,d_charge,g.d_x,g.d_y,g.d_z,d_kx,d_ky,d_kz,d_Sr,d_Si);
        kernel_kspace_force<<<g.grid,MD_BLOCK_SIZE>>>(g.N,nk,d_charge,g.d_x,g.d_y,g.d_z,d_kx,d_ky,d_kz,d_G,d_Sr,d_Si,g.d_fx,g.d_fy,g.d_fz,g.d_pe);
        kernel_self_energy<<<g.grid,MD_BLOCK_SIZE>>>(g.N,alpha,d_charge,g.d_pe); };
    g.build_nb_deterministic(); compute(); cudaDeviceSynchronize();

    if (save_ref) {
        save_ref_forces(ref_file, g.N, g.d_fx, g.d_fy, g.d_fz, g.d_pe);
    } else {
        ForceRef ref;
        if (load_ref_forces(ref_file, ref))
            res.force = compare_forces(g.N, g.d_fx, g.d_fy, g.d_fz, g.d_pe, ref, force_tol);
    }

    double E0 = g.total_energy(mass);
    cudaEvent_t ts,te; cudaEventCreate(&ts); cudaEventCreate(&te); cudaEventRecord(ts);
    double mi=1.0/mass;
    for(int s=0;s<steps;s++){
        kernel_verlet_step1<<<g.grid,MD_BLOCK_SIZE>>>(g.N,dt,mi,g.box,g.d_fx,g.d_fy,g.d_fz,g.d_x,g.d_y,g.d_z,g.d_vx,g.d_vy,g.d_vz);
        if(s%5==0) g.build_nb(); compute();
        kernel_verlet_step2<<<g.grid,MD_BLOCK_SIZE>>>(g.N,dt,mi,g.d_fx,g.d_fy,g.d_fz,g.d_vx,g.d_vy,g.d_vz);
    }
    cudaEventRecord(te); cudaEventSynchronize(te); cudaEventElapsedTime(&res.time_ms,ts,te);
    double E1 = g.total_energy(mass);
    double es=fmax(fabs(E0),fabs(E1))/g.N;
    res.energy_drift=(es>1e-10)?fabs((E1-E0)/g.N)/es:0;
    res.energy_passed = res.energy_drift < 0.01;
    cudaFree(d_charge);cudaFree(d_kx);cudaFree(d_ky);cudaFree(d_kz);cudaFree(d_G);cudaFree(d_Sr);cudaFree(d_Si);
    g.cleanup(); cudaEventDestroy(ts); cudaEventDestroy(te);
    return res;
}

// ═══════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    const char* data_dir = "../data";
    bool save_ref = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--save-ref") == 0) save_ref = true;
        else data_dir = argv[i];
    }

    printf("=== GPU Molecular Dynamics Benchmark (3 Materials) ===\n");
    printf("Data directory: %s\n", data_dir);
    if (save_ref) printf("Mode: saving reference forces\n");
    printf("\n");

    // Ensure ref_forces directory exists
    char ref_dir[512];
    snprintf(ref_dir, sizeof(ref_dir), "%s/ref_forces", data_dir);
    if (save_ref) {
        char cmd[600]; snprintf(cmd, sizeof(cmd), "mkdir -p %s", ref_dir);
        system(cmd);
    }

    bool all_passed = true;
    float grand_total = 0;
    int n_tests = 0, n_force_pass = 0, n_energy_pass = 0;

    // Force tolerance: LJ/Tersoff use double → tight; Coulomb uses float k-space → looser
    struct Test {
        const char* name; const char* file; const char* ref_name;
        RunResult(*fn)(const char*,int,double,double,double,const char*,bool,double);
        int steps; double dt, T, mass, force_tol;
    };

    char path[512], ref_path[512];

    Test tests[] = {
        {"LJ_Ar_256K",      "Ar_256000.xyz",   "lj_Ar_256000.bin",       run_lj,       100, 0.001,  0.5,  39.948, 1e-4},
        {"LJ_Ar_500K",      "Ar_500000.xyz",   "lj_Ar_500000.bin",       run_lj,        50, 0.001,  0.5,  39.948, 1e-4},
        {"Tersoff_Si_27K",   "Si_27000.xyz",    "tersoff_Si_27000.bin",   run_tersoff,   30, 0.0005, 0.1,  28.0855, 1e-4},
        {"Tersoff_Si_64K",   "Si_64000.xyz",    "tersoff_Si_64000.bin",   run_tersoff,   20, 0.0005, 0.1,  28.0855, 1e-4},
        {"Coulomb_NaCl_8K",  "NaCl_8000.xyz",   "coulomb_NaCl_8000.bin",  run_coulomb,   30, 0.001,  0.05, 29.22,  1e-2},
        {"Coulomb_NaCl_32K", "NaCl_32768.xyz",  "coulomb_NaCl_32768.bin", run_coulomb,   20, 0.001,  0.05, 29.22,  1e-2},
    };

    for (auto& t : tests) {
        snprintf(path, sizeof(path), "%s/%s", data_dir, t.file);
        snprintf(ref_path, sizeof(ref_path), "%s/%s", ref_dir, t.ref_name);
        printf("--- %s (%s) ---\n", t.name, t.file);

        auto r = t.fn(path, t.steps, t.dt, t.T, t.mass, ref_path, save_ref, t.force_tol);

        if (save_ref) continue;

        n_tests++;
        bool test_pass = true;

        // Force check
        if (r.force.checked) {
            bool fp = r.force.correct;
            if (fp) n_force_pass++;
            else test_pass = false;
            printf("  Force check: max_rel_err=%.6e max_abs_err=%.6e rms_err=%.6e [%s]\n",
                   r.force.max_rel_err, r.force.max_abs_err, r.force.rms_err,
                   fp ? "PASS" : "FAIL");
        } else {
            printf("  Force check: no reference available\n");
        }

        // Energy conservation
        bool ep = r.energy_passed;
        if (ep) n_energy_pass++;
        else test_pass = false;
        printf("  MD run: time=%.2f ms, energy_drift=%.6e [%s]\n",
               r.time_ms, r.energy_drift, ep ? "PASS" : "FAIL");

        // Structured output line
        int correct = test_pass ? 1 : 0;
        int fc = r.force.checked ? (r.force.correct ? 1 : 0) : -1;
        printf("KERNEL %s: force_correct=%d max_rel_err=%.6e max_abs_err=%.6e "
               "energy_drift=%.6e time_ms=%.4f correct=%d\n",
               t.name, fc, r.force.max_rel_err, r.force.max_abs_err,
               r.energy_drift, r.time_ms, correct);

        grand_total += r.time_ms;
        if (!test_pass) all_passed = false;
    }

    if (save_ref) {
        printf("\nReference forces saved.\n");
        return 0;
    }

    printf("\n=== Summary ===\n");
    printf("Force check: %d/%d passed\n", n_force_pass, n_tests);
    printf("Energy check: %d/%d passed\n", n_energy_pass, n_tests);
    printf("Passed kernels: %d/%d\n", (int)(all_passed ? n_tests : 0), n_tests);

    if (all_passed)
        printf("Passed\n");
    else
        printf("FAILED\n");

    printf("Kernel time: %.4f ms\n", grand_total);
    return all_passed ? 0 : 1;
}
