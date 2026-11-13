// Device helpers shared by the factorization (ref_solve.cu):
// complex FP32 arithmetic and the one-block partial LDL^T of a panel (kernel v2's panel routine).
#pragma once
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__device__ __forceinline__ cuComplex cmul(cuComplex a, cuComplex b)
{
    return make_cuComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ cuComplex cdiv(cuComplex a, cuComplex b)
{
    const float den = b.x * b.x + b.y * b.y;
    return make_cuComplex((a.x * b.x + a.y * b.y) / den, (a.y * b.x - a.x * b.y) / den);
}

// 1 / b with a single division, which stays on its fast path: the slow path of an IEEE division is taken whenever
// the dividend is zero, as it is for idle lanes and for the zero entries of a front, so a / b is cmul(a, crcp(b))
__device__ __forceinline__ cuComplex crcp(cuComplex b)
{
    const float r = 1.0f / (b.x * b.x + b.y * b.y);
    return make_cuComplex(b.x * r, -b.y * r);
}

constexpr int NB = 16;  // panel block columns

constexpr int PC = 64;  // inner chunk of the block-column update
constexpr int KC = 16;  // inner chunk of the Schur update
constexpr int kShElems = 2048;  // shared memory of a v2 block, in complex values (phases reuse it)
static_assert(NB * PC + 2 * NB * (NB + 1) + NB <= kShElems && 2 * KC * 64 <= kShElems, "v2 shared memory");

__device__ __forceinline__ cuComplex csubmul(cuComplex acc, cuComplex a, cuComplex b)  // acc - a * b
{
    acc.x = fmaf(-a.x, b.x, fmaf(a.y, b.y, acc.x));
    acc.y = fmaf(-a.x, b.y, fmaf(-a.y, b.x, acc.y));
    return acc;
}

__device__ __forceinline__ cuComplex caddmul(cuComplex acc, cuComplex a, cuComplex b)  // acc + a * b
{
    acc.x = fmaf(a.x, b.x, fmaf(-a.y, b.y, acc.x));
    acc.y = fmaf(a.x, b.y, fmaf(a.y, b.x, acc.y));
    return acc;
}

__device__ __forceinline__ cuComplex shflC(cuComplex v, int src)
{
    return make_cuComplex(__shfl_sync(0xffffffffu, v.x, src), __shfl_sync(0xffffffffu, v.y, src));
}

// Partial LDL^T of the k pivot columns of the panel P (m x k, column-major, leading dimension ld) by one
// block of 256 threads: L below the diagonal, D on it.
__device__ void panelFactor(cuComplex* P, int64_t ld, int m, int k, cuComplex* sh, unsigned* zeroPivots)
{
    cuComplex (*sW)[PC] = reinterpret_cast<cuComplex (*)[PC]>(sh);                             // L[jb+c, p] d_p
    cuComplex (*sD)[NB + 1] = reinterpret_cast<cuComplex (*)[NB + 1]>(sh + NB * PC);             // diagonal block
    cuComplex (*sL)[NB + 1] = reinterpret_cast<cuComplex (*)[NB + 1]>(sh + NB * PC + NB * (NB + 1));  // L_cp d_p
    cuComplex* sDi = sh + NB * PC + 2 * NB * (NB + 1);                                           // 1 / d_c
    const cuComplex zero = make_cuComplex(0, 0);
    const int tid = threadIdx.x, nt = blockDim.x, lane = tid & 31;
    for (int jb = 0; jb < k; jb += NB) {
        const int jn = min(NB, k - jb);
        // 1. block column [jb, jb+jn), rows >= jb, minus the finished columns p < jb
        for (int p0 = 0; p0 < jb; p0 += PC) {
            const int pn = min(PC, jb - p0);
            for (int e = tid; e < NB * PC; e += nt) {
                const int c = e % NB, pp = e / NB;
                sW[c][pp] = c < jn && pp < pn ? cmul(P[(jb + c) + ld * (p0 + pp)], P[(int64_t)(p0 + pp) * (ld + 1)])
                                              : zero;
            }
            __syncthreads();
            for (int i = jb + tid; i < m; i += nt) {
                cuComplex acc[NB];
#pragma unroll
                for (int c = 0; c < NB; ++c) acc[c] = c < jn ? P[i + ld * (jb + c)] : zero;
                for (int pp = 0; pp < pn; ++pp) {
                    const cuComplex l = P[i + ld * (p0 + pp)];
#pragma unroll
                    for (int c = 0; c < NB; ++c) acc[c] = csubmul(acc[c], l, sW[c][pp]);
                }
#pragma unroll
                for (int c = 0; c < NB; ++c)
                    if (c < jn && jb + c <= i) P[i + ld * (jb + c)] = acc[c];
            }
            __syncthreads();
        }
        // 2. LDL^T of the diagonal block by warp 0, lane r owning row r, loaded by all threads at once
        for (int e = tid; e < NB * NB; e += nt) {
            const int r = e % NB, c = e / NB;
            if (r < jn && c <= r) sD[r][c] = P[(jb + r) + ld * (jb + c)];
        }
        __syncthreads();
        if (tid < 32) {
            for (int c = 0; c < jn; ++c) {
                cuComplex d = sD[c][c];
                if (d.x == 0 && d.y == 0) {
                    d = make_cuComplex(1e-30f, 0);
                    if (lane == 0) atomicAdd(zeroPivots, 1u);
                }
                const bool mine = lane > c && lane < jn;
                const cuComplex arc = mine ? sD[lane][c] : zero;  // a_rc before the division
                const cuComplex lrc = cmul(arc, crcp(d));
                __syncwarp();
                for (int c2 = c + 1; c2 < jn; ++c2) {
                    const cuComplex a2 = shflC(arc, c2);  // a_{c2,c}
                    if (mine && lane >= c2) sD[lane][c2] = csubmul(sD[lane][c2], lrc, a2);
                }
                if (mine) sD[lane][c] = lrc;
                if (lane == c) sD[c][c] = d;
                __syncwarp();
            }
            if (lane < jn) {
                for (int p = 0; p < lane; ++p) sL[lane][p] = cmul(sD[lane][p], sD[p][p]);
                sDi[lane] = crcp(sD[lane][lane]);
                for (int c = 0; c <= lane; ++c) P[(jb + lane) + ld * (jb + c)] = sD[lane][c];
            }
        }
        __syncthreads();
        // 3. rows below the diagonal block: L[i, J] = A[i, J] L_JJ^-T D_J^-1
        for (int i = jb + jn + tid; i < m; i += nt) {
            cuComplex x[NB];
#pragma unroll
            for (int c = 0; c < NB; ++c) x[c] = c < jn ? P[i + ld * (jb + c)] : zero;
#pragma unroll
            for (int c = 0; c < NB; ++c) {
                if (c < jn) {
                    cuComplex t = x[c];
#pragma unroll
                    for (int p = 0; p < c; ++p) t = csubmul(t, x[p], sL[c][p]);
                    x[c] = cmul(t, sDi[c]);
                }
            }
#pragma unroll
            for (int c = 0; c < NB; ++c)
                if (c < jn) P[i + ld * (jb + c)] = x[c];
        }
        __syncthreads();
    }
}


// ---------------- large fronts: dense partial LDL^T with cuBLAS

__global__ void diagKer(cuComplex* F, int64_t ld, int jn, unsigned* zeroPivots)
{
    __shared__ __align__(16) cuComplex sh[kShElems];
    panelFactor(F, ld, jn, jn, sh, zeroPivots);
}

// the same with the diagonal block staged in shared memory (jn <= kDiagShMax)
constexpr int kDiagShMax = 96;
__global__ void diagShKer(cuComplex* F, int64_t ld, int jn, unsigned* zeroPivots)
{
    extern __shared__ __align__(16) cuComplex dyn[];
    cuComplex* sh = dyn;
    cuComplex* B = dyn + kShElems;  // jn x jn, leading dimension jn
    for (int idx = threadIdx.x; idx < jn * jn; idx += blockDim.x) {
        const int i = idx % jn, j = idx / jn;
        B[idx] = i >= j ? F[i + ld * j] : make_cuComplex(0, 0);
    }
    __syncthreads();
    panelFactor(B, jn, jn, jn, sh, zeroPivots);
    __syncthreads();
    for (int idx = threadIdx.x; idx < jn * jn; idx += blockDim.x) {
        const int i = idx % jn, j = idx / jn;
        if (i >= j) F[i + ld * j] = B[idx];
    }
}

// W = F21 (workspace, leading dimension nr), F21 = F21 D^-1 for the nr x jn block below the diagonal block
__global__ void scaleKer(cuComplex* F21, int64_t ld, const cuComplex* Fdiag, int nr, int jn, cuComplex* W)
{
    __shared__ cuComplex dinv[256];
    if (threadIdx.x < jn) dinv[threadIdx.x] = crcp(Fdiag[(int64_t)threadIdx.x * (ld + 1)]);
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr) return;
    for (int c = 0; c < jn; ++c) {
        const cuComplex v = F21[i + ld * c];
        W[i + (int64_t)nr * c] = v;
        F21[i + ld * c] = cmul(v, dinv[c]);
    }
}

// Lower triangle of the m x m front F (leading dimension m) to zero, one block per column
__global__ void frontZeroKer(cuComplex* F, int m)
{
    for (int j = blockIdx.x; j < m; j += gridDim.x)
        for (int i = j + threadIdx.x; i < m; i += blockDim.x) F[i + (int64_t)m * j] = make_cuComplex(0, 0);
}

// F[pos[e]] += val[e] for the entries of A assembled into the front (distinct positions)
__global__ void frontAKer(cuComplex* F, const int* pos, const cuComplex* val, int64_t cnt)
{
    for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < cnt; e += (int64_t)gridDim.x * blockDim.x) {
        const cuComplex v = val[e];
        F[pos[e]].x += v.x;
        F[pos[e]].y += v.y;
    }
}

// Extend-add a child's update matrix U (mc x mc, lower, leading dimension ldu) into the front F (m x m),
// row i of U going to row rel[i] of F; one block per column of U
__global__ void extendAddKer(cuComplex* F, int m, const cuComplex* U, int64_t ldu, int mc, const int* rel)
{
    for (int b = blockIdx.x; b < mc; b += gridDim.x) {
        const int64_t cb = (int64_t)m * rel[b];
        for (int i = b + threadIdx.x; i < mc; i += blockDim.x) {
            const cuComplex u = U[i + ldu * b];
            cuComplex* dst = F + rel[i] + cb;
            dst->x += u.x;
            dst->y += u.y;
        }
    }
}

// The L panel (first k columns of F, upper part of the diagonal block set to zero) to contiguous storage
__global__ void panelCopyKer(cuComplex* P, const cuComplex* F, int m, int k)
{
    const int64_t total = (int64_t)m * k;
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        const int i = (int)(idx % m), j = (int)(idx / m);
        P[idx] = i >= j ? F[idx] : make_cuComplex(0, 0);
    }
}

inline void frontCk(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s in %s\n", cudaGetErrorString(e), what);
        std::exit(1);
    }
}

inline void frontCublas(cublasStatus_t s, const char* what)
{
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error %d in %s\n", (int)s, what);
        std::exit(1);
    }
}

// The streams and buffers of one large front's LDL^T with lookahead: s0 runs the bulk of each trailing
// update, s1 (high priority, the critical path) the panels; ev: 2 + 2 events (panel done, rest done, by
// parity) and one for "front assembled on s0"; W[2]: panel workspaces of at least (m - nb) x nb values.
struct FrontLdltStreams
{
    cublasHandle_t h0, h1;
    cudaStream_t s0, s1;
    cudaEvent_t ev[5];
    cuComplex* W[2];
};

// Partial LDL^T of the first k columns of the front F (m x m, column-major, leading dimension ld, lower
// triangle), queued on s0 (after what s0 already holds) and s1: blocked right-looking in column blocks of nb
// (<= 256), with lookahead. s1 updates block column j+1 by block j and factorizes its panel (diagonal block,
// trsm, scale into W[(j+1) % 2]) while s0 updates the rest of the trailing matrix by block j, in CGEMMs on
// column panels of width panelW (they also overwrite the unused upper part of the diagonal tiles). The
// diagonal block runs on one CTA, so without lookahead the GPU sat idle behind it: on the largest test
// systems it had no other kernel running for over 100 ms per factorization. On return s0 has waited for everything on s1.
inline void largeFrontLdlt(const FrontLdltStreams& q, cuComplex* F, int64_t ld, int m, int k, int nb, int panelW,
                           unsigned* zeroPivots)
{
    const cuComplex one = make_cuComplex(1, 0), minusOne = make_cuComplex(-1, 0);
    const int steps = (k + nb - 1) / nb;
    auto gemmNT = [&](cublasHandle_t h, int mr, int nc, int kk, const cuComplex* A, int64_t lda, const cuComplex* B,
                      int64_t ldb, cuComplex* C, int64_t ldc) {
        frontCublas(cublasCgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, mr, nc, kk, &minusOne, A, (int)lda, B, (int)ldb, &one, C, (int)ldc),
                    "cublasCgemm");
    };
    auto panel = [&](int j) {  // on s1: diagonal block, trsm, scale into W[j % 2]
        const int jb = j * nb, jn = std::min(nb, k - jb), r0 = jb + jn, nr = m - r0;
        cuComplex* D = F + jb + ld * jb;
        cuComplex* F21 = F + r0 + ld * jb;
        if (jn <= kDiagShMax) diagShKer<<<1, 256, (kShElems + jn * jn) * sizeof(cuComplex), q.s1>>>(D, ld, jn, zeroPivots);
        else diagKer<<<1, 256, 0, q.s1>>>(D, ld, jn, zeroPivots);
        if (nr == 0) return;
        frontCublas(cublasCtrsm(q.h1, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT, nr, jn, &one, D,
                                (int)ld, F21, (int)ld), "cublasCtrsm");
        scaleKer<<<(nr + 255) / 256, 256, 0, q.s1>>>(F21, ld, D, nr, jn, q.W[j % 2]);
    };
    frontCk(cudaEventRecord(q.ev[4], q.s0), "event");  // the front is assembled
    frontCk(cudaStreamWaitEvent(q.s1, q.ev[4], 0), "wait");
    panel(0);
    frontCk(cudaEventRecord(q.ev[0], q.s1), "event");
    for (int j = 0; j < steps; ++j) {
        const int jb = j * nb, jn = std::min(nb, k - jb), r0 = jb + jn, nr = m - r0;
        const int jnNext = j + 1 < steps ? std::min(nb, k - r0) : 0;
        const cuComplex* Wj = q.W[j % 2];
        const cuComplex* Lj = F + r0 + ld * jb;
        cuComplex* C = F + r0 + ld * r0;
        if (jnNext > 0) {  // s1: block column j+1 by block j (after the rest of step j-1), then its panel
            if (j > 0) frontCk(cudaStreamWaitEvent(q.s1, q.ev[2 + (j - 1) % 2], 0), "wait");
            gemmNT(q.h1, nr, jnNext, jn, Wj, nr, Lj, ld, C, ld);
            panel(j + 1);
            frontCk(cudaEventRecord(q.ev[(j + 1) % 2], q.s1), "event");
        }
        frontCk(cudaStreamWaitEvent(q.s0, q.ev[j % 2], 0), "wait");
        const int n = nr - jnNext;  // s0: the rest of the trailing lower triangle
        for (int c0 = 0; c0 < n; c0 += panelW) {
            const int w = std::min(panelW, n - c0);
            gemmNT(q.h0, n - c0, w, jn, Wj + jnNext + c0, nr, Lj + jnNext + c0, ld, C + (jnNext + c0) + ld * (jnNext + c0), ld);
        }
        frontCk(cudaEventRecord(q.ev[2 + j % 2], q.s0), "event");
    }
    frontCk(cudaStreamWaitEvent(q.s0, q.ev[(steps - 1) % 2], 0), "wait");
}
