// Double-double outer refinement on the GPU (three-precision iterative refinement), used by
// ref_solve.
//
// The residual r = b - A x is accumulated in double-double over whole rows of A, and x is kept as
// double-double (hi + lo). A's values stay in the driver's upper CSR on the device: the refiner adds only
// an index of the strictly lower part (SymCsr in bench_kernels.h), so they are stored once. Each step hands r (rounded to double) to the
// caller's correct(r), which must solve A d = r in working precision into dX; d is then added to
// x in double-double. Refinement stops when x at B's nonzero rows (the ports) changes by less
// than portTol, or after maxSteps corrections.
#ifndef DD_REFINE_H
#define DD_REFINE_H

#include "bench_common.h"
#include "bench_kernels.h"

#include <cuda_runtime.h>

#include <climits>
#include <cstring>

inline void ddCheck(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s in %s\n", cudaGetErrorString(e), what);
        std::exit(1);
    }
}

struct DDResult
{
    int steps = 0;               // corrections applied
    int solves = 0;              // working-precision solves reported by correct()
    bool converged = false;      // port change fell below portTol
    double seconds = 0;
    std::vector<double> relResid;  // ||b - A x||_2 / ||b||_2 of the delivered (double) x, per column
};

class DDRefiner
{
    static cuDoubleComplex* cz(Complex* p) { return reinterpret_cast<cuDoubleComplex*>(p); }
    static const cuDoubleComplex* cz(const Complex* p) { return reinterpret_cast<const cuDoubleComplex*>(p); }

public:
    // Builds the index of A's strictly lower part on the device. dRowU, dColU: the driver's upper CSR of A
    // on the device (A's pattern), whose values come with setValues()/refine(). Port rows are B's nonzero rows.
    DDRefiner(const Csr& A, const std::vector<Complex>& b, int nrhs, const int32_t* dRowU, const int32_t* dColU)
        : n_(A.n), nrhs_(nrhs)
    {
        // Row i's lower entries are the upper entries (j, i), j < i, in ascending j. Built in parallel; the
        // layout does not depend on the thread count.
        std::vector<int64_t> rowL(n_ + 1, 0);
#pragma omp parallel for
        for (int64_t i = 0; i < n_; ++i)
            for (int32_t q = A.rowPtr[i]; q < A.rowPtr[i + 1]; ++q)
                if (A.col[q] != i) __atomic_fetch_add(&rowL[A.col[q] + 1], 1, __ATOMIC_RELAXED);
        for (int64_t i = 0; i < n_; ++i) rowL[i + 1] += rowL[i];
        nnzL_ = rowL[n_];
        std::vector<int32_t> colL(nnzL_), srcL(nnzL_);
        std::vector<int64_t> next(rowL.begin(), rowL.end() - 1);
#pragma omp parallel for
        for (int64_t i = 0; i < n_; ++i)
            for (int32_t q = A.rowPtr[i]; q < A.rowPtr[i + 1]; ++q) {
                const int32_t j = A.col[q];
                if (j == i) continue;
                const int64_t l = __atomic_fetch_add(&next[j], 1, __ATOMIC_RELAXED);
                colL[l] = (int32_t)i;
                srcL[l] = q;
            }
#pragma omp parallel
        {
            std::vector<std::pair<int32_t, int32_t>> seg;
#pragma omp for schedule(dynamic, 4096)
            for (int64_t i = 0; i < n_; ++i) {  // each row's lower part by column
                seg.clear();
                for (int64_t l = rowL[i]; l < rowL[i + 1]; ++l) seg.emplace_back(colL[l], srcL[l]);
                std::sort(seg.begin(), seg.end());
                for (int64_t l = rowL[i]; l < rowL[i + 1]; ++l) {
                    colL[l] = seg[l - rowL[i]].first;
                    srcL[l] = seg[l - rowL[i]].second;
                }
            }
        }
        const int64_t len = n_ * nrhs_;
        ddCheck(cudaMalloc(&rowL_, (n_ + 1) * sizeof(int64_t)), "malloc");
        ddCheck(cudaMalloc(&colL_, std::max<int64_t>(nnzL_, 1) * sizeof(int32_t)), "malloc");
        ddCheck(cudaMalloc(&srcL_, std::max<int64_t>(nnzL_, 1) * sizeof(int32_t)), "malloc");
        for (Complex** p : {&xHi_, &xLo_, &rDD_, &b_}) ddCheck(cudaMalloc(p, len * sizeof(Complex)), "malloc");
        ddCheck(cudaMemcpy(rowL_, rowL.data(), (n_ + 1) * sizeof(int64_t), cudaMemcpyHostToDevice), "copy");
        ddCheck(cudaMemcpy(colL_, colL.data(), nnzL_ * sizeof(int32_t), cudaMemcpyHostToDevice), "copy");
        ddCheck(cudaMemcpy(srcL_, srcL.data(), nnzL_ * sizeof(int32_t), cudaMemcpyHostToDevice), "copy");
        A_.n = n_;
        A_.rowU = dRowU;
        A_.colU = dColU;
        A_.rowL = rowL_;
        A_.colL = colL_;
        A_.srcL = srcL_;
        for (int64_t i = 0; i < n_; ++i)
            for (int c = 0; c < nrhs_; ++c)
                if (b[(size_t)c * n_ + i] != Complex(0)) { ports_.push_back(i); break; }
        const size_t np = std::max<size_t>(ports_.size(), 1);
        ddCheck(cudaMalloc(&dPorts_, np * sizeof(int64_t)), "malloc");
        ddCheck(cudaMalloc(&portVals_, 2 * np * nrhs_ * sizeof(Complex)), "malloc");
        if (!ports_.empty())
            ddCheck(cudaMemcpy(dPorts_, ports_.data(), ports_.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "copy");
        std::printf("  dd-IR setup              : lower-part index %lld entries, values shared with the upper CSR "
                    "(%.2f GB on device with 4 vectors)\n", (long long)nnzL_,
                    ((n_ + 1) * 8.0 + nnzL_ * 8.0 + 4.0 * len * 16) / (1024.0 * 1024 * 1024));
    }

    ~DDRefiner()
    {
        for (void* p : {(void*)rowL_, (void*)colL_, (void*)srcL_, (void*)xHi_, (void*)xLo_, (void*)rDD_, (void*)b_,
                        (void*)dPorts_, (void*)portVals_})
            cudaFree(p);
    }

    // The current frequency's values: the driver's upper-CSR values on the device, which must stay in place
    // while they are used (refine() also sets them)
    void setValues(const Complex* dVal) { A_.valU = cz(dVal); }

    // y = A x for one column with double-double accumulation; needs setValues() for this frequency
    void spmvAccurate(const Complex* x, Complex* y) const { spmvDD(A_, cz(x), cz(y)); }

    // r = b - A x for nrhs columns with double-double accumulation; needs setValues() for this frequency
    void residualAccurate(const Complex* x, const Complex* b, Complex* r) const
    {
        residualDD(A_, cz(x), nullptr, cz(b), cz(r), nrhs_);
    }

    // Plain FP64 y = A x (ncol columns at stride n) and r = b - A x (nrhs columns) over whole rows: faster
    // than the upper-storage kernels, which need atomics for the mirrored half; needs setValues(). With two
    // columns they interleave x in rDD_, which holds nothing then: these run outside refine() or inside
    // correct(), whose r (rDD_) the caller has copied before solving.
    void spmvFast(const Complex* x, Complex* y, int ncol = 1) const
    {
        spmvSym(A_, cz(x), cz(y), ncol, ncol == 2 ? cz(rDD_) : nullptr);
    }
    void residualFast(const Complex* x, const Complex* b, Complex* r) const
    {
        residualSym(A_, cz(x), cz(b), cz(r), nrhs_, nrhs_ == 2 ? cz(rDD_) : nullptr);
    }

    DDRefiner(const DDRefiner&) = delete;
    DDRefiner& operator=(const DDRefiner&) = delete;

    // dVal: upper-CSR values of the current frequency; dB: b (restored on return); dX: initial x,
    // overwritten with the refined x rounded to double. correct(const Complex* dR) solves A d = dR
    // into dX and returns the number of working-precision solves it used.
    template <class Correct>
    DDResult refine(const Complex* dVal, Complex* dB, Complex* dX, int maxSteps, Correct correct,
                    double portTol = 1e-14)
    {
        ddCheck(cudaDeviceSynchronize(), "sync");
        auto t = Clock::now();
        const int64_t len = n_ * nrhs_;
        setValues(dVal);
        ddCheck(cudaMemcpy(b_, dB, len * sizeof(Complex), cudaMemcpyDeviceToDevice), "copy");
        ddCheck(cudaMemcpy(xHi_, dX, len * sizeof(Complex), cudaMemcpyDeviceToDevice), "copy");
        ddCheck(cudaMemset(xLo_, 0, len * sizeof(Complex)), "memset");
        std::vector<double> bn(nrhs_), rn(nrhs_), rh(nrhs_), ih(nrhs_);
        columnNorms(cz(b_), n_, nrhs_, bn.data());

        DDResult res;
        std::vector<Complex> prev = ports();
        for (;; ++res.steps) {
            residualDD(A_, cz(xHi_), cz(xLo_), cz(b_), cz(rDD_), nrhs_);
            columnNorms(cz(rDD_), n_, nrhs_, rn.data());
            std::printf("  dd-IR step %d             : resid(x_dd)", res.steps);
            for (int c = 0; c < nrhs_; ++c) std::printf(" %.3e", rn[c] / bn[c]);
            double change = -1;
            if (res.steps > 0) {
                std::vector<Complex> p = ports();
                double dn = 0, pn = 0;
                for (size_t m = 0; m < p.size(); ++m) { dn += std::norm(p[m] - prev[m]); pn += std::norm(p[m]); }
                change = std::sqrt(dn / pn);
                prev = p;
                std::printf(" | port change %.2e", change);
            }
            std::printf("\n");
            if (change >= 0 && change < portTol) { res.converged = true; break; }
            if (res.steps == maxSteps) break;
            res.solves += correct(static_cast<const Complex*>(rDD_));
            accumulateDD(cz(xHi_), cz(xLo_), cz(dX), len);
        }
        // the delivered x is x_dd rounded to double: its residual, once (into rDD, whose largest entries
        // are kept for the report below)
        std::vector<double> id(nrhs_);
        columnMaxAbs(cz(rDD_), n_, nrhs_, id.data());
        residualDD(A_, cz(xHi_), nullptr, cz(b_), cz(rDD_), nrhs_);
        columnNorms(cz(rDD_), n_, nrhs_, rh.data());
        columnMaxAbs(cz(rDD_), n_, nrhs_, ih.data());
        std::printf("  dd-IR delivered x        : resid(double x)");
        for (int c = 0; c < nrhs_; ++c) std::printf(" %.3e", rh[c] / bn[c]);
        std::printf("\n");
        res.relResid.resize(nrhs_);
        for (int c = 0; c < nrhs_; ++c) res.relResid[c] = rh[c] / bn[c];
        {  // absolute residuals of the last step: delivered double x, and x_dd
            std::printf("  absolute residual        : ||b||_2");
            for (int c = 0; c < nrhs_; ++c) std::printf(" %.4g", bn[c]);
            std::printf(" | double x: ||r||_2");
            for (int c = 0; c < nrhs_; ++c) std::printf(" %.3e", rh[c]);
            std::printf(" ||r||_inf");
            for (int c = 0; c < nrhs_; ++c) std::printf(" %.3e", ih[c]);
            std::printf(" | x_dd: ||r||_2");
            for (int c = 0; c < nrhs_; ++c) std::printf(" %.3e", rn[c]);
            std::printf(" ||r||_inf");
            for (int c = 0; c < nrhs_; ++c) std::printf(" %.3e", id[c]);
            std::printf("\n");
        }
        ddCheck(cudaMemcpy(dB, b_, len * sizeof(Complex), cudaMemcpyDeviceToDevice), "copy");
        ddCheck(cudaMemcpy(dX, xHi_, len * sizeof(Complex), cudaMemcpyDeviceToDevice), "copy");
        ddCheck(cudaDeviceSynchronize(), "sync");
        res.seconds = since(t);
        std::printf("  dd-IR (GPU)              : %8.2f s for %d steps, %d working-precision solves%s\n", res.seconds,
                    res.steps, res.solves, res.converged ? "" : " (not converged)");
        return res;
    }

private:
    std::vector<Complex> ports() const  // x at the port rows, hi + lo (one gather, one copy)
    {
        const int64_t P = (int64_t)ports_.size(), m = P * nrhs_;
        gatherRows(cz(xHi_), n_, nrhs_, dPorts_, P, cz(portVals_));
        gatherRows(cz(xLo_), n_, nrhs_, dPorts_, P, cz(portVals_) + m);
        std::vector<Complex> hl(2 * m), v(m);
        ddCheck(cudaMemcpy(hl.data(), portVals_, 2 * m * sizeof(Complex), cudaMemcpyDeviceToHost), "copy");
        for (int64_t k = 0; k < m; ++k) v[k] = hl[k] + hl[m + k];
        return v;
    }

    int64_t n_, nnzL_ = 0;
    int nrhs_;
    SymCsr A_;                       // the driver's upper CSR plus the lower index below
    int64_t* rowL_ = nullptr;
    int32_t *colL_ = nullptr, *srcL_ = nullptr;
    Complex *xHi_ = nullptr, *xLo_ = nullptr, *rDD_ = nullptr, *b_ = nullptr;
    std::vector<int64_t> ports_;
    int64_t* dPorts_ = nullptr;      // ports_ on the device
    Complex* portVals_ = nullptr;    // 2 * nrhs * ports_.size() gathered values
};

#endif
