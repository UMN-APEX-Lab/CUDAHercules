// The FP64 refinement around a working-precision preconditioner, used by the solver driver
// (ref_solve with its FP32 factors): right-preconditioned FGMRES with
// Krylov recycling, optionally inside the double-double outer refinement of dd_refine.h.
//
//   precond(v, z)  z = M^-1 v for one n x nports block (complex double, column-major), where M is the
//                  driver's factorization; FGMRES needs nothing else from the driver
//
// Options (parseMixedOption, the same flags in every driver):
//   --gmres              right-preconditioned FGMRES(--gmres-restart, default 30) on A x = b, until the
//                        relative residual reaches --mp-tol (default 1e-12) or
//                        --mp-iters preconditioner applications (default 30); --port-tol T stops once the
//                        solution at B's nonzero rows changes by less than T on two consecutive iterations
//   --dd-ir N            double-double outer refinement (up to N corrections, until the port values stop
//                        changing); each correction is an FGMRES solve to --inner-tol (default 1e-6), or
//                        until --inner-stall K iterations fail to halve its residual estimate
//   --recycle            Krylov recycling: after every restart cycle, harmonic Ritz vectors whose values are
//                        far from 1 (|theta - 1| > 0.5, |theta| > 1e-12: directions the preconditioner gets
//                        badly wrong, whether it amplifies them, |theta| << 1, or turns them around) join a
//                        recycle space U, C = A U (double-double, orthonormal), which later cycles deflate
//                        (GCRO); emptied at every new frequency
//   experiments          --recycle-keep (keep U across frequencies, C rebuilt), --warm-start (previous x as
//                        the initial guess when its residual is below 1), --fgmres-dd-res / --fgmres-dd-spmv
//                        (restart residuals / A z in double-double), --dump-hess P (unrotated Hessenberg of
//                        the first cycle of the first four calls), --dump-x P (initial and refined x)
#ifndef TESTBENCH_MIXED_SOLVE_H
#define TESTBENCH_MIXED_SOLVE_H

#include "bench_common.h"
#include "bench_kernels.h"
#include "dd_refine.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#define MKL_Complex16 std::complex<double>
#include <mkl_lapacke.h>

#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <numeric>

struct MixedOptions
{
    bool gmres = false, recycle = false, recycleKeep = false, warmStart = false, fgmresDD = false, fgmresDDres = false;
    int restart = 30, mpIters = 30, ddIr = 0, innerStall = 0;
    double mpTol = 1e-12, portTol = 0, innerTol = 1e-6;
    std::string dumpHess, dumpX;
};

// One of the options above at argv[a] (advancing a past its value): true if it was one
inline bool parseMixedOption(int& a, int argc, char** argv, MixedOptions& o)
{
    const char* s = argv[a];
    const bool v = a + 1 < argc;
    if (!std::strcmp(s, "--gmres")) o.gmres = true;
    else if (!std::strcmp(s, "--recycle")) o.recycle = true;
    else if (!std::strcmp(s, "--recycle-keep")) o.recycleKeep = true;
    else if (!std::strcmp(s, "--warm-start")) o.warmStart = true;
    else if (!std::strcmp(s, "--fgmres-dd-spmv")) o.fgmresDD = true;
    else if (!std::strcmp(s, "--fgmres-dd-res")) o.fgmresDDres = true;
    else if (!std::strcmp(s, "--gmres-restart") && v) o.restart = std::atoi(argv[++a]);
    else if (!std::strcmp(s, "--mp-iters") && v) o.mpIters = std::atoi(argv[++a]);
    else if (!std::strcmp(s, "--mp-tol") && v) o.mpTol = std::atof(argv[++a]);
    else if (!std::strcmp(s, "--port-tol") && v) o.portTol = std::atof(argv[++a]);
    else if (!std::strcmp(s, "--dd-ir") && v) o.ddIr = std::atoi(argv[++a]);
    else if (!std::strcmp(s, "--inner-tol") && v) o.innerTol = std::atof(argv[++a]);
    else if (!std::strcmp(s, "--inner-stall") && v) o.innerStall = std::atoi(argv[++a]);
    else if (!std::strcmp(s, "--dump-hess") && v) o.dumpHess = argv[++a];
    else if (!std::strcmp(s, "--dump-x") && v) o.dumpX = argv[++a];
    else return false;
    return true;
}

class MixedSolver
{
public:
    using Precond = std::function<void(const Complex* v, Complex* z)>;

    struct Result
    {
        int solves = 0;               // preconditioner applications
        double delivered = -1;        // residual of the delivered x, when the refinement computes it
        double ddWait = 0;            // wait for the background dd-IR setup (pattern only: analysis time)
        std::vector<double> hist;     // residuals at the restarts (of the initial FGMRES with --dd-ir)
    };

    // A: upper CSR on the host (its pattern; the dd-IR setup starts in the background and is joined by the
    // first solve, so A's pattern and b must stay put until then); dRow/dCol/dVal: the same on the device
    // (dVal holds the current frequency's values); dB, dX: n x nports right-hand sides and solution.
    MixedSolver(const Csr& A, const std::vector<Complex>& b, int nports, const int32_t* dRow, const int32_t* dCol,
                const Complex* dVal, Complex* dB, Complex* dX, const MixedOptions& o)
        : A_(A), n_(A.n), nports_(nports), len_(A.n * nports), dRow_(dRow), dCol_(dCol), dVal_(dVal), dB_(dB), dX_(dX), o_(o)
    {
        ck(cudaMalloc(&dR_, len_ * sizeof(Complex)), "cudaMalloc r");
        if (cublasCreate(&blas_) != CUBLAS_STATUS_SUCCESS) die("cublasCreate failed");
        if (o.gmres && o.ddIr > 0)  // lower-part index built on the host, then uploaded: overlaps analysis and factorization
            ddSetup_ = std::async(std::launch::async, [&A, &b, nports, dRow, dCol]() {
                return std::make_unique<DDRefiner>(A, b, nports, dRow, dCol);
            });
    }

    ~MixedSolver()
    {
        if (ddSetup_.valid()) ddSetup_.wait();
        for (void* p : owned_) cudaFree(p);
        for (void* p : {(void*)dR_, (void*)dScr_, (void*)dPart_, (void*)dTmp_, (void*)dPortRows_, (void*)dPortVals_,
                        (void*)dVp_, (void*)dZp_, (void*)dUp_, (void*)dCp_})
            cudaFree(p);
        if (blas_) cublasDestroy(blas_);
    }

    // Device memory that the driver uses for something else between solves (ref_solve: the factorization's
    // update matrices). The Krylov storage is allocated as the iterations reach it: FGMRES's basis block by
    // block (V_j and Z_j, n x nports each), the recycle space slot by slot (u_i and c_i). It is taken from
    // this buffer while it lasts, then from cudaMalloc while more than kReserve stays free; when neither has
    // room, the cycle restarts early (or no more directions are recycled), which costs iterations but never
    // accuracy. What was allocated is kept for the next solves, so the run's peak is its largest need. The
    // recycle space survives no factorization (it is emptied at every frequency), except with --recycle-keep,
    // whose slots therefore never come from this buffer.
    void setWorkspace(void* p, size_t bytes)
    {
        extWs_ = p;
        extBytes_ = bytes;
    }

    static constexpr int kMaxRecycle = 16;
    static constexpr size_t kReserve = (size_t)2 << 30;  // free device memory the Krylov storage leaves alone
    static constexpr double kFarFromOne = 0.5;  // harmonic Ritz values recycled: |theta - 1| above this

    // Solve A x = b for the current values (dVal) and right-hand sides (dB): FGMRES, inside the double-double
    // refinement with --dd-ir. fi: frequency index (0 for the first), tag: name for the dumps.
    Result solve(int fi, const std::string& tag, const Precond& precond, const std::vector<int64_t>& portRows)
    {
        if (!o_.gmres) die("MixedSolver::solve needs --gmres");
        Result res;
        precond_ = &precond;
        portRows_ = &portRows;
        mpTol_ = o_.mpTol;
        portTol_ = o_.portTol;
        stallWindow_ = 0;
        const bool warm = o_.warmStart && fi > 0 && o_.ddIr > 0;  // decided once dd has this A
        if (!warm) ck(cudaMemset(dX_, 0, len_ * sizeof(Complex)), "memset x");
        ck(cudaMemcpy(dR_, dB_, len_ * sizeof(Complex), cudaMemcpyDeviceToDevice), "copy b");
        std::vector<double> bn(nports_), rn(nports_);
        columnNorms(cz(dB_), n_, nports_, bn.data());
        const bool ddOuter = o_.ddIr > 0;
        if (ddOuter) {
            if (!dd_) {  // before FGMRES takes the free memory
                auto td = Clock::now();
                if (!ddSetup_.valid()) die("the dd-IR setup was not started");  // the constructor starts it
                dd_ = ddSetup_.get();
                res.ddWait = since(td);  // only the part not hidden behind analysis and factorization
            }
            dd_->setValues(dVal_);
            if (warm) {  // keep the previous frequency's x as the initial guess if it beats x = 0
                dd_->residualFast(dX_, dB_, dR_);
                columnNorms(cz(dR_), n_, nports_, rn.data());
                double worst = 0;
                for (int c = 0; c < nports_; ++c) worst = std::max(worst, rn[c] / bn[c]);
                if (!(worst < 1)) ck(cudaMemset(dX_, 0, len_ * sizeof(Complex)), "memset x");
                std::printf("  warm start               : previous x has residual %.2e, %s\n", worst,
                            worst < 1 ? "used" : "not used (x = 0)");
                ck(cudaMemcpy(dR_, dB_, len_ * sizeof(Complex), cudaMemcpyDeviceToDevice), "copy b");
            }
            if (o_.recycle) {
                if (!dUp_) recycleInit();
                kCap_ = kMaxRecycle;  // retried at every frequency if device memory ran out before
                if (o_.recycleKeep && kRec_ > 0) {  // keep U, rebuild C = A U for this frequency's A
                    const int kOld = kRec_;
                    kRec_ = 0;
                    for (int i = 0; i < kOld; ++i) {
                        if (kRec_ != i) ck(cudaMemcpy(dU_[kRec_], dU_[i], n_ * sizeof(Complex), cudaMemcpyDeviceToDevice), "copy u");
                        dd_->spmvAccurate(dU_[kRec_], dC_[kRec_]);
                        if (orthoSlot(kRec_)) ++kRec_;
                    }
                    std::printf("  recycle space kept       : %d of %d directions\n", kRec_, kOld);
                } else kRec_ = 0;  // otherwise start every frequency empty (outlier directions move)
            }
            mpTol_ = o_.innerTol;
            portTol_ = 0;
            stallWindow_ = o_.innerStall;
        }
        res.solves = fgmres(res.hist, bn);
        if (ddOuter && !o_.dumpX.empty()) dumpVector(o_.dumpX + "_" + tag + "_x0.xbin", dX_);
        if (ddOuter) {  // outer double-double refinement, each correction an FGMRES solve to innerTol
            std::printf("  initial FGMRES           : %d preconditioner solves to %.0e, residual at restarts", res.solves,
                        o_.innerTol);
            for (double r : res.hist) std::printf(" %.1e", r);
            std::printf("\n");
            res.hist.clear();
            const DDResult dr = dd_->refine(dVal_, dB_, dX_, o_.ddIr, [&](const Complex* r) {
                ck(cudaMemcpy(dB_, r, len_ * sizeof(Complex), cudaMemcpyDeviceToDevice), "copy r");
                ck(cudaMemset(dX_, 0, len_ * sizeof(Complex)), "memset x");
                std::vector<double> rnorm(nports_), ih;
                columnNorms(cz(dB_), n_, nports_, rnorm.data());
                const int its = fgmres(ih, rnorm);
                std::printf("    inner FGMRES: %d preconditioner solves, relative residual %.1e%s\n", its, ih.back(),
                            o_.recycle ? (", recycled k = " + std::to_string(kRec_)).c_str() : "");
                return its;
            });
            res.solves += dr.solves;
            for (double r : dr.relResid) res.delivered = std::max(res.delivered, r);
            if (!o_.dumpX.empty()) dumpVector(o_.dumpX + "_" + tag + "_x.xbin", dX_);
        }
        ck(cudaDeviceSynchronize(), "solve");
        reportStorage();
        return res;
    }

private:
    static void ck(cudaError_t e, const char* what)
    {
        if (e != cudaSuccess) {
            std::fprintf(stderr, "CUDA error %s in %s\n", cudaGetErrorString(e), what);
            std::exit(1);
        }
    }
    static cuDoubleComplex* cz(Complex* p) { return reinterpret_cast<cuDoubleComplex*>(p); }
    static const cuDoubleComplex* cz(const Complex* p) { return reinterpret_cast<const cuDoubleComplex*>(p); }

    void dumpVector(const std::string& path, const Complex* dv)  // n x nports complex128
    {
        std::vector<Complex> hv(len_);
        ck(cudaMemcpy(hv.data(), dv, hv.size() * sizeof(Complex), cudaMemcpyDeviceToHost), "copy x");
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f || std::fwrite(hv.data(), sizeof(Complex), hv.size(), f) != hv.size()) die("cannot write x dump");
        std::fclose(f);
    }

    // Krylov recycling: U (solution space) and C = A U (orthonormal columns), single n-vectors shared by all
    // right-hand sides. A U is computed with double-double accumulation so the deflation is exact even for
    // the directions where the FP32 preconditioner is off by orders of magnitude.
    // `bytes` of Krylov storage: from the driver's buffer while it lasts (if `shared`), else from cudaMalloc while
    // kReserve stays free; nullptr when there is no room
    Complex* grab(size_t bytes, bool shared)
    {
        bytes = (bytes + 255) & ~(size_t)255;
        if (shared && extWs_ && extUsed_ + bytes <= extBytes_) {
            Complex* p = reinterpret_cast<Complex*>(static_cast<char*>(extWs_) + extUsed_);
            extUsed_ += bytes;
            return p;
        }
        size_t freeB = 0, totalB = 0;
        ck(cudaMemGetInfo(&freeB, &totalB), "cudaMemGetInfo");
        void* p = nullptr;
        if (freeB < bytes + kReserve || cudaMalloc(&p, bytes) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        owned_.push_back(p);
        ownedBytes_ += bytes;
        return static_cast<Complex*>(p);
    }

    bool ensureV(int j)  // V_0..V_j exist (and their pointers are in dVp_)
    {
        while ((int)vBlk_.size() <= j) {
            Complex* p = grab(len_ * sizeof(Complex), true);
            if (!p) return false;
            ck(cudaMemcpy(dVp_ + vBlk_.size(), &p, sizeof(p), cudaMemcpyHostToDevice), "copy V pointer");
            vBlk_.push_back(p);
        }
        return true;
    }

    bool ensureZ(int j)  // Z_0..Z_j exist (and their pointers are in dZp_)
    {
        while ((int)zBlk_.size() <= j) {
            Complex* p = grab(len_ * sizeof(Complex), true);
            if (!p) return false;
            ck(cudaMemcpy(dZp_ + zBlk_.size(), &p, sizeof(p), cudaMemcpyHostToDevice), "copy Z pointer");
            zBlk_.push_back(p);
        }
        return true;
    }

    bool ensureSlot(int i)  // recycle slot i: u_i and c_i, one allocation
    {
        while ((int)dU_.size() <= i) {
            Complex* p = grab(2 * n_ * sizeof(Complex), !o_.recycleKeep);
            if (!p) return false;
            Complex* c = p + n_;
            ck(cudaMemcpy(dUp_ + dU_.size(), &p, sizeof(p), cudaMemcpyHostToDevice), "copy U pointer");
            ck(cudaMemcpy(dCp_ + dC_.size(), &c, sizeof(c), cudaMemcpyHostToDevice), "copy C pointer");
            dU_.push_back(p);
            dC_.push_back(c);
        }
        return true;
    }

    void recycleInit()
    {
        kCap_ = kMaxRecycle;
        ck(cudaMalloc(&dUp_, kMaxRecycle * sizeof(Complex*)), "cudaMalloc U pointers");
        ck(cudaMalloc(&dCp_, kMaxRecycle * sizeof(Complex*)), "cudaMalloc C pointers");
        ck(cudaMalloc(&dTmp_, n_ * sizeof(Complex)), "cudaMalloc tmp");
    }

    // one line when the Krylov storage has grown since the last report
    void reportStorage()
    {
        const size_t slots = vBlk_.size() + zBlk_.size() + dU_.size();
        if (slots == reported_) return;
        reported_ = slots;
        const double GB = 1073741824.0;
        std::printf("  Krylov storage           : V %zu, Z %zu blocks, recycle %zu slots: %.2f GB (%.2f GB in the "
                    "driver's buffer)%s\n", vBlk_.size(), zBlk_.size(), dU_.size(), (extUsed_ + ownedBytes_) / GB,
                    extUsed_ / GB, shortCycles_ ? ", cycles cut short by device memory" : "");
    }

    // orthogonalize slot `slot` (c = A u already in dC[slot]) against slots < kRec; false if dependent
    bool orthoSlot(int slot)
    {
        Complex *u = dU_[slot], *c = dC_[slot];
        double c0 = 0;
        cublasDznrm2(blas_, n_, cz(c), 1, &c0);
        for (int pass = 0; pass < 2; ++pass)
            for (int i = 0; i < kRec_; ++i) {
                cuDoubleComplex h;
                cublasZdotc(blas_, n_, cz(dC_[i]), 1, cz(c), 1, &h);
                cuDoubleComplex mh = make_cuDoubleComplex(-h.x, -h.y);
                cublasZaxpy(blas_, n_, &mh, cz(dC_[i]), 1, cz(c), 1);
                cublasZaxpy(blas_, n_, &mh, cz(dU_[i]), 1, cz(u), 1);
            }
        double nu = 0;
        cublasDznrm2(blas_, n_, cz(c), 1, &nu);
        if (!(nu > 1e-2 * c0) || !std::isfinite(nu)) return false;  // nearly dependent: not a new direction
        double inv = 1.0 / nu;
        cublasZdscal(blas_, n_, &inv, cz(c), 1);
        cublasZdscal(blas_, n_, &inv, cz(u), 1);
        return true;
    }

    bool recycleAppend(const Complex* u)  // add direction u if it is new; true if added
    {
        if (kRec_ >= kCap_) return false;
        if (!ensureSlot(kRec_)) {
            kCap_ = kRec_;  // no device memory for more slots
            return false;
        }
        ck(cudaMemcpy(dU_[kRec_], u, n_ * sizeof(Complex), cudaMemcpyDeviceToDevice), "copy u");
        dd_->spmvAccurate(dU_[kRec_], dC_[kRec_]);
        if (!orthoSlot(kRec_)) return false;
        ++kRec_;
        return true;
    }

    // Right-preconditioned FGMRES on A x = b for every column; x (dX) holds the initial guess. Returns the
    // number of preconditioner applications.
    int fgmres(std::vector<double>& hist, const std::vector<double>& bn)
    {
        const int call = fgmresCalls_++;
        int cycle = 0;
        const int64_t n = n_;
        const int nports = nports_;
        if (!dScr_) {  // coefficients and pointer arrays; the blocks themselves come as the iterations need them
            m_ = o_.restart;
            if (m_ < 1) die("--gmres-restart must be positive");
            ck(cudaMalloc(&dScr_, (size_t)(kMaxRecycle + 2 * (m_ + 1)) * sizeof(Complex)), "cudaMalloc scratch");
            ck(cudaMalloc(&dPart_, multiDotScratch(std::max(m_ + 1, kMaxRecycle)) * sizeof(Complex)), "cudaMalloc partial");
            ck(cudaMalloc(&dVp_, (m_ + 1) * sizeof(Complex*)), "cudaMalloc V pointers");
            ck(cudaMalloc(&dZp_, m_ * sizeof(Complex*)), "cudaMalloc Z pointers");
        }
        if (!ensureV(0)) die("no device memory for FGMRES's residual block");
        const int m = m_;
        auto vcol = [&](int j, int c) { return vBlk_[j] + (size_t)c * n; };
        auto zcol = [&](int j, int c) { return zBlk_[j] + (size_t)c * n; };
        const cuDoubleComplex* const* Vp = reinterpret_cast<const cuDoubleComplex* const*>(dVp_);
        const cuDoubleComplex* const* Zdev = reinterpret_cast<const cuDoubleComplex* const*>(dZp_);
        const cuDoubleComplex* const* Up = reinterpret_cast<const cuDoubleComplex* const*>(dUp_);
        const cuDoubleComplex* const* Cp = reinterpret_cast<const cuDoubleComplex* const*>(dCp_);

        // the port rows' solution values decide the port stopping test (one gather, one copy)
        const std::vector<int64_t>& portRows = *portRows_;
        const size_t P = portRows.size();
        if (P > portCap_) {
            cudaFree(dPortRows_);
            cudaFree(dPortVals_);
            ck(cudaMalloc(&dPortRows_, P * sizeof(int64_t)), "cudaMalloc port rows");
            ck(cudaMalloc(&dPortVals_, P * nports * sizeof(Complex)), "cudaMalloc port values");
            portCap_ = P;
        }
        if (P) ck(cudaMemcpy(dPortRows_, portRows.data(), P * sizeof(int64_t), cudaMemcpyHostToDevice), "copy port rows");
        auto gatherPorts = [&](const Complex* base) {  // values at the port rows, every column
            std::vector<Complex> v(nports * P);
            if (!P) return v;
            gatherRows(cz(base), n, nports, dPortRows_, (int64_t)P, cz(dPortVals_));
            ck(cudaMemcpy(v.data(), dPortVals_, v.size() * sizeof(Complex), cudaMemcpyDeviceToHost), "copy port values");
            return v;
        };

        int applied = 0;
        bool portConverged = false;
        std::vector<std::vector<Complex>> H(nports), sn(nports), g(nports), Hraw(nports);
        std::vector<std::vector<double>> cs(nports);
        std::vector<double> beta(nports);
        for (;;) {
            Complex* r0 = vBlk_[0];
            if (o_.fgmresDDres && dd_) dd_->residualAccurate(dX_, dB_, r0);
            else if (dd_) dd_->residualFast(dX_, dB_, r0);
            else residualSymUpper(n, dRow_, dCol_, cz(dVal_), cz(dX_), cz(dB_), cz(r0), nports);
            columnNorms(cz(r0), n, nports, beta.data());
            double rel = 0;
            for (int c = 0; c < nports; ++c) rel = std::max(rel, beta[c] / bn[c]);
            hist.push_back(rel);
            if (portConverged || rel <= mpTol_ || applied >= o_.mpIters || !std::isfinite(rel)) break;
            const int kc = o_.recycle ? kRec_ : 0;  // recycled directions deflated in this cycle
            std::vector<std::vector<Complex>> Bm(nports, std::vector<Complex>((size_t)kc * m));  // C^H A z_j
            if (kc > 0) {  // x += U C^H r, r -= C C^H r (C, U are n x kc matrices)
                for (int c = 0; c < nports; ++c) {
                    multiDotc(Cp, kc, 0, cz(vcol(0, c)), n, cz(dScr_), cz(dPart_));
                    multiAxpy(Up, kc, 0, cz(dScr_), 1, cz(dX_) + (size_t)c * n, n);
                    multiAxpy(Cp, kc, 0, cz(dScr_), -1, cz(vcol(0, c)), n);
                }
                columnNorms(cz(r0), n, nports, beta.data());
            }
            const std::vector<Complex> x0p = gatherPorts(dX_);
            std::vector<std::vector<Complex>> Zp;  // port-row values of each Z block
            std::vector<Complex> xpPrev;
            int calm = 0;
            for (int c = 0; c < nports; ++c) {
                double inv = 1.0 / beta[c];
                cublasZdscal(blas_, n, &inv, cz(vcol(0, c)), 1);
                H[c].assign((size_t)(m + 1) * m, Complex(0));
                if (!o_.dumpHess.empty() || o_.recycle) Hraw[c].assign((size_t)(m + 1) * m, Complex(0));
                cs[c].assign(m, 0);
                sn[c].assign(m, Complex(0));
                g[c].assign(m + 1, Complex(0));
                g[c][0] = beta[c];
            }
            int j = 0;
            std::vector<double> est;  // worst relative residual estimate after each iteration of this cycle
            bool stalled = false;
            while (j < m && applied < o_.mpIters) {
                if (!ensureZ(j) || !ensureV(j + 1)) {  // no device memory for another block: restart here
                    if (j == 0) die("no device memory for FGMRES's first iteration");
                    shortCycles_ = true;
                    break;
                }
                (*precond_)(vcol(j, 0), zcol(j, 0));
                ++applied;
                if (portTol_ > 0) Zp.push_back(gatherPorts(zcol(j, 0)));
                double worst = 0;
                if (dd_ && !o_.fgmresDD) dd_->spmvFast(zcol(j, 0), vcol(j + 1, 0), nports);  // one pass over A
                for (int c = 0; c < nports; ++c) {
                    Complex* w = vcol(j + 1, c);
                    if (o_.fgmresDD && dd_) dd_->spmvAccurate(zcol(j, c), w);
                    else if (!dd_) spmvSymUpper(n, dRow_, dCol_, cz(dVal_), cz(zcol(j, c)), cz(w));
                    // w -= C C^H w (keeping C^H w for the solution update), then classical Gram-Schmidt
                    // against v_0..v_j twice (CGS2): each step is one multi-vector dot and one multi-vector
                    // axpy over the basis (column c of the blocks V_0..V_j). Coefficients stay on the device
                    // until one copy at the end: [C^H w (kc) | pass 1 (j+1) | pass 2 (j+1)].
                    const int nb = j + 1;
                    if (kc > 0) {
                        multiDotc(Cp, kc, 0, cz(w), n, cz(dScr_), cz(dPart_));
                        multiAxpy(Cp, kc, 0, cz(dScr_), -1, cz(w), n);
                    }
                    for (int pass = 0; pass < 2; ++pass) {
                        cuDoubleComplex* hp = cz(dScr_) + kc + pass * nb;
                        multiDotc(Vp, nb, (int64_t)c * n, cz(w), n, hp, cz(dPart_));
                        multiAxpy(Vp, nb, (int64_t)c * n, hp, -1, cz(w), n);
                    }
                    std::vector<Complex> coef(kc + 2 * nb);
                    ck(cudaMemcpy(coef.data(), dScr_, coef.size() * sizeof(Complex), cudaMemcpyDeviceToHost), "copy coef");
                    for (int i = 0; i < kc; ++i) Bm[c][i + (size_t)j * kc] = coef[i];
                    auto Hc = [&](int i, int k) -> Complex& { return H[c][i + (size_t)k * (m + 1)]; };
                    for (int i = 0; i < nb; ++i) Hc(i, j) = coef[kc + i] + coef[kc + nb + i];
                    double hn = 0;
                    cublasDznrm2(blas_, n, cz(w), 1, &hn);
                    Hc(j + 1, j) = hn;
                    if (!o_.dumpHess.empty() || o_.recycle)
                        for (int i = 0; i <= j + 1; ++i) Hraw[c][i + (size_t)j * (m + 1)] = Hc(i, j);
                    if (hn > 0) {
                        double inv = 1.0 / hn;
                        cublasZdscal(blas_, n, &inv, cz(w), 1);
                    }
                    for (int i = 0; i < j; ++i) {  // previous Givens rotations
                        Complex t = cs[c][i] * Hc(i, j) + sn[c][i] * Hc(i + 1, j);
                        Hc(i + 1, j) = -std::conj(sn[c][i]) * Hc(i, j) + cs[c][i] * Hc(i + 1, j);
                        Hc(i, j) = t;
                    }
                    Complex a = Hc(j, j), bb = Hc(j + 1, j);
                    double t = std::sqrt(std::norm(a) + std::norm(bb));
                    if (std::abs(a) == 0) {
                        cs[c][j] = 0;
                        sn[c][j] = 1;
                    } else {
                        cs[c][j] = std::abs(a) / t;
                        sn[c][j] = (a / std::abs(a)) * std::conj(bb) / t;
                    }
                    Hc(j, j) = cs[c][j] * a + sn[c][j] * bb;
                    Hc(j + 1, j) = 0;
                    g[c][j + 1] = -std::conj(sn[c][j]) * g[c][j];
                    g[c][j] = cs[c][j] * g[c][j];
                    worst = std::max(worst, std::abs(g[c][j + 1]) / bn[c]);
                }
                ++j;
                if (worst <= mpTol_) break;
                est.push_back(worst);
                if (stallWindow_ > 0 && (int)est.size() > stallWindow_ && worst > 0.5 * est[est.size() - 1 - stallWindow_]) {
                    stalled = true;
                    break;
                }
                if (portTol_ > 0) {  // x at the port rows if the cycle stopped here: x0 + Z y
                    std::vector<Complex> xp(x0p);
                    double delta = 0;
                    for (int c = 0; c < nports; ++c) {
                        auto Hc = [&](int i, int k) { return H[c][i + (size_t)k * (m + 1)]; };
                        std::vector<Complex> y(j);
                        for (int i = j - 1; i >= 0; --i) {
                            Complex s = g[c][i];
                            for (int k = i + 1; k < j; ++k) s -= Hc(i, k) * y[k];
                            y[i] = s / Hc(i, i);
                        }
                        double dn = 0, xn = 0;
                        for (size_t k = 0; k < P; ++k) {
                            for (int i = 0; i < j; ++i) xp[c * P + k] += y[i] * Zp[i][c * P + k];
                            if (!xpPrev.empty()) dn += std::norm(xp[c * P + k] - xpPrev[c * P + k]);
                            xn += std::norm(xp[c * P + k]);
                        }
                        if (!xpPrev.empty()) delta = std::max(delta, std::sqrt(dn / xn));
                    }
                    calm = (!xpPrev.empty() && delta < portTol_) ? calm + 1 : 0;
                    xpPrev = xp;
                    if (calm >= 2) {
                        portConverged = true;
                        break;
                    }
                }
            }
            if (!o_.dumpHess.empty() && call < 4 && cycle == 0) {  // (j+1) x j Hessenberg, first column
                std::string path = o_.dumpHess + "_call" + std::to_string(call) + ".hess";
                FILE* f = std::fopen(path.c_str(), "wb");
                int32_t dims[2] = {j + 1, j};
                std::fwrite(dims, sizeof(int32_t), 2, f);
                for (int k = 0; k < j; ++k) std::fwrite(&Hraw[0][(size_t)k * (m + 1)], sizeof(Complex), j + 1, f);
                std::fclose(f);
            }
            ++cycle;
            for (int c = 0; c < nports; ++c) {  // x += Z y with H y = g (upper triangular)
                auto Hc = [&](int i, int k) { return H[c][i + (size_t)k * (m + 1)]; };
                std::vector<Complex> y(j);
                for (int i = j - 1; i >= 0; --i) {
                    Complex s = g[c][i];
                    for (int k = i + 1; k < j; ++k) s -= Hc(i, k) * y[k];
                    y[i] = s / Hc(i, i);
                }
                std::vector<Complex> coef(y);  // [y | -B y]: x += Z y, then x -= U (B y) (A Z = C B + V H)
                for (int i = 0; i < kc; ++i) {
                    Complex bc = 0;
                    for (int l = 0; l < j; ++l) bc += Bm[c][i + (size_t)l * kc] * y[l];
                    coef.push_back(-bc);
                }
                ck(cudaMemcpy(dScr_, coef.data(), coef.size() * sizeof(Complex), cudaMemcpyHostToDevice), "copy y");
                multiAxpy(Zdev, j, (int64_t)c * n, cz(dScr_), 1, cz(dX_) + (size_t)c * n, n);  // one pass over x
                multiAxpy(Up, kc, 0, cz(dScr_) + j, 1, cz(dX_) + (size_t)c * n, n);
            }
            if (o_.recycle && kRec_ < kCap_ && j >= 2) {  // harvest harmonic Ritz vectors with theta far from 1
                int added = 0;
                std::vector<double> found;
                for (int c = 0; c < nports && kRec_ < kCap_; ++c) {
                    std::vector<Complex> G((size_t)j * j), f(j, Complex(0));
                    for (int k = 0; k < j; ++k)
                        for (int i = 0; i < j; ++i) G[i + (size_t)k * j] = Hraw[c][i + (size_t)k * (m + 1)];
                    const double hl = std::norm(Hraw[c][j + (size_t)(j - 1) * (m + 1)]);
                    std::vector<Complex> GH((size_t)j * j);  // G^H for f = G^-H e_j
                    for (int k = 0; k < j; ++k)
                        for (int i = 0; i < j; ++i) GH[i + (size_t)k * j] = std::conj(G[k + (size_t)i * j]);
                    std::vector<lapack_int> ipiv(j);
                    f[j - 1] = 1;
                    if (LAPACKE_zgesv(LAPACK_COL_MAJOR, j, 1, GH.data(), j, ipiv.data(), f.data(), j) != 0) continue;
                    for (int i = 0; i < j; ++i) G[i + (size_t)(j - 1) * j] += hl * f[i];
                    std::vector<Complex> theta(j), vr((size_t)j * j);
                    if (LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', 'V', j, G.data(), j, theta.data(), nullptr, 1, vr.data(), j) != 0)
                        continue;
                    std::vector<int> order(j);
                    std::iota(order.begin(), order.end(), 0);
                    auto far = [&](int i) { return std::abs(theta[i] - 1.0); };
                    std::sort(order.begin(), order.end(), [&](int x, int y) { return far(x) > far(y); });
                    for (int idx : order) {
                        const double at = std::abs(theta[idx]);
                        if (far(idx) <= kFarFromOne || kRec_ >= kCap_) break;
                        if (at <= 1e-12) continue;
                        ck(cudaMemset(dTmp_, 0, n * sizeof(Complex)), "memset tmp");
                        ck(cudaMemcpy(dScr_, vr.data() + (size_t)idx * j, j * sizeof(Complex), cudaMemcpyHostToDevice),
                           "copy g");
                        multiAxpy(Zdev, j, (int64_t)c * n, cz(dScr_), 1, cz(dTmp_), n);  // u = Z g
                        if (recycleAppend(dTmp_)) {
                            ++added;
                            found.push_back(at);
                        }
                    }
                }
                if (added) {
                    std::printf("    recycle: +%d direction(s) |theta|", added);
                    for (double t : found) std::printf(" %.1e", t);
                    std::printf(" -> k = %d\n", kRec_);
                }
            }
            if (stalled) portConverged = true;  // reuse the stop flag: one more residual, then return
        }
        return applied;
    }

    const Csr& A_;
    int64_t n_;
    int nports_;
    int64_t len_;
    const int32_t *dRow_, *dCol_;
    const Complex* dVal_;
    Complex *dB_, *dX_, *dR_ = nullptr;
    MixedOptions o_;
    cublasHandle_t blas_ = nullptr;
    std::unique_ptr<DDRefiner> dd_;
    std::future<std::unique_ptr<DDRefiner>> ddSetup_;
    const Precond* precond_ = nullptr;
    const std::vector<int64_t>* portRows_ = nullptr;
    double mpTol_ = 0, portTol_ = 0;
    int stallWindow_ = 0;
    int m_ = 0, fgmresCalls_ = 0;
    Complex *dScr_ = nullptr, *dPart_ = nullptr, *dTmp_ = nullptr;
    void* extWs_ = nullptr;      // the driver's buffer (setWorkspace) and how much of it the Krylov storage took
    size_t extBytes_ = 0, extUsed_ = 0, ownedBytes_ = 0, reported_ = 0;
    std::vector<void*> owned_;   // the Krylov storage from cudaMalloc
    bool shortCycles_ = false;   // a cycle restarted early for lack of device memory
    std::vector<Complex*> vBlk_, zBlk_;  // V_j, Z_j (n x nports each)
    std::vector<Complex*> dU_, dC_;      // recycle slots (one n-vector each)
    Complex **dVp_ = nullptr, **dZp_ = nullptr, **dUp_ = nullptr, **dCp_ = nullptr;  // the same pointers on the device
    int kRec_ = 0, kCap_ = 0;
    int64_t* dPortRows_ = nullptr;
    Complex* dPortVals_ = nullptr;
    size_t portCap_ = 0;
};

#endif
