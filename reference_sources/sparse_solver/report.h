// Run report of the testbench, shared by the solver driver. Each run ends with one
// "[testbench]" block with fixed phase names, which a benchmark harness can parse:
//
//   [testbench] solver <name>: <description of the path>
//   [testbench] matrix n <n> nnz <nnz of the upper triangle> frequencies <F>
//   [testbench] once init <s> read <s> convert <s> upload <s> analysis <s> verify <s> wall <s>
//   [testbench] freq <Hz> factor <s> solve <s> [<check> <value> ...]
//
// Phases, in seconds ("-" when the driver does not have the phase):
//   init      CUDA context creation and device query
//   read      file reading, all frequencies
//   convert   COO -> CSR on the host, all frequencies
//   upload    host -> device copies of values and right-hand sides, all frequencies
//   analysis  once per dataset: everything that depends only on the pattern (symbolic analysis, the
//             structures derived from it); the ordering runs in ref_order
//   verify    checks on the host (CPU residual, factorization check)
//   wall      from the start of main to the report
//   factor    numerical factorization of one frequency
//   solve     solve and refinement of one frequency
// Checks of one frequency, named by what they measure:
//   residual      max over right-hand sides of ||b - A x||_2 / ||b||_2 of the delivered x
//   solves        working-precision (FP32) solves
//   factor-check  ||A x - L D L^T x|| / ||A x|| for a random x (when there is no solve)
#ifndef TESTBENCH_REPORT_H
#define TESTBENCH_REPORT_H

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

struct RunReport
{
    struct Freq
    {
        double hz = 0, factor = -1, solve = -1;
        std::vector<std::pair<std::string, double>> checks;
        void check(const std::string& name, double v) { checks.emplace_back(name, v); }
    };
    std::string solver, description;
    int64_t n = 0, nnz = 0;
    double init = -1, read = 0, convert = 0, upload = 0, analysis = 0, verify = 0;
    std::vector<Freq> freqs;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    RunReport(std::string name, std::string desc) : solver(std::move(name)), description(std::move(desc)) {}

    Freq& add(double hz)
    {
        freqs.emplace_back();
        freqs.back().hz = hz;
        return freqs.back();
    }

    void print() const
    {
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        auto sec = [](const char* name, double v) {
            if (v < 0) std::printf(" %s -", name);
            else std::printf(" %s %.4f", name, v);
        };
        std::printf("\n[testbench] solver %s: %s\n", solver.c_str(), description.c_str());
        std::printf("[testbench] matrix n %lld nnz %lld frequencies %zu\n", (long long)n, (long long)nnz, freqs.size());
        std::printf("[testbench] once");
        sec("init", init);
        sec("read", read);
        sec("convert", convert);
        sec("upload", upload);
        sec("analysis", analysis);
        sec("verify", verify);
        sec("wall", wall);
        std::printf("\n");
        for (const Freq& f : freqs) {
            std::printf("[testbench] freq %.6g", f.hz);
            sec("factor", f.factor);
            sec("solve", f.solve);
            for (const auto& c : f.checks) std::printf(" %s %.4g", c.first.c_str(), c.second);
            std::printf("\n");
        }
    }
};

#endif
