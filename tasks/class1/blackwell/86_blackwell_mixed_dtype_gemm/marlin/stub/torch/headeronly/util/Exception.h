#pragma once
#include <cstdio>
#include <cstdlib>
#define STD_TORCH_CHECK(cond, ...) do { if (!(cond)) { std::fprintf(stderr, "check failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); std::abort(); } } while (0)
