// Minimal stand-in for the PyTorch header that defines at::PhiloxCudaState (only referenced by
// the dropout path of FlashAttention-2, which the reference never enables).
#pragma once
#include <cstdint>

namespace at {
struct PhiloxCudaState {
  PhiloxCudaState() = default;
  union Payload {
    uint64_t val;
    int64_t *ptr;
  };
  Payload seed_{};
  Payload offset_{};
  uint32_t offset_intragraph_ = 0;
  bool captured_ = false;
};
}  // namespace at
