/***********************************************************************[cuda13_compat.cuh]
CUDA 13 source-compatibility shim for ParaFROST.

CUDA 13.0 changed the signatures of cudaMemAdvise / cudaMemPrefetchAsync to
take a cudaMemLocation struct in place of the old `int device` argument
(and added a `flags` argument to cudaMemPrefetchAsync). This header
preserves the old call shape used throughout ParaFROST by providing inline
wrappers and redirecting via macros, so memory.cu / memory.cuh do not need
to change.

Include this header before the first call site. The redirect is a no-op on
older toolkits.
**********************************************************************************/
#ifndef __CUDA13_COMPAT_CUH_
#define __CUDA13_COMPAT_CUH_

#include <cuda_runtime.h>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000

namespace ParaFROST {
namespace cuda13 {

	inline cudaMemLocation make_mem_location(int device) {
		cudaMemLocation loc;
		if (device == cudaCpuDeviceId) {
			loc.type = cudaMemLocationTypeHost;
			loc.id   = 0;
		} else {
			loc.type = cudaMemLocationTypeDevice;
			loc.id   = device;
		}
		return loc;
	}

	inline cudaError_t mem_advise(const void* devPtr, size_t count,
								  cudaMemoryAdvise advice, int device) {
		return ::cudaMemAdvise(devPtr, count, advice, make_mem_location(device));
	}

	inline cudaError_t mem_prefetch_async(const void* devPtr, size_t count,
										  int device, cudaStream_t stream = 0) {
		return ::cudaMemPrefetchAsync(devPtr, count, make_mem_location(device),
									  /*flags=*/0u, stream);
	}

} // namespace cuda13
} // namespace ParaFROST

// Redirect old-style call sites to the wrappers. Defined AFTER the inline
// functions so the wrapper bodies still resolve to the real CUDA symbols.
#define cudaMemAdvise(...)        ::ParaFROST::cuda13::mem_advise(__VA_ARGS__)
#define cudaMemPrefetchAsync(...) ::ParaFROST::cuda13::mem_prefetch_async(__VA_ARGS__)

#endif // CUDART_VERSION >= 13000

#endif // __CUDA13_COMPAT_CUH_
