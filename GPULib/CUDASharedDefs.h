#pragma once
#include <cuda_runtime.h>

namespace kernelWrapper
{
	typedef struct KernelParams
	{
		size_t bX;
		size_t bY;
		size_t gX;
		size_t gY;
		bool   bUseTimer;
		float  fTimeCount;
		cudaEvent_t cudaEvtStart;
		cudaEvent_t cudaEvtStop;
	}KERNEL_PARAMS;
}