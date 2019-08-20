#pragma once
// ----------------------------------------------------------------------------------------------------------------- //
// KernelWrappers.h                                                                                                  //
// ----------------------------------------------------------------------------------------------------------------- //
// Description: prototypes of wrapper functions for the GPU kernels for calling from the DLL files                   //
// ----------------------------------------------------------------------------------------------------------------- //
#include "CUDASharedDefs.h"
namespace kernelWrapper
{
	extern "C" int filterImage5x5_g(KernelParams* params, float* imageIn, float* kernel, float* imageOut, float scaleFactor, int imW, int imH);
	extern "C" int filterImage5x5_s(KernelParams* params, float* imageIn, float* kernel, float* imageOut, float scaleFactor, int imW, int imH);
}