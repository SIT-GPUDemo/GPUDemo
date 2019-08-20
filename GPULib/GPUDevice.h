#pragma once

#ifdef _DLL
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif

#include "GPUTypes.h"

class IGPU
{
public:
	IGPU(){}
	virtual ~IGPU() {}

	virtual int InitDevice( uint8_t devIndex, bool bUseTimer = false ) = 0;
	virtual void ResetTimer(void) = 0;
	virtual float GetExecutionTime(void) = 0;
	virtual int AllocateMemory(size_t width, size_t height, DATA_TYPE dType, MEMORY_TYPE type) = 0;
	virtual void FreeMemory(int device) = 0;
	virtual int WriteData(int8_t* src, int dst, WRITE_PATH write) = 0;
	virtual int WriteData(uint8_t* src, int dst, WRITE_PATH write) = 0;
	virtual int WriteData(float_t* src, int dst, WRITE_PATH write) = 0;
	virtual int Execute(KERNEL_FUNCTION_ID kernel, 
		                int buffer, 
		                int kernelBuffer, 
		                int outBuffer, 
		                float_t scale,
		                size_t blockW, 
		                size_t blockH,
		                size_t imWidth, 
		                size_t imHeight, 
		                bool bUsedSharedMem = false) = 0;
};


extern "C"
{
	DLLEXPORT IGPU* __cdecl CreateGPU();
}