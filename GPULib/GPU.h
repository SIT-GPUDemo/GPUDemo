#pragma once
#include<cuda_runtime.h>
#include "GPUDevice.h"
#include "GPUTypes.h"
#include<vector>
#include<map>

///\Description:  This file defines a class that provides an interface for a GPU device.
///\              The calls essentially map the CUDA and OpenCL calls.  The reason for providing a wrapper for the
///\              CUDA function calls is to allow CUDA to easily be replaced by OpenCL.  The client programme (FilterTests.cpp)
///\              demonstrates the sequence of calls needed to run a programme on a GPU.  The implementation of this wrapper shows 
///\              the underlying calls to the CUDA API which can be replaced by the OpenCL API if desired.



///\Definition:  Struct BUFFER_t
///\Description:  structure to wrap the pointers to the actual GPU buffers/resources
///\Members:   buffer width
///\           buffer height
///\           buffer pitch
///\           buffer handle
///\           buffer type: how the buffer was allocated - device, page locked buffer, managed buffer/int,uint,float data type
///\           pointer to buffer (uint8_t*, int8_t*, float* pointers are included, the appropriate 
///\           pointe is selected depending on buffer type)
typedef struct BUFFER_t
{
	int8_t* ptr8s = nullptr;
	uint8_t* ptr8u = nullptr;
	float* ptr32f = nullptr;

	size_t width;
	size_t height;
	size_t pitch;

	DATA_TYPE type;

	size_t bufferIndex;
}BUFFER;


///\Definition:  GPU device interface
///\Description: Basic GPU operations: create buffers, read/write buffers, and execute functions
///\Members:
///\   1.  InitDevice(optional): IF multiple GPUs exist, you can select which GPU to use
///\   2.  AllocateMemory:  Creates memory on GPU device
///\                          CUDA:  cudaMemAlloc...()
///\                          OpenCL: clMemAlloc...()
///\       parameters allow specification of where memory is to be allocated: MEMORY_MANAGED (memory managed by GPU driver)
///\                                                                          MEMORY_DEVICE ( a chunk of memory on the GPU device)
///\                                                                          MEMORY_DEVICE_ALIGNED (block of memory is memory bounds
///\                                                                          aligned - this means this memory may have some padding in
///\                                                                          each row of data, but it also means faster data transfer)
///\                                                                          MEMORY_HOST_LOCKED: buffer is created in page-locked host
///\                                                                          memory - meaning it can be seen from GPU so can be read/write
///\                                                                          directly to/from host memory; the drawback is that the more
///\                                                                          page-locked memory is used, the overall system memory access 
///\                                                                          is slowed down as page-swapping is more constrained so 
///\                                                                          overall system performance decreases).
///\  3.  WriteMemory():  overloaded method to read/write 8-bit signed, 8-bit unsigned, or 32-bit float data to GPU memory or 
///\                      read from GPU memory WRITE_HOST_2_DEV. WRITE_DEV_2_HOST (DEV=device).  There is also an option for
///\                      writing device to device WRITE_DEV_2_DEV to transfer between buffers on GPU device memory.
///\  4.  Execute():  This is a method that is to be overloaded for the different kernel function types.  The method signature
///\                  must match the signature of the wrapper functions in the *.cu file that encapsulate the kernel function
///\                  launches.  All Execute() overloaded functions, however, must include the parameters blockWidth, blockHeight
///\                  imageHeight, imageWidth (if the data is simply a 1-D array, the imageHeight can simply be set to 1, and the
///\                  block height must also be set to 1).
class GPU:public IGPU
{
public:
	GPU();
	~GPU();

	//Basic operations
	int InitDevice( uint8_t deviceIndex, bool bUseTiming );
	void ResetTimer(void);
	float GetExecutionTime(void);
	int AllocateMemory( size_t width, size_t height, DATA_TYPE dType, MEMORY_TYPE type );
	void FreeMemory(int deviceID);
	virtual int WriteData(int8_t* src, int dst, WRITE_PATH write);
	virtual int WriteData(uint8_t* src, int dst, WRITE_PATH write);
	virtual int WriteData(float_t* src, int dst, WRITE_PATH write);
	int Execute(KERNEL_FUNCTION_ID kernel, 
                int buffer, 
		        int kernelBuffer, 
		        int outBuffer, 
		        float_t scale, 
		        size_t blockW, 
		        size_t blockH, 
		        size_t imWidth, 
		        size_t imHeight, 
		        bool bUseSharedMemory=false);
	int GetDeviceCount();

private:
	int8_t m_nbrDevices;
	int8_t m_currentDevice;

	cudaEvent_t m_CUDAStart;
	cudaEvent_t m_CUDAStop;

	bool m_bUseTimer;

	float_t m_fKernelTime;

	std::vector<BUFFER_t> m_Buffer;
	std::map<uint8_t, std::string> m_FunctionList;
};

