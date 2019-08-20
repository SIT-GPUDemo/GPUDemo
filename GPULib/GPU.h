#pragma once
#include<cuda_runtime.h>
#include "GPUDevice.h"
#include "GPUTypes.h"
#include<vector>
#include<map>

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

