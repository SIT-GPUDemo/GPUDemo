// ------------------------------------------------------------------------------------------------------------------ //
// GPU.cpp                                                                                                            //
// -----------------------------------------------------------------------------------------------------------------  //
#include<iostream>
#include<map>

#include<cuda_runtime.h>
#include "GPU.h"
#include "KernelWrappers.h"

//Function list: in an actual application, could have an INI file where function names are read
//from and loaded into the map of function indices and kernel function name strings; for now, a global
//list of kernel function names
std::string kerneFunctions[] = { std::string("Filter_3x3"),
								 std::string("Filter_5x5"),
								 std::string("Filter_7x7") };

GPU::GPU()
{
	m_nbrDevices = -1;
	m_bUseTimer = false;
	m_fKernelTime = 0.0f;
	m_CUDAStart = 0;
	m_CUDAStop = 0;
}


GPU::~GPU()
{
}

int GPU::InitDevice( uint8_t deviceIndex, bool bUseTimer )
{
	cudaError_t cudaStatus = cudaSuccess;

	cudaGetDeviceCount((int*)(&m_nbrDevices));

	if (m_nbrDevices < 1)
	{
		std::cout << "No GPU devices found." << std::endl;
		return -1;
	}
	if (deviceIndex > m_nbrDevices - 1)
	{
		std::cout << "Device index exceeds number of devices on GPU.  Check your device index" << std::endl;
		return -2;
	}

	cudaSetDevice(deviceIndex);
	m_currentDevice = deviceIndex;

	m_FunctionList.insert(std::pair<uint8_t, std::string>(static_cast<uint8_t>(KERNEL_FUNCTION_ID::KERNEL_FILTER_3x3), kerneFunctions[static_cast<uint8_t>(KERNEL_FUNCTION_ID::KERNEL_FILTER_3x3)]));
	m_FunctionList.insert(std::pair<uint8_t, std::string>(static_cast<uint8_t>(KERNEL_FUNCTION_ID::KERNEL_FILTER_5x5), kerneFunctions[static_cast<uint8_t>(KERNEL_FUNCTION_ID::KERNEL_FILTER_5x5)]));
	m_FunctionList.insert(std::pair<uint8_t, std::string>(static_cast<uint8_t>(KERNEL_FUNCTION_ID::KERNEL_FILTER_7x7), kerneFunctions[static_cast<uint8_t>(KERNEL_FUNCTION_ID::KERNEL_FILTER_7x7)]));

	m_bUseTimer = bUseTimer;
	m_fKernelTime = 0.0f;

	if (m_bUseTimer == true)
	{
		cudaStatus = cudaEventCreate(&m_CUDAStart);
		if (cudaStatus != cudaSuccess)
		{
			std::cerr << "ERROR: Unable to create event: START" << std::endl;
			m_bUseTimer = false;
		}
		cudaStatus = cudaEventCreate(&m_CUDAStop);
		if (cudaStatus != cudaSuccess)
		{
			std::cerr << "ERROR: Unable to create event: START" << std::endl;
			m_bUseTimer = false;
		}
	}

	return 1;
}


void GPU::ResetTimer(void)
{
	m_fKernelTime = 0.0f;
}

float GPU::GetExecutionTime(void)
{
	return m_fKernelTime;
}

//Note: we use the position in the vector of buffers as the buffer index that the caller will use
//to access specific buffers.  As long as we add buffers and release buffers all at once, that is fine.
//If we clear some of the buffers before we release the others, the sequencing of buffers will change.
//Hence, the buffer struct will now record the buffer index to be used to look up particular buffers 
//in subsequent searches (for when we want to add/delete buffers dynamically).
int GPU::AllocateMemory( size_t width, size_t height, DATA_TYPE dType, MEMORY_TYPE type )
{
	cudaError_t cudaStatus = cudaSuccess;
	int status = 1;

	BUFFER_t dataBuffer;
	size_t memoryBlockSize = 0;
	
	//Set the memory block size
	if (dType == DATA_TYPE::INT8 || dType == DATA_TYPE::UINT8)
	{
		memoryBlockSize = width * height;
	}
	else 
	{
		memoryBlockSize = width * height * 4;
	}
	
	//Allocate the buffers 
	if (type == MEMORY_TYPE::MEMORY_DEVICE)
	{
		if (dType == DATA_TYPE::INT8)
		{
			dataBuffer.ptr8u = nullptr;
			dataBuffer.ptr32f = nullptr;
			dataBuffer.type = DATA_TYPE::INT8;
			cudaStatus = cudaMalloc((void**)&dataBuffer.ptr8s, memoryBlockSize);
		}
		else if (dType == DATA_TYPE::UINT8)
		{
			dataBuffer.ptr8s = nullptr;
			dataBuffer.ptr32f = nullptr;
			dataBuffer.type = DATA_TYPE::UINT8;
			cudaStatus = cudaMalloc((void**)&dataBuffer.ptr8u, memoryBlockSize);
		}
		else
		{
			dataBuffer.ptr8s = nullptr;
			dataBuffer.ptr8u = nullptr;
			dataBuffer.type = DATA_TYPE::FLOAT32;
			cudaStatus = cudaMalloc((void**)&dataBuffer.ptr32f, memoryBlockSize );
		}
		dataBuffer.pitch = 0;
	}
	else if (type == MEMORY_TYPE::MEMORY_HOST_LOCKED)
	{
		if (dType == DATA_TYPE::INT8)
		{
			dataBuffer.ptr8u = nullptr;
			dataBuffer.ptr32f = nullptr;
			dataBuffer.type = DATA_TYPE::INT8;
			cudaStatus = cudaMallocHost((void**)&dataBuffer.ptr8s, memoryBlockSize);

		}
		else if (dType == DATA_TYPE::UINT8)
		{
			dataBuffer.ptr8s = nullptr;
			dataBuffer.ptr32f = nullptr;
			dataBuffer.type = DATA_TYPE::UINT8;
			cudaStatus = cudaMallocHost((void**)&dataBuffer.ptr8u, memoryBlockSize);
		}
		else
		{
			dataBuffer.ptr8s = nullptr;
			dataBuffer.ptr8u = nullptr;
			dataBuffer.type = DATA_TYPE::FLOAT32;
			cudaStatus = cudaMallocHost((void**)&dataBuffer.ptr32f, memoryBlockSize);
		}
		dataBuffer.pitch = 0;
	}
	else if (type == MEMORY_TYPE::MEMORY_DEVICE_ALIGNED)
	{
		if (dType == DATA_TYPE::INT8)
		{
			dataBuffer.ptr8u = nullptr;
			dataBuffer.ptr32f = nullptr;
			dataBuffer.type = DATA_TYPE::INT8;
			cudaStatus = cudaMallocPitch((void**)&dataBuffer.ptr8s, &dataBuffer.pitch, width, height);
		}
		else if (dType == DATA_TYPE::UINT8)
		{
			dataBuffer.ptr8s = nullptr;
			dataBuffer.ptr32f = nullptr;
			dataBuffer.type = DATA_TYPE::UINT8;
			cudaStatus = cudaMallocPitch((void**)&dataBuffer.ptr8u, &dataBuffer.pitch, width, height);
		}
		else
		{
			dataBuffer.ptr8s = nullptr;
			dataBuffer.ptr8u = nullptr;
			dataBuffer.type = DATA_TYPE::FLOAT32;
			cudaStatus = cudaMallocPitch((void**)&dataBuffer.ptr32f, &dataBuffer.pitch, width, height );
		}
	}
	else
	{
		if (dType == DATA_TYPE::INT8)
		{
			dataBuffer.ptr8u = nullptr;
			dataBuffer.ptr32f = nullptr;
			dataBuffer.type = DATA_TYPE::INT8;
			cudaStatus = cudaMallocManaged((void**)&dataBuffer.ptr8s, memoryBlockSize);
		}
		else if (dType == DATA_TYPE::UINT8)
		{
			dataBuffer.ptr8s = nullptr;
			dataBuffer.ptr32f = nullptr;
			dataBuffer.type = DATA_TYPE::UINT8;
			cudaStatus = cudaMallocManaged((void**)&dataBuffer.ptr8u, memoryBlockSize);
		}
		else
		{
			dataBuffer.ptr8s = nullptr;
			dataBuffer.ptr8u = nullptr;
			dataBuffer.type = DATA_TYPE::FLOAT32;
			cudaStatus = cudaMallocManaged((void**)&dataBuffer.ptr32f, memoryBlockSize);
		}
		dataBuffer.pitch = 0;
	}

	int bufferIndex = static_cast<int>(m_Buffer.size());  //This is the current buffer queue length, before we add this buffer
	                                                      //Because the indices are zero-based, the last element of the vector will be 
	                                                      //indexed by the size of the old vector
	dataBuffer.width = width;
	dataBuffer.height = height;
	dataBuffer.bufferIndex = bufferIndex;
	m_Buffer.push_back(dataBuffer);

	if (cudaStatus != cudaSuccess)
	{
		bufferIndex = -cudaStatus;
	}
	return bufferIndex;
}


int GPU::WriteData( int8_t* src, int dst, WRITE_PATH write )
{
	//The destination buffer index should be the index of the vector where the buffer is found
	//I say 'SHOULD' because when we add the capability to delete individual buffers, the sequence of buffer indices
	//in the may change so we would need to search through the list (vector) of buffers to find the appropriate one.  
	//However, for this demo, this has not been included, so we can leave this check aside here.
	cudaError_t cudaStatus = cudaSuccess;
	int status = 1;

	//Here we check if the allocated buffer was memory aligned (with cudaMallocPitch) in which case the buffer pitch is 
	//non-zero or wether it was not memory aligned
	if (m_Buffer[dst].pitch != 0)
	{
		//For now, write host to device and device to host
		//Also, since input is 8-bit signed, we will write to the 8-bit signed device buffer (only check will be that this has
		//been allocated)
		if (m_Buffer[dst].ptr8s == nullptr)
		{
			std::cout << "ERROR: Device buffer for 8-bit signed input not allocated." << std::endl;
			return -1;
		}
		if (write == WRITE_PATH::WRITE_HOST_2_DEV)
		{
			cudaStatus = cudaMemcpy2D(m_Buffer[dst].ptr8s, m_Buffer[dst].pitch, src, m_Buffer[dst].width, m_Buffer[dst].width, m_Buffer[dst].height, cudaMemcpyHostToDevice);
		}
		else
		{
			cudaStatus = cudaMemcpy2D(src, m_Buffer[dst].pitch, m_Buffer[dst].ptr8s, m_Buffer[dst].width, m_Buffer[dst].width, m_Buffer[dst].height, cudaMemcpyDeviceToHost);
		}
	}
	else
	{
		//For now, write host to device and device to host
		//Also, since input is 8-bit signed, we will write to the 8-bit signed device buffer (only check will be that this has
		//been allocated)
		if (m_Buffer[dst].ptr8s == nullptr)
		{
			std::cout << "ERROR: Device buffer for 8-bit signed input not allocated." << std::endl;
			return -1;
		}
		if (write == WRITE_PATH::WRITE_HOST_2_DEV)
		{
			cudaStatus = cudaMemcpy(m_Buffer[dst].ptr8s, src, m_Buffer[dst].width*m_Buffer[dst].height, cudaMemcpyHostToDevice);
		}
		else
		{
			cudaStatus = cudaMemcpy(src, m_Buffer[dst].ptr8s, m_Buffer[dst].width*m_Buffer[dst].height, cudaMemcpyDeviceToHost);
		}
	}

	if (cudaStatus != cudaSuccess)
	{
		status = cudaStatus;
	}
	return status;
}

int GPU::WriteData(uint8_t* src, int dst, WRITE_PATH write)
{
	//The destination buffer index should be the index of the vector where the buffer is found
	//I say 'SHOULD' because when we add the capability to delete individual buffers, the sequence of buffer indices
	//in the may change so we would need to search through the list (vector) of buffers to find the appropriate one.  
	//However, for this demo, this has not been included, so we can leave this check aside here.

	cudaError_t cudaStatus = cudaSuccess;
	int status = 1;

	//Here we check if the allocated buffer was memory aligned (with cudaMallocPitch) in which case the buffer pitch is 
	//non-zero or wether it was not memory aligned
	if (m_Buffer[dst].pitch != 0)
	{
		//For now, write host to device and device to host
		//Also, since input is 8-bit unsigned, we will write to the 8-bit unsigned device buffer (only check will be that this has
		//been allocated)
		if (m_Buffer[dst].ptr8u == nullptr)
		{
			std::cout << "ERROR: Device buffer for 8-bit unsigned input not allocated." << std::endl;
			return -1;
		}
		if (write == WRITE_PATH::WRITE_HOST_2_DEV)
		{
			cudaStatus = cudaMemcpy2D(m_Buffer[dst].ptr8u, m_Buffer[dst].pitch, src, m_Buffer[dst].width, m_Buffer[dst].width, m_Buffer[dst].height, cudaMemcpyHostToDevice);
		}
		else
		{
			cudaStatus = cudaMemcpy2D(src, m_Buffer[dst].pitch, m_Buffer[dst].ptr8u, m_Buffer[dst].width, m_Buffer[dst].width, m_Buffer[dst].height, cudaMemcpyDeviceToHost);
		}
	}
	else
	{
		//For now, write host to device and device to host
		//Also, since input is 8-bit unsigned, we will write to the 8-bit unsigned device buffer (only check will be that this has
		//been allocated)
		if (m_Buffer[dst].ptr8u == nullptr)
		{
			std::cout << "ERROR: Device buffer for 8-bit signed input not allocated." << std::endl;
			return -1;
		}
		if (write == WRITE_PATH::WRITE_HOST_2_DEV)
		{
			cudaStatus = cudaMemcpy(m_Buffer[dst].ptr8u, src, m_Buffer[dst].width*m_Buffer[dst].height, cudaMemcpyHostToDevice);
		}
		else
		{
			cudaStatus = cudaMemcpy(src, m_Buffer[dst].ptr8u, m_Buffer[dst].width*m_Buffer[dst].height, cudaMemcpyDeviceToHost);
		}
	}

	if (cudaStatus != cudaSuccess)
	{
		status = cudaStatus;
	}

	return status;
}

int GPU::WriteData(float_t* src, int dst, WRITE_PATH write)
{
	cudaError_t cudaStatus = cudaSuccess;

	//The destination buffer index should be the index of the vector where the buffer is found
	//I say 'SHOULD' because when we add the capability to delete individual buffers, the sequence of buffer indices
	//in the may change so we would need to search through the list (vector) of buffers to find the appropriate one.  
	//However, for this demo, this has not been included, so we can leave this check aside here.

	//Here we check if the allocated buffer was memory aligned (with cudaMallocPitch) in which case the buffer pitch is 
	//non-zero or wether it was not memory aligned
	if (m_Buffer[dst].pitch != 0)
	{
		//For now, write host to device and device to host
		//Also, since input is 32-bit float, we will write to the 32-bit float device buffer (only check will be that this has
		//been allocated)
		if (m_Buffer[dst].ptr32f == nullptr)
		{
			std::cout << "ERROR: Device buffer for 32-bit floating point input not allocated." << std::endl;
			return -1;
		}
		if (write == WRITE_PATH::WRITE_HOST_2_DEV)
		{
			cudaStatus = cudaMemcpy2D(m_Buffer[dst].ptr32f, m_Buffer[dst].pitch, src, m_Buffer[dst].width, m_Buffer[dst].width, m_Buffer[dst].height, cudaMemcpyHostToDevice);
		}
		else
		{
			cudaStatus = cudaMemcpy2D(src, m_Buffer[dst].pitch, m_Buffer[dst].ptr32f, m_Buffer[dst].width, m_Buffer[dst].width, m_Buffer[dst].height, cudaMemcpyDeviceToHost);
		}
	}
	else
	{
		//For now, write host to device and device to host
		//Also, since input is 32-bit floating point buffer, we will write to the 32-bit floating point device buffer (only check will be that this has
		//been allocated)
		if (m_Buffer[dst].ptr32f == nullptr)
		{
			std::cout << "ERROR: Device buffer for 32-bit floating point input not allocated." << std::endl;
			return -1;
		}
		if (write == WRITE_PATH::WRITE_HOST_2_DEV)
		{
			cudaStatus = cudaMemcpy(m_Buffer[dst].ptr32f, src, m_Buffer[dst].width*m_Buffer[dst].height*sizeof(float_t), cudaMemcpyHostToDevice);
		}
		else
		{
			cudaStatus = cudaMemcpy(src, m_Buffer[dst].ptr32f, m_Buffer[dst].width*m_Buffer[dst].height*sizeof(float_t), cudaMemcpyDeviceToHost);
		}
	}

	int status = 1;
	if (cudaStatus != cudaSuccess)
	{
		status = cudaStatus;
	}
	return status;
}

int GPU::Execute(KERNEL_FUNCTION_ID kernel, 
	             int buffer, 
	             int kernelBuffer, 
	             int outBuffer, 
	             float_t scale, 
	             size_t blockX, 
	             size_t blockY, 
	             size_t imWidth, 
	             size_t imHeight, 
	             bool bUseSharedMem)
{
	int status = 1;

	size_t gX = (imWidth + blockX - 1 )/ blockX;
	size_t gY = (imHeight + blockY - 1) / blockY;
	int w = static_cast<int>(imWidth);
	int h = static_cast<int>(imHeight);

	kernelWrapper::KernelParams params;
	params.bX = blockX;
	params.bY = blockY;
	params.gX = gX;
	params.gY = gY;
	params.fTimeCount = 0.0f;

	if (m_bUseTimer)
	{
		params.bUseTimer = true;
		params.cudaEvtStart = m_CUDAStart;
		params.cudaEvtStop = m_CUDAStop;
	}

	switch (kernel)
	{
		case KERNEL_FUNCTION_ID::KERNEL_FILTER_3x3:

			break;
		case KERNEL_FUNCTION_ID::KERNEL_FILTER_5x5:
			if (bUseSharedMem == true)
			{
				status = kernelWrapper::filterImage5x5_s((kernelWrapper::KernelParams*)&params, m_Buffer[buffer].ptr32f, m_Buffer[kernelBuffer].ptr32f,  m_Buffer[outBuffer].ptr32f, scale, w, h );
				m_fKernelTime += params.fTimeCount;
			}
			else
			{
				status = kernelWrapper::filterImage5x5_g((kernelWrapper::KernelParams*)&params, m_Buffer[buffer].ptr32f, m_Buffer[kernelBuffer].ptr32f, m_Buffer[outBuffer].ptr32f, scale, w, h);
				m_fKernelTime += params.fTimeCount;
			}
			break;
		case KERNEL_FUNCTION_ID::KERNEL_FILTER_7x7:
			break;
		default:
			break;
	}

	return status;
}

int GPU::GetDeviceCount()
{
	int count = 0;
	cudaGetDeviceCount(&count);
	return count;
}

void GPU::FreeMemory(int deviceID)
{
	int index = 0;
	int bufferIndex = 0;
	for (auto element : m_Buffer)
	{
		if (element.bufferIndex == deviceID)
		{
			if (element.ptr32f != nullptr)
			{
				cudaFree(element.ptr32f);
				element.ptr32f = nullptr;
			}
			else if (element.ptr8s != nullptr)
			{
				cudaFree(element.ptr8s);
				element.ptr8s = nullptr;
			}
			else
			{
				cudaFree(element.ptr8u);
				element.ptr8u = nullptr;
			}
			bufferIndex = index;
			break;
		}
		index++;
	}

	m_Buffer.erase(m_Buffer.begin() + bufferIndex);

}