// ------------------------------------------------------------------------------------------------------------- //
// FilterTests.cpp                                                                                               //
// ------------------------------------------------------------------------------------------------------------- //
// Description: Programme to load the image data from disk, call the functions to initialise the GPU and allocate//
//              resources, and dispatch the kernel.  The file contains OpenCV references, as well as any other   //
//              external libraries. Calls to GPU initialisation/resource allocation are in a separate file to    //
//              keep GPU software references separate from the application specific code.                        //
// ------------------------------------------------------------------------------------------------------------- //

#include<cv.h>
#include<highgui.h>
#include<imgcodecs.hpp>
#include<iostream>
#include<memory>
#include<new>
#include<thread>
#include<chrono>
#include"../GPULib/GPUDevice.h"
#include"../GPULib/GPUTypes.h"
#include"kernelConvolutions.h"

#define INPUT_DEMO_FILE "RoadLarge.bmp"
#define OUTPUT_DEMO_FILE "RoadLargeWithFilter.png"
#define OUTPUT_DEMO_SHARED_MEM_FILE "RoadLargeWithFilterOnSharedMemory.png"

int main(int argc, char** argv)
{
	int status = 0;

	//Loading disk image and check input data validity
	std::string inputImageFile = INPUT_DEMO_FILE;
	cv::Mat inputImage = cv::imread(inputImageFile, cv::IMREAD_GRAYSCALE);
	size_t w = inputImage.size().width;
	size_t h = inputImage.size().height;

	cv::Mat inputImageF(static_cast<int>(h), static_cast<int>(w), CV_32FC1);
	inputImage.convertTo(inputImageF, CV_32FC1, 1.0, 0.0);

	if (inputImage.empty() == true)
	{
		std::cerr << "Image load failed.  Check image file name and path." << std::endl;
		std::this_thread::sleep_for(std::chrono::microseconds(2000000));
		return 0;
	}

	
	//Create the GPU devices and allocate resources (memory)
	IGPU* pGPU = CreateGPU();
	bool bUseTimer = true;
	pGPU->InitDevice(0, bUseTimer );

	//Allocate image buffers (in different configurations)
	//   1.  Device buffer (image lines not memory bounds aligned)
	//   2.  Device buffer (image in page-locked host memory mapped to GPU)
	//   3.  Device buffer (image in GPU driver managed memory)
	//   4,  Device buffer - memory bounds aligned allocation (so start of image rows are aligned with memory bounds - padding 
	//       may be occasionally added )
	int devImageBuffer = pGPU->AllocateMemory(w, h, DATA_TYPE::FLOAT32, MEMORY_TYPE::MEMORY_DEVICE);
	int devImageBufferHostLocked = pGPU->AllocateMemory(w, h, DATA_TYPE::FLOAT32, MEMORY_TYPE::MEMORY_HOST_LOCKED);
	int devImageBufferManaged = pGPU->AllocateMemory(w, h, DATA_TYPE::FLOAT32, MEMORY_TYPE::MEMORY_MANAGED);
	int devImageBufferAligned = pGPU->AllocateMemory(w, h, DATA_TYPE::FLOAT32, MEMORY_TYPE::MEMORY_DEVICE_ALIGNED);

	//Output buffers
	int devOutputBuffer1 = pGPU->AllocateMemory(w, h, DATA_TYPE::FLOAT32, MEMORY_TYPE::MEMORY_DEVICE);
	int devOutputBuffer2 = pGPU->AllocateMemory(w, h, DATA_TYPE::FLOAT32, MEMORY_TYPE::MEMORY_DEVICE);

	//Write the host image data to the GPU
	status = pGPU->WriteData((float_t*)(inputImageF.data), devImageBuffer, WRITE_PATH::WRITE_HOST_2_DEV);
	status = pGPU->WriteData((float_t*)(inputImageF.data), devImageBufferHostLocked, WRITE_PATH::WRITE_HOST_2_DEV);
	status = pGPU->WriteData((float_t*)(inputImageF.data), devImageBufferManaged, WRITE_PATH::WRITE_HOST_2_DEV);
	status = pGPU->WriteData((float_t*)(inputImageF.data), devImageBufferAligned, WRITE_PATH::WRITE_HOST_2_DEV);

	//Create a convolution kernel and initialise the kernel
	int devKernelLoG5x5 = pGPU->AllocateMemory(5, 5, DATA_TYPE::FLOAT32, MEMORY_TYPE::MEMORY_DEVICE);
	status = pGPU->WriteData((float_t*)kernelsConvolution::LoG5x5, devKernelLoG5x5, WRITE_PATH::WRITE_HOST_2_DEV);

	//Execute: Laplacian of Gradient (5x5)
	float fScale = 1.0f / 16.0f;

	//1.  using global memory access-------------------------------------------------------------------------
	//  First call is to complete allocation/loading of kernel
	//  Then call is executed 10 times with the timer for all 10 calls to average out individual scheduling delays
	status = pGPU->Execute(KERNEL_FUNCTION_ID::KERNEL_FILTER_5x5, devImageBuffer, devKernelLoG5x5, devOutputBuffer1, fScale, 32, 16, w, h, false);
	if (bUseTimer == true)
	{
		pGPU->ResetTimer();
	}

	float fExecutionTime = 0.0f;
	for (size_t iteration = 0; iteration < 10; iteration++)
	{
		status = pGPU->Execute(KERNEL_FUNCTION_ID::KERNEL_FILTER_5x5, devImageBuffer, devKernelLoG5x5, devOutputBuffer1, fScale, 32, 16, w, h);
	}
	if (bUseTimer == true)
	{
		float executionTime = pGPU->GetExecutionTime();
		pGPU->ResetTimer();
		std::cout << "Kernel execution time: " << executionTime / 10.0f << std::endl;
	}
	//
	//Retrieve data
	cv::Mat outImageF(static_cast<int>(h), static_cast<int>(w), CV_32FC1);
	cv::Mat outImage(static_cast<int>(h), static_cast<int>(w), CV_8UC1);


	status = pGPU->WriteData((float_t*)outImageF.data, devOutputBuffer1, WRITE_PATH::WRITE_DEV_2_HOST);
	outImageF.convertTo(outImage, CV_8UC1, 1.0, 0.0);

	cv::imwrite(std::string(OUTPUT_DEMO_FILE), outImage);


	//2.  using shared memory access-------------------------------------------------------------------------
	//  First call is to complete allocation/loading of kernel
	//  Then call is executed 10 times with the timer for all 10 calls to average out individual scheduling delays
	status = pGPU->Execute(KERNEL_FUNCTION_ID::KERNEL_FILTER_5x5, devImageBuffer, devKernelLoG5x5, devOutputBuffer2, fScale, 128, 8, w, h, true);
	if (bUseTimer == true)
	{
		pGPU->ResetTimer();
	}

	fExecutionTime = 0.0f;
	for (size_t iteration = 0; iteration < 10; iteration++)
	{
		status = pGPU->Execute(KERNEL_FUNCTION_ID::KERNEL_FILTER_5x5, devImageBuffer, devKernelLoG5x5, devOutputBuffer2, fScale, 128, 8, w, h, true);
	}
	if (bUseTimer == true)
	{
		float executionTime = pGPU->GetExecutionTime();
		pGPU->ResetTimer();
		std::cout << "Kernel execution time: " << executionTime / 10.0f << std::endl;
	}
	//
	//Retrieve data
	status = pGPU->WriteData((float_t*)outImageF.data, devOutputBuffer2, WRITE_PATH::WRITE_DEV_2_HOST);
	outImageF.convertTo(outImage, CV_8UC1, 1.0, 0.0);

	cv::imwrite(std::string(OUTPUT_DEMO_SHARED_MEM_FILE), outImage);

	return 0;
}