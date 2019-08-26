#include<iostream>
#include<cv.h>
#include<imgcodecs.hpp>
#include"../GPULib/GPUDevice.h"

#define OUTPUT_DEMO_SHARED_MEM_FILE "RoadLargeWithFilterOnSharedMemory.png"

int FilterOnSharedMemory(IGPU* pGPU,
	int devImageBuffer,
	int devKernelLoG5x5,
	int devOutputBuffer2,
	float fScale,
	int bW,
	int bH,
	int w,
	int h,
	bool bUseTimer)
{
	int status = 0;
	status = pGPU->Execute(KERNEL_FUNCTION_ID::KERNEL_FILTER_5x5, devImageBuffer, devKernelLoG5x5, devOutputBuffer2, fScale, 128, 8, w, h, true);
	if (bUseTimer == true)
	{
		pGPU->ResetTimer();
	}

	float fExecutionTime = 0.0f;
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
	//OpenCV cv::Mat structures for retrieving data
	cv::Mat outImageF(static_cast<int>(h), static_cast<int>(w), CV_32FC1);
	cv::Mat outImage(static_cast<int>(h), static_cast<int>(w), CV_8UC1);


	status = pGPU->WriteData((float_t*)outImageF.data, devOutputBuffer2, WRITE_PATH::WRITE_DEV_2_HOST);
	outImageF.convertTo(outImage, CV_8UC1, 1.0, 0.0);

	cv::imwrite(std::string(OUTPUT_DEMO_SHARED_MEM_FILE), outImage);

	return status;
}