#include <stdio.h>
#include <iostream>
#include "CUDASharedDefs.h"
#define CONVOLUTION_KERNEL_SIZE_3   3
#define CONVOLUTION_KERNEL_SIZE_5	5
#define CONVOLUTION_KERNEL_SIZE_7   7
#define CONVOLUTION_KERNEL_SIZE_9   9

#define BLOCK_SIZE_X 256
#define BLOCK_SIZE_Y 16

extern struct KERNEL_PARAMS* params;

__global__ void s_filterImage_5x5(float* imageIn, float* kernel, float* imageOut, float scaleFactor, int imW, int imH)
{
	unsigned int c = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int r = blockDim.y*blockIdx.y + threadIdx.y;
	unsigned int bufferIndex = r * imW + c;

	//check that the thread is within the region of the image
	if (c >= imW || r >= imH)
	{
		return;
	}

	__shared__ float inBuffer[BLOCK_SIZE_X * BLOCK_SIZE_Y + 2 * CONVOLUTION_KERNEL_SIZE_5 + CONVOLUTION_KERNEL_SIZE_5 * CONVOLUTION_KERNEL_SIZE_5];
	__shared__ float outBuffer[BLOCK_SIZE_X* BLOCK_SIZE_Y + 2 * CONVOLUTION_KERNEL_SIZE_5 + CONVOLUTION_KERNEL_SIZE_5 * CONVOLUTION_KERNEL_SIZE_5];
	__shared__ float s_kernel[CONVOLUTION_KERNEL_SIZE_5*CONVOLUTION_KERNEL_SIZE_5];

	//check that the thread is within the region of the image
	if (c >= imW || r >= imH)
	{
		return;
	}

	//check that the threads are within the kernel mask region; if they are within, copy the
	//mask to the shared buffer
	if (threadIdx.x < CONVOLUTION_KERNEL_SIZE_5 && threadIdx.y < CONVOLUTION_KERNEL_SIZE_5)
	{
		s_kernel[threadIdx.y*CONVOLUTION_KERNEL_SIZE_5 + threadIdx.x] = kernel[threadIdx.y*CONVOLUTION_KERNEL_SIZE_5 + threadIdx.x];
	}

	//fetch the data into the shared memory
	//Contiguous threads fetch data from contiguous addresses in global memory (bufferIndex varies by 1 between 
	//contiguous threads) so coalescing
	//First 2 feteches: block of image data copied to local memory; 2nd fetch is offset by the convolution kernel size 
	inBuffer[threadIdx.y*BLOCK_SIZE_X + threadIdx.x] = imageIn[bufferIndex];

	//Compute the convolution and write to the shared output buffer
	if ((c < (CONVOLUTION_KERNEL_SIZE_5 / 2)) || 
		(c > (imW - CONVOLUTION_KERNEL_SIZE_5 / 2)) || 
		(r < (CONVOLUTION_KERNEL_SIZE_5/2)) ||
		(r > (imH-CONVOLUTION_KERNEL_SIZE_5/2)))
	{
		imageOut[bufferIndex] = imageIn[threadIdx.y*(BLOCK_SIZE_X + CONVOLUTION_KERNEL_SIZE_5)];
	}
	else
	{
		float partialSum = 0.0f;
		inBuffer[threadIdx.y*(BLOCK_SIZE_X + CONVOLUTION_KERNEL_SIZE_5)] = imageIn[(blockIdx.y*blockDim.y + threadIdx.y-CONVOLUTION_KERNEL_SIZE_5/2)*imW + (blockIdx.x*blockDim.x + threadIdx.x-CONVOLUTION_KERNEL_SIZE_5/2)];
		inBuffer[threadIdx.y*(BLOCK_SIZE_X + CONVOLUTION_KERNEL_SIZE_5) + CONVOLUTION_KERNEL_SIZE_5] = imageIn[(blockIdx.y*blockDim.y + threadIdx.y-CONVOLUTION_KERNEL_SIZE_5/2)*imW + (blockIdx.x*blockDim.x + threadIdx.x + CONVOLUTION_KERNEL_SIZE_5/2)];
		inBuffer[(threadIdx.y + CONVOLUTION_KERNEL_SIZE_5)*(BLOCK_SIZE_X + CONVOLUTION_KERNEL_SIZE_5) + threadIdx.x] = imageIn[(blockIdx.y*blockDim.y + threadIdx.y+CONVOLUTION_KERNEL_SIZE_5/2)*imW + (blockIdx.x*blockDim.x + threadIdx.x-CONVOLUTION_KERNEL_SIZE_5/2)];
		inBuffer[(threadIdx.y + CONVOLUTION_KERNEL_SIZE_5)*(BLOCK_SIZE_X + CONVOLUTION_KERNEL_SIZE_5) + threadIdx.x + (3*CONVOLUTION_KERNEL_SIZE_5/2)] = imageIn[(blockIdx.y*blockDim.y + threadIdx.y + CONVOLUTION_KERNEL_SIZE_5 / 2)*imW + (blockIdx.x*blockDim.x + threadIdx.x + CONVOLUTION_KERNEL_SIZE_5 / 2)];

		//Loop unrolling candidate *************************************************
		for (int row = 0; row < CONVOLUTION_KERNEL_SIZE_5; row++)
		{
			for (int col = 0; col < CONVOLUTION_KERNEL_SIZE_5; col++)
			{
				//Multiple accesses to the image buffer, by each thread
				//However, by having stored in shared memory, MUCH faster since no 
				//need for global access
				float val = inBuffer[threadIdx.y*(BLOCK_SIZE_X + CONVOLUTION_KERNEL_SIZE_5) + row * (BLOCK_SIZE_X + CONVOLUTION_KERNEL_SIZE_5) + threadIdx.x + col];
				float kVal = s_kernel[row*CONVOLUTION_KERNEL_SIZE_5 + col];
				partialSum +=  kVal*val;
			}
		}
		//outBuffer[pixIdx] = partialSum * scaleFactor;
		imageOut[bufferIndex] = partialSum * scaleFactor;
		// ****************************************************************************
	}
	//imageOut[bufferIndex] = outBuffer[pixIdx];

}


__global__ void g_filterImage_5x5(float* imageIn, float* kernel, float* imageOut, float scaleFactor, int imW, int imH)
{
	unsigned int c = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int r = blockDim.y*blockIdx.y + threadIdx.y;
	unsigned int bufferIndex = r * imW + c;

	//check that the thread is within the region of the image
	if (c >= imW || r >= imH)
	{
		return;
	}

	//Compute the convolution and write to the shared output buffer
	if ((c < (CONVOLUTION_KERNEL_SIZE_5 / 2)) ||
		(c > (imW - CONVOLUTION_KERNEL_SIZE_5 / 2)) ||
		(r < (CONVOLUTION_KERNEL_SIZE_5 / 2)) ||
		(r > (imH - CONVOLUTION_KERNEL_SIZE_5 / 2)))
	{
		imageOut[bufferIndex] = imageIn[bufferIndex];
	}
	else
	{
		float cumulativeSum = 0.0;

		for (int kY = 0; kY < CONVOLUTION_KERNEL_SIZE_5; kY++)
		{
			for (int kX = 0; kX < CONVOLUTION_KERNEL_SIZE_5; kX++)
			{
				cumulativeSum += imageIn[(r + kY - 2)*imW + c + kX - 2] * kernel[kY*CONVOLUTION_KERNEL_SIZE_5 + kX];
			}
		}
		imageOut[r*imW + c] = cumulativeSum*scaleFactor;
	}



	return;
}

namespace kernelWrapper
{
	extern "C" int filterImage5x5_g(kernelWrapper::KernelParams* p, float* imageIn, float* kernel, float* imageOut, float scaleFactor, int imW, int imH)
	{
		cudaError_t cudaStatus = cudaSuccess;

		//dim3 gridSize(static_cast<int>(gX), static_cast<int>(gY));
		dim3 gridSize(static_cast<int>(p->gX), static_cast<int>(p->gY));
		dim3 blockSize(static_cast<int>(p->bX), static_cast<int>(p->bY));
		float runTime = 0.0f;

		if (p->bUseTimer == true)
		{
			cudaStatus = cudaEventRecord(p->cudaEvtStart);
			if (cudaStatus != cudaSuccess)
			{
				std::cerr << "ERROR: cudaEventRecord() START failed.  Error code: " << cudaStatus << std::endl;
				std::cerr << "Error message: " << cudaGetErrorString(cudaStatus) << std::endl;
			}
		}
		g_filterImage_5x5 << <gridSize, blockSize >> > (imageIn, kernel, imageOut, scaleFactor, imW, imH);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			std::cerr << "ERROR: Kernel execution failure.  Error code: " << cudaStatus << std::endl;
			std::cerr << "Error message: " << cudaGetErrorString(cudaStatus) << std::endl;
		}
		else
		{
			if (p->bUseTimer == true)
			{
				cudaStatus = cudaEventRecord(p->cudaEvtStop);
				if (cudaStatus != cudaSuccess)
				{
					std::cerr << "ERROR: cudaEventRecord() STOP failed.  Error code: " << cudaStatus << std::endl;
					std::cerr << "Error message: " << cudaGetErrorString(cudaStatus) << std::endl;
				}

				cudaStatus = cudaThreadSynchronize();
				cudaStatus = cudaEventElapsedTime(&runTime, p->cudaEvtStart, p->cudaEvtStop);
				if (cudaStatus != cudaSuccess)
				{
					std::cerr << "ERROR: cudaEventElapsedTime() failed.  Error code: " << cudaStatus << std::endl;
					std::cerr << "Error message: " << cudaGetErrorString(cudaStatus) << std::endl;
				}


				p->fTimeCount += runTime;
				printf("Execution time: %g\n", runTime);
			}
		}

		return static_cast<int>(cudaStatus);
	}

	extern "C" int filterImage5x5_s(kernelWrapper::KERNEL_PARAMS* p, float* imageIn, float* kernel, float* imageOut, float scaleFactor, int imW, int imH)
	{
		cudaError_t cudaStatus = cudaSuccess;

		dim3 gridSize(static_cast<int>(p->gX), static_cast<int>(p->gY));
		dim3 blockSize(static_cast<int>(p->bX), static_cast<int>(p->bY));
		float runTime = 0.0f;

		if (p->bUseTimer == true)
		{
			cudaStatus = cudaEventRecord(p->cudaEvtStart);
			if (cudaStatus != cudaSuccess)
			{
				std::cerr << "ERROR: cudaEventRecord() START failed.  Error code: " << cudaStatus << std::endl;
				std::cerr << "Error message: " << cudaGetErrorString(cudaStatus) << std::endl;
			}
		}
		s_filterImage_5x5 << <gridSize, blockSize >> > (imageIn, kernel, imageOut, scaleFactor, imW, imH);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			std::cerr << "ERROR: Kernel execution failure.  Error code: " << cudaStatus << std::endl;
			std::cerr << "Error message: " << cudaGetErrorString(cudaStatus) << std::endl;
		}
		else
		{
			if (p->bUseTimer == true)
			{

				cudaStatus = cudaEventRecord(p->cudaEvtStop);
				if (cudaStatus != cudaSuccess)
				{
					std::cerr << "ERROR: cudaEventRecord() STOP failed.  Error code: " << cudaStatus << std::endl;
					std::cerr << "Error message: " << cudaGetErrorString(cudaStatus) << std::endl;
				}
				cudaStatus = cudaThreadSynchronize();
				cudaStatus = cudaEventElapsedTime(&runTime, p->cudaEvtStart, p->cudaEvtStop);
				if (cudaStatus != cudaSuccess)
				{
					std::cerr << "ERROR: cudaEventElapsedTime() failed.  Error code: " << cudaStatus << std::endl;
					std::cerr << "Error message: " << cudaGetErrorString(cudaStatus) << std::endl;
				}

				p->fTimeCount += runTime;
				printf("Execution time: %g\n", runTime);
			}
		}

		return static_cast<int>(cudaStatus);
	}
}