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

///Kernel: s_filterImage_5x5
///Description: 5x5 Laplace of Gaussian filter - using shared memory
///Notes: This is an example of how you would use shared memory; Note that in some cases, the effort of copying the 
///       data to shared memory may add overhead and make the kernel slower.  
/// 
///       Shared memory is especially useful when you have data in the source that you need to use repeatedly.  
/// 
///       If you only need to use the data once (as is the case here, where you multiply the pixel value by corresponding
///       kernel element, and you no longer need the original pixel value), you will often get a slower kernel.
///
///       However, suppose that you apply a convolution filter and then want to subtract the initial value
///       in the image from the convolved image (so you access the same pixel for the initial convolution 
///       and then for the subtraction), then you have multiple accesses to the same pixel, at which point shared memory
///       makes sens.
///
///        Of course, there is another issue with this kernel:  I am copying some data multiple times.  One of the problems
///        in convolution is how to handle the boundaries.  I have chosen to copy the pixels surrounding the area I am 
///        processing into the shared buffer so that my convolution kernel can move over all the pixels in the block of the
///        image I am working on in the shared buffer.  However, the boundary region is not actually updated.  And every
///        block of threads that is dispatched has to fetch the boundary pixels from the image in global memory, adding to
///        extra memory transfers.  
///       
///        I could avoid this problem by processing only the pixels where the convolution kernel is entirely within the 
///        block of pixels stored in the shared memory, but that means there are bands of pixels on the edges of the working
///        image block (in the shared memory buffer) that will not be convolved.  That means the convolution will not be 
///        as accurate.  However, the tradeoff is that speed will be improved.
///   
///        Actually, there is another approach, not shown in my demo, where you can use texture memory.  The image gets mapped
///        onto texture memory where read access is very fast (but loading into texture memory does have significant overhead)
///        so this is only useful if you will be using the image over and over again.    Texture memory is a special memory 
///        which are usable by texture shaders, which are graphics programmes that are used for geometric image manipulation
///        (scaling, rotation, interpolation)
///
///  ---------------------------------------------------------------------------------------------------------------------
///  Description of code:
///  The CUDA compiler (and OpenCL compiler, also) provide some special variables (whose values are initialised by the driver at
///  runtime) that identify the location of the thread in the data buffer.  These variables are:
///     blockDim -> provides the size of the thread block (work-group) in the x, y, z dimensions.  These values are set 
///                 when the kernel is dispatches and when you specify the thread block (work-group) size.  This will be 
///                 described further below.
///     threadIdx -> The compiler defines these variables, but their values are explicitly set by the driver at run time.
///                  This variable indicates the thread in the x, y, z directions of the work group/thread block.  I have not
///                  discussed it, but the threads are launched in a configuration that defines their memory access patterns.
///                  Linear thread blocks/work-groups are launched in a manner that each thread operates on a memory address 
///                  where each thread access a sequential memory address.
///                  2-D thread blocks/work-groups have threads which can access memory addresses in a 2-D layout (arrays).
///                  It is alos possible to access 3-D patterns.  The threadIdx variables has x, y, z fields, to show the
///                  the configuration of the thread in terms of memory access.
///
///     Next, shared memory buffer blocks are allocated using the keyword __shared__ telling the compiler that the
///     requested buffer is to be stored in shared memory (that is, on-chip memory).  Note that most GPUs only have up
///     to 48k of shared memory.
///
///     After this, we fetch the data in global memory and copy into the shared memory.  We have to consider the memory 
///     access patterns when fetching data from global memory.  If each thread fetches from a location that is scattered
///     im memory (so contiguous threads do not access contiguous memory addresses), each fetch (or write, as the case
///     may be) is serialised.  So instead of having one memory transaction fetching N data items, there will be N memory
///     transactions fetching one memory item).  In the example shown below, each global memory read is offset by one
///     item (the global memory index for the read is):
///         (blockIdx.y*blockDim.y + threadIdx.y-CONVOLUTION_KERNEL_SIZE_5/2)*imW + (blockIdx.x*blockDim.x + threadIdx.x-CONVOLUTION_KERNEL_SIZE_5/2)
///     The first part of the index 
///           (blockIdx.y*blockDim.y + threadIdx.y-CONVOLUTION_KERNEL_SIZE_5/2)*imW
///     accesses different rows, and there is an offset of one image width between image rows.  However, the index in the horizontal direction is
///     given by: 
///           (blockIdx.x*blockDim.x + threadIdx.x-CONVOLUTION_KERNEL_SIZE_5/2)
///     where adjacent threads are offset by threadIdx.x, whose value changes by 1 between adjacent threads.
///     As a result, the driver will use a technique called memory coalescing, which will merge the transactions from a same block of
///     memory into a single transaction, thereby increasing the data throughput of the GPU (i.e. effective bandwidth).  Note that 
///     this technique is valid whether you are using shared memory or not.
///
///
__global__ void s_filterImage_5x5(float* imageIn, float* kernel, float* imageOut, float scaleFactor, int imW, int imH)
{
	///Thread block configuration section -----------------------------------------------------------------------------
	unsigned int c = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned int r = blockDim.y*blockIdx.y + threadIdx.y;
	unsigned int bufferIndex = r * imW + c;

	//check that the thread is within the region of the image; we do not want the thread to access an 
	//area of memory that is outside the buffer (potentially leading to illegal memory access if the 
	//specified address is protected - otherwise, it will simply fetch rubbish but you cannot know this ahead of time
	//so it is best to avoid going out of memory bounds.
	if (c >= imW || r >= imH)
	{
		return;
	}

	///Define blocks of memory that are reserved on the shared memory of the GPU core/compute unit
	///Note the size of the input buffer.  It is the size of block
	__shared__ float inBuffer[BLOCK_SIZE_X * BLOCK_SIZE_Y + 2 * CONVOLUTION_KERNEL_SIZE_5 + CONVOLUTION_KERNEL_SIZE_5 * CONVOLUTION_KERNEL_SIZE_5];
	__shared__ float s_kernel[CONVOLUTION_KERNEL_SIZE_5*CONVOLUTION_KERNEL_SIZE_5];


	//check that the threads are within the kernel mask region; if they are within, copy the
	//mask to the shared buffer.  In fact, this is not really necessary, but if you will use the 
	//kernel multiple times, it may make sense rather than repeatedly requesting the kernel data
	//to be fetched from the global memory.
	if (threadIdx.x < CONVOLUTION_KERNEL_SIZE_5 && threadIdx.y < CONVOLUTION_KERNEL_SIZE_5)
	{
		s_kernel[threadIdx.y*CONVOLUTION_KERNEL_SIZE_5 + threadIdx.x] = kernel[threadIdx.y*CONVOLUTION_KERNEL_SIZE_5 + threadIdx.x];
	}

	// ------------------- fetch the data into the shared memory --------------------------------------------------------------------
	//Contiguous threads fetch data from contiguous addresses in global memory (bufferIndex varies by 1 between 
	//contiguous threads) so the driver uses memory coalescing to reduce the number of transactions to copy the data from the
	///global memory.
	//First 2 feteches: block of image data copied to local memory; 2nd fetch is offset by the convolution kernel size 
	inBuffer[threadIdx.y*BLOCK_SIZE_X + threadIdx.x] = imageIn[bufferIndex];

	//Compute the convolution and write to the shared output buffer
	if ((c < (CONVOLUTION_KERNEL_SIZE_5 / 2)) || 
		(c > (imW - CONVOLUTION_KERNEL_SIZE_5 / 2)) || 
		(r < (CONVOLUTION_KERNEL_SIZE_5/2)) ||
		(r > (imH-CONVOLUTION_KERNEL_SIZE_5/2)))
	{
		//----Note: this is the outer boundary of the image.  The convolution kernel would be outside the image
		//----so we do not apply the convolution kernel and simply copy the input image straight to the output buffer.
		imageOut[bufferIndex] = imageIn[threadIdx.y*(BLOCK_SIZE_X + CONVOLUTION_KERNEL_SIZE_5)];
	}
	else
	{
		//Here, we are away from the convolution boundaries.  This is where we copy the block size of data from the global memory
		//(first line of the copy).  Note that the global memory index starts -1/2*convolution kernel size above the working data and
		//-1/2*convolution kernel size to the left of the working image data.  We then make another copy that is offset by the convolution
		//kernel size so that we fetch the image data (from global memory) that extends 1/2*convolution kernel size to the other
		//side of the global image data working area (to allow for the convolution boundaries).  This is an area which leads to redundenat
		//copies and could be improved (although, actually, using a texture to store the image would be beneficial, in this case
		//The 3rd and 4th copies are copying the lower convolution boundaries from the global image data set (and the lower/right boundary
		//region image data).
		float partialSum = 0.0f;
		inBuffer[threadIdx.y*(BLOCK_SIZE_X + CONVOLUTION_KERNEL_SIZE_5)+threadIdx.x] = imageIn[(blockIdx.y*blockDim.y + threadIdx.y-CONVOLUTION_KERNEL_SIZE_5/2)*imW + (blockIdx.x*blockDim.x + threadIdx.x-CONVOLUTION_KERNEL_SIZE_5/2)];
		inBuffer[threadIdx.y*(BLOCK_SIZE_X + CONVOLUTION_KERNEL_SIZE_5) + threadIdx.x+CONVOLUTION_KERNEL_SIZE_5] = imageIn[(blockIdx.y*blockDim.y + threadIdx.y-CONVOLUTION_KERNEL_SIZE_5/2)*imW + (blockIdx.x*blockDim.x + threadIdx.x + CONVOLUTION_KERNEL_SIZE_5/2)];
		inBuffer[(threadIdx.y + CONVOLUTION_KERNEL_SIZE_5)*(BLOCK_SIZE_X + CONVOLUTION_KERNEL_SIZE_5) + threadIdx.x] = imageIn[(blockIdx.y*blockDim.y + threadIdx.y+CONVOLUTION_KERNEL_SIZE_5/2)*imW + (blockIdx.x*blockDim.x + threadIdx.x-CONVOLUTION_KERNEL_SIZE_5/2)];
		inBuffer[(threadIdx.y + CONVOLUTION_KERNEL_SIZE_5)*(BLOCK_SIZE_X + CONVOLUTION_KERNEL_SIZE_5) + threadIdx.x + (3*CONVOLUTION_KERNEL_SIZE_5/2)] = imageIn[(blockIdx.y*blockDim.y + threadIdx.y + CONVOLUTION_KERNEL_SIZE_5 / 2)*imW + (blockIdx.x*blockDim.x + threadIdx.x + CONVOLUTION_KERNEL_SIZE_5 / 2)];

		//****** Convolution: multiply the block of data in the shared memory with the convolution kernel (in shared memory) and 
		//****** add up the partial sums
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
		//Write the output data to global memory
		imageOut[bufferIndex] = partialSum * scaleFactor;
		// ****************************************************************************
	}
	//In this case, not necessary, but could put a __synchthreads() to ensure all threads are 
	//writing at the same time (the reason it is not necessary is that there are no further reads 
	//from the global memory in this kernel function, so we do not need to ensure that the global memory
	//addresses have been fully written to before other threads start reading - this is a mechanism to
	//ensure all the threads read the latest data in global memory (so that they are not working with
	//stale data)
	__syncthreads();

}



///Kernel: g_filterImage_5x5
///Description: 5x5 Laplace of Gaussian filter - using global memory read/write
///Notes:  see above
__global__ void g_filterImage_5x5(float* imageIn, float* kernel, float* imageOut, float scaleFactor, int imW, int imH)
{
	///Thread block configuration section -----------------------------------------------------------------------------
	///Get the thread and block IDs for the current thread
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


///Description: Below, we have two functions that dispatch the kernel functions (the code that runs on the GPU cores/compute units)
///             We provide a wrapper for each kernel function with a 'C' interface.  The reason for this is that the kernel dispatch,
///             which uses CUDA-specific (or Open-CL specific, in the case of OpenCL) syntax is handled by the nvcc compiler so must
///             be included in a GPU specific translation unit (technical name for file to compile) - in this case denoted by the
///             *.cu file extension (*.cl if it is an OpenCL translation unit).  However, if we want to load it into a C or C++ 
///             translation unit (for compilation by a C or C++ compiler; cl or g++ or gcc), we need a 'C' interface call, which the
///             wrapper provides.  
///             In this file, we have two 'C' style functions for linking against a C/C++ project:
///                filterImage5x5_g:    dispatches the kernel function that reads and writes data directly to/from the global memory
///                filterImage5x5_s:    dispatches the kernel function that copies data onto the shared memory for processing
///
///            Kernel dispatch:
///                   g_filterImage_5x5 << <gridSize, blockSize >> > (imageIn, kernel, imageOut, scaleFactor, imW, imH);
///                   s_filterImage_5xt << <gridSize, blockSize >> > (imageIn, kernel, imageOut, scaleFactor, imW, imH);
///            The <<< ...>>> brackets are CUDA specific and define the configuration of the threads to be dispatched.  There are 
///            two parameters:  block_size     and     grid_size
///            The block_size parameter has 3 parameters: block_size.x, block_size.y, block_size.z   defining the number of threads in 
///            each of the 3 axes.
///            The grid_size parameter defines the number of blocks to process the complete data set.  The grid_size parameter also has
///            3 members:  
///                    grid_size.x, grid_size.y, grid_size.z
///            and is computed as data size (x-direction)/block_size.x,   data size (y-direction)/block_size.y, data size(z-direction)/block_size.z
///            
///            The parameters are the same values as those in the kernel function definition (see above)
///            Notes:  
///            (1)  the actual kernel function is defined by the keyword __global__ and the void return type
///            __global__  void   KernelFunction (....parameter list....)
///            (2)  the parameters which are pointers corresponding to buffers are the pointers defined by the memory allocation
///                 functions cudaMallocXXX( ) or clMemoryAllocXXX() [for OpenCL]
///         
///             In the case of OpenCL, there is a function called clEnqueueKernel() that leads to the kernel being executed
///             However, because OpenCL is a lower level language, closer to the driver, a number of operations have to be called
///             beforehand.
///             First, you need to create a command queue 
///                  clCreateCommandQueue() 
///             to which kernels get mapped (the scheduler then takes the command queue and dispatches the kernels in the queue)
///             In addition, the parameters to the kernel function have to be written to a stack
///                   clParameteri/v() 
///             which takes the parameter, the type, and the position in the stack (or in the argument list)
///             
///             The additional overhead of OpenCL is why the demonstration is in CUDA.  However, it is worth noting that OpenCL is 
///             supported by NVIDIA, AMD GPUs (although NVIDIA supports OpenCL 1.2 and AMD supports OpenCL 2.0) as well as Intel GPU
///             (Intel HD integrated graphics) and Intel CPUs and some FPGA vendors (Altera), so there is more portability with
///             OpenCL.
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