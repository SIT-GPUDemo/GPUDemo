#pragma once
// ------------------------------------------------------------------------------------------------------------------ //
// GPUTypes.h                                                                                                         //
// ------------------------------------------------------------------------------------------------------------------ //
// Definition of data types, constants, enumerations used by the GPU class                                            //
// ------------------------------------------------------------------------------------------------------------------ //
#include<cstdlib>
#include<cstdio>
#include<cstdint>

enum class MEMORY_TYPE:int8_t
{
	MEMORY_MANAGED = 1,
	MEMORY_HOST_LOCKED = 2,
	MEMORY_DEVICE = 3,
	MEMORY_DEVICE_ALIGNED = 4
};

enum class DATA_TYPE :int8_t
{
	UINT8 = 1,
	INT8 = 2,
	FLOAT32 = 3
};


enum class WRITE_PATH :int8_t
{
	WRITE_HOST_2_DEV = 1,
	WRITE_DEV_2_HOST = 2,
	WRITE_DEV_2_DEV = 3,
	WRITE_HOST_2_HOST = 4
};

enum class KERNEL_FUNCTION_ID :uint8_t
{
	KERNEL_FILTER_3x3 = 0,
	KERNEL_FILTER_5x5 = 1,
	KERNEL_FILTER_7x7 = 2
};