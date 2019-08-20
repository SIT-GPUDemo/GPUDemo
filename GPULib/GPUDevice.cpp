#include "GPUDevice.h"
#include "GPU.h"

DLLEXPORT IGPU* __cdecl CreateGPU()
{
	return new GPU();
}