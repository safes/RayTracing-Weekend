#include <optix.h>
#include "../lib/raydata.cuh"
// The shadow ray program for all materials with no cutout opacity.
extern "C" __global__ void __anyhit__shadow()
{
	PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

	thePrd->flags |= FLAG_SHADOW; // Visbility check failed.

	optixTerminateRay();
}

