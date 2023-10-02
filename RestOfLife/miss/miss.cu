#include <optix.h>
#include <sutil/vec_math.h>

#include "../lib/raydata.cuh"
#include "../shaders/sysparameter.h"
extern "C" __constant__ SysParamter Parameter;
//rtDeclareVariable(PerRayData, thePrd, rtPayload, );
inline __device__ float3 missColor(const PerRayData& theRay)
{
    if (Parameter.skyLight) {
        float3 unitDirection = normalize(theRay.scattered_direction);
        float t = 0.5f * (unitDirection.y + 1.0f);
        // "sky" gradient
        float3 missColor = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f)
            + t * make_float3(0.5f, 0.7f, 1.0f);
        return missColor;
    }
    else {
        return make_float3(0.0f); // darkness in the void
    }
}

extern "C" __global__ void __miss__Program()
{
    PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
    thePrd->emitted = make_float3(0.f);
    thePrd->radiance = missColor(*thePrd);
    
    thePrd->scatterEvent = Ray_Miss;
}
