
#include <optix.h>
#include <vector_types.h>
// defines CUDART_PI_F
#include "math_constants.h"
#include "../lib/sampling.cuh"
#include "sysparameter.h"

extern "C" __constant__ SysParamter Parameter;

extern "C" __device__ void __direct_callable__perspectiveCamera(const float s, const float t, unsigned int& seed,
    float3& origin, float3& direction)
{
    
    const float3 rd = Parameter.cameraLensRadius * random_in_unit_disk(seed);
    const float3 lens_offset = Parameter.cameraU * rd.x + Parameter.cameraV * rd.y;
    origin = Parameter.cameraOrigin + lens_offset;
    direction = Parameter.cameraLowerLeftCorner + (s * Parameter.cameraHorizontal) + (t * Parameter.cameraVertical) - origin;
}


