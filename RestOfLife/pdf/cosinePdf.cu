
#include "pdf.cuh"
#include "../lib/onb.cuh"
#include "../shaders/sysparameter.h"
#include "../lib/sampling.cuh"
extern "C" __constant__ SysParamter Parameter;

extern "C" __device__ LightSample __direct_callable__cosineGenerate(LightDefinition const& light,
    pdf_in &in, uint32_t& seed) {
    float3 temp = randomCosineDirection(seed);
    LightSample lightSample;
    //in.light_direction = in.uvw.local(temp);
    return lightSample;
}

extern "C" __device__ float __direct_callable__cosineValue(pdf_in &in) {
    float cosine = dot(normalize(in.direction), in.uvw.w);
    if(cosine > 0.f)
        return cosine / CUDART_PI_F;
    else
        return 0.0000001f;
}
