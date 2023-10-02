
#include "texture.cuh"
#include "sutil/vec_math.h"
#include "../shaders/sysparameter.h"
//rtDeclareVariable(float3, blank, , );
extern "C" __constant__ SysParamter Parameter;
extern "C" __device__ float3 __direct_callable__nullTexture(MaterialParams const& matPar, float u, float v, float3 p) {
    
   //textureParam* texPar = (textureParam*)optixGetSbtDataPointer();

    return make_float3(0.f,0.f,0.f);
}
