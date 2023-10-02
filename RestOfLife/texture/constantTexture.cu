#include "texture.cuh"
#include "../shaders/sysparameter.h"
//rtDeclareVariable(float3, color, , );
extern "C" __constant__ SysParamter Parameter;
extern "C" __device__ float3 __direct_callable__constantTexture(MaterialParams const& matPar, float u, float v, float3 p) {
 //   MaterialParams matPar = Parameter.MatParams[matIdx];
    
    return matPar.texArr->color;
    
}
