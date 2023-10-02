#include "texture.cuh"
#include <texture_indirect_functions.h>
#include "sutil/vec_math.h"
#include "../shaders/sysparameter.h"
//rtDeclareVariable(int, nx, , );
//rtDeclareVariable(int, ny, , );
//rtDeclareVariable(int, nn, , );
//rtTextureSampler<float4, 2> data;
extern "C" __constant__ SysParamter Parameter;
extern "C" __device__ float3 __direct_callable__imagetexture(MaterialParams const& matPar, float u, float v, float3 p)
{
 //   MaterialParams const& matPar = Parameter.MatParams[matIdx];
    textureParam* texPar = (textureParam*)matPar.texArr;
        
    return make_float3(tex2D<float4>(texPar->imagetex, u, v));
}
