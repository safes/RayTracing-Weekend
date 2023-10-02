
#include "texture.cuh"
#include "sutil/vec_math.h"
#include "../shaders/sysparameter.h"

//rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, odd, , );
//rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, even, , );
extern "C" __constant__ SysParamter Parameter;
extern "C" __device__ float3 __direct_callable__checkerTexture(MaterialParams const& matPar, float u, float v, float3 p) {
    float sines = sinf(10.f * p.x) * sinf(10.f - p.y) * sinf(10.f * p.z);
  //  MaterialParams const& matPar = Parameter.MatParams[matIdx];

    textureParam* param = (textureParam*)matPar.texArr;
    if (sines < 0)
       return optixDirectCall<float3, MaterialParams const&, float, float, float3 >(param->odd,matPar, u, v, p);
        //return odd(u, v, p);
    else
        return optixDirectCall<float3, MaterialParams const&, float, float, float3 >(param->even,matPar, u, v, p);
      //  return even(u, v, p);
}
