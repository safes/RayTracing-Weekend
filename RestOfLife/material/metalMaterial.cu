
#include <optix.h>

#include "../lib/raydata.cuh"
#include "../lib/sampling.cuh"
#include "../shaders/sysparameter.h"
#include "../shaders/FunctionIdx.h"
// Ray state variables
//rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
//rtDeclareVariable(PerRayData, thePrd, rtPayload,  );
//
//// "Global" variables
//rtDeclareVariable(rtObject, sysWorld, , );
//
//// The point and normal of intersection
//rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );
//
//// Material variables
//rtDeclareVariable(float, fuzz, , );
//
//rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, sampleTexture, , );
//

inline __device__ float3 emitted(){
    return make_float3(0.f, 0.f, 0.f);
}

inline __device__ float scatteringPdf() {
  return false;
}

extern "C" __device__ void __direct_callable__metal(MaterialParams const& matParam,
    PerRayData& thePrd, float3 direction, HitRecord const& hitRecord)
{
    thePrd.emitted = emitted();
    thePrd.is_specular = true;
    thePrd.materialType = Metallic;

    thePrd.hit_normal = hitRecord.normal;

    // optix::reflect expects normalized (unit vector) inputs
    float3 reflected = reflect(direction, hitRecord.normal);
    float3 scatterDirection = reflected +matParam.fuzz * randomInUnitSphere(thePrd.seed);

    thePrd.scattered_origin = hitRecord.point;
    thePrd.scattered_direction = normalize(scatterDirection);
   
    thePrd.scattered_pdf = scatteringPdf();

    int callidx = matParam.texArr[0].texCallidx + NUM_CALLABLE_CAMERA +
        NUM_CALLABLE_MAT_IDS;
    thePrd.attenuation = optixDirectCall<float3, MaterialParams const&, float, float, float3 >(
        callidx, matParam,hitRecord.u,
        hitRecord.v, hitRecord.point);
    
    if (dot(thePrd.scattered_direction, hitRecord.normal) <= 0.0f ) {
        thePrd.scatterEvent = Ray_Cancel;
        return;
    }

    thePrd.pdf = 1.f;
    thePrd.scatterEvent = Ray_Hit;

}

extern "C" __device__ float4 __direct_callable__eval_bsdf_metal_reflection(MaterialParams const& matParam,
    PerRayData & thePrd, const float3 wiL)
{
    float3 f = thePrd.attenuation;
    const float  pdf = 1.f;//fmaxf(0.0f, dot(wiL, thePrd.hit_normal) * M_1_PIf);
    return make_float4(f,pdf);
}
