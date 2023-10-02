
#include <optix.h>
#include "sutil/vec_math.h"

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
//rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, sampleTexture, , );

inline __device__ float3 emitted() {
    return make_float3(0.f, 0.f, 0.f);
}

inline __device__ float scatteringPdf() {
  return false;
}

extern "C" __device__ void __direct_callable__ISOTROPIC(MaterialParams const& matParam,
    PerRayData& thePrd, float3 direction, HitRecord const& hitRecord)
 {
    float3 scatterDirection = randomInUnitSphere(thePrd.seed);

    
    thePrd.is_specular = true; // ???
    thePrd.materialType = Isotropic;

    thePrd.scatterEvent = Ray_Hit;
    thePrd.hit_normal = hitRecord.normal;
    thePrd.scattered_origin = hitRecord.point;
    thePrd.scattered_direction = scatterDirection;
    thePrd.emitted = emitted();
    int callidx = matParam.texArr[0].texCallidx + NUM_CALLABLE_CAMERA +
        NUM_CALLABLE_MAT_IDS;
    thePrd.attenuation = optixDirectCall<float3, MaterialParams const&, float, float, float3 >(
        callidx, matParam, hitRecord.u,
        hitRecord.v, hitRecord.point);
  //  thePrd.attenuation = sampleTexture(hitRecord.u, hitRecord.v, hitRecord.point);
    thePrd.scattered_pdf = scatteringPdf();
}
