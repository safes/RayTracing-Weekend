#include <optix.h>

#include "../shaders/sysparameter.h"
#include "sutil/vec_math.h"
#include "../lib/raydata.cuh"

// Ray state variables
//rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
//rtDeclareVariable(PerRayData, thePrd, rtPayload,  );
//
//// "Global" variables
//rtDeclareVariable(rtObject, sysWorld, , );
//
//// The point and normal of intersection
//rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );

inline __device__ float3 emitted(){
    return make_float3(0.f, 0.f, 0.f);
}

extern "C" __device__ void __direct_callable__normal(MaterialParams const& matParam,
    PerRayData & thePrd, float3 direction, HitRecord const& hitRecord)
{
    thePrd.emitted = emitted();
    thePrd.is_specular = true;
    thePrd.materialType = Normalic;

    thePrd.hit_normal = hitRecord.normal;
    thePrd.scatterEvent = Ray_Finish;
    thePrd.attenuation = 0.5f * (hitRecord.normal + make_float3(1.0f, 1.0f, 1.0f));
}
