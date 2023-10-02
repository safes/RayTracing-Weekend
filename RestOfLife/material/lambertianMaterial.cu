#include <optix.h>

#include "../lib/raydata.cuh"
#include "../lib/sampling.cuh"
#include "sutil/vec_math.h"
#include "../shaders/sysparameter.h"
#include "../shaders/FunctionIdx.h"
#include "../lib/onb.cuh"
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
//
//// Texture program
//rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, sampleTexture, , );
//
inline __device__ float3 emitted(){
    return make_float3(0.f, 0.f, 0.f);
}


inline __device__ float scatteringPdf(HitRecord const& hitRecord, PerRayData const& thePrd) {
    float cosine = dot(hitRecord.normal, normalize(thePrd.scattered_direction));

    if(cosine < 0.f) {
        //cosine = 0.f;
        return 0.f;
    } else {
        return cosine / CUDART_PI_F;
    }
}


extern "C" __device__ void __direct_callable__lambertian(MaterialParams const& matParam,
    PerRayData &thePrd, float3 direction, HitRecord const& hitRecord)
    {
    onb uvw;
    uvw.buildFromW(hitRecord.normal);
    float3 w = randomCosineDirection(thePrd.seed);
    thePrd.pdf = w.z * M_1_PIf;
    float3 scatterDirection = normalize(uvw.local(w));

    thePrd.emitted = emitted();
    thePrd.is_specular = false;
    thePrd.materialType = Lambertian;

    thePrd.scatterEvent = Ray_Hit;
    thePrd.hit_normal = hitRecord.normal;
    thePrd.scattered_origin = hitRecord.point;
    thePrd.scattered_direction = scatterDirection;
    thePrd.scattered_pdf = scatteringPdf(hitRecord,thePrd );
    if (thePrd.scattered_pdf <=0 || thePrd.pdf <=0 )
    {
        thePrd.scatterEvent = Ray_Cancel;
        return;
    }
    
    unsigned int idx = matParam.texArr->texCallidx + NUM_CALLABLE_CAMERA + NUM_CALLABLE_MAT_IDS;
    thePrd.attenuation = optixDirectCall<float3, MaterialParams const& , float , float , float3 >(
        idx, matParam, hitRecord.u, hitRecord.v, hitRecord.point);
    
    //thePrd.pdf = dot(uvw.w, scatterDirection) / CUDART_PI_F;

}


extern "C" __device__ float4 __direct_callable__eval_bsdf_diffuse_reflection(
    PerRayData & thePrd, const float3 wiL)
{
    const float3 f = thePrd.attenuation * M_1_PIf;
    const float  pdf = fmaxf(0.0f, dot(wiL, thePrd.hit_normal) * M_1_PIf);

    return make_float4(f, pdf);
}
