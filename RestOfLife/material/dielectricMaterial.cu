#include <optix.h>
#include "../shaders/sysparameter.h"
#include "sutil/vec_math.h"
#include "../lib/raydata.cuh"
#include "../lib/sampling.cuh"

//// Ray state variables
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
//rtDeclareVariable(float, eta, , );

inline __device__ float fresnelSchlick(
    const float cosThetaI, const float etaI, const float etaT)
{
    float r0 = (etaI-etaT) / (etaI+etaT);
    r0 = r0*r0;
    return r0 + (1.f-r0)*powf((1.f-cosThetaI), 5.f);
}

inline __device__ float3 emitted() {
    return make_float3(0.f, 0.f, 0.f);
}

inline __device__ float scatteringPdf(){

  return false;
}

extern "C" __device__ void __direct_callable__dielectricMat(MaterialParams const& matParam,
    PerRayData& thePrd, float3 direction , HitRecord const& hitRecord)

{
    // Get the ray's unit direction
    float3 unitDirection = normalize(-thePrd.lightDirection);

    // Determine if inside or outside of object
    float3 localNormal;
    float etaI, etaT;
    if (dot(direction, hitRecord.normal) < 0.0f)
    {
        // Outside the object
        localNormal = hitRecord.normal;
        etaI = 1.0f;
        etaT = matParam.eta;
    }
    else
    {
        // Inside the object
        localNormal = -hitRecord.normal;
        etaI = matParam.eta;
        etaT = 1.0f;
    }

    // Snell's Law
    //  etaI * sinThetaI = etaT * sinThetaT
    // If
    //  (etaI/etaT) * sinThetaI > 1.0
    // Then
    //  sinThetaT does not exist, and no transmission is possible
    float3 scatterDirection;
    float cosThetaI = min(dot(-unitDirection, localNormal), 1.0f);
    float sinThetaI = sqrtf(1.0f - cosThetaI*cosThetaI);
    if ( ((etaI/etaT)*sinThetaI) > 1.0f )
    {
        // No Transmission is possible
        scatterDirection = reflect(unitDirection, localNormal);
    }
    else
    {
        // Transmission + Reflection
        float reflectProb = fresnelSchlick(cosThetaI, etaI, etaT);
        if (rnd(thePrd.seed) < reflectProb)
        { 
            // Reflection
            scatterDirection = reflect(unitDirection, localNormal);
        }
        else
        { 
            // Transmission
            float sinThetaT = min((etaI/etaT)*sinThetaI, 1.0f);
            float cosThetaT = sqrtf(1.0f - sinThetaT*sinThetaT);
            scatterDirection =
                (etaI/etaT)*(unitDirection + cosThetaI*localNormal) -
                cosThetaT*localNormal;

        }
    }

    // if(cosThetaI > 1.0f)
    //     printf("costThetaI is greater than unity: %f", cosThetaI);
    // if(!(sinThetaI == sinThetaI))
    //     printf("sinThetaI is NaN: %f", sinThetaI);

    thePrd.emitted = emitted();
    thePrd.is_specular = true;
    thePrd.materialType = Dielectric;
    thePrd.scatterEvent = Ray_Hit;
    thePrd.scattered_origin = hitRecord.point;
    thePrd.scattered_direction = scatterDirection;
    // thePrd.attenuation = make_float3(0.8f, 0.85f, 0.82f); // for greenish glass
    thePrd.attenuation = make_float3(1.0f, 1.0f, 1.0f);
    thePrd.scattered_pdf = scatteringPdf();
    thePrd.hit_normal = hitRecord.normal;
    thePrd.pdf = 1.f;

}


extern "C" __device__ float4 __direct_callable__eval_bsdf_dielect_reflection(MaterialParams const& matParam,
    PerRayData & thePrd, const float3 wiL)
{
    float3 f = thePrd.attenuation;
    const float  pdf = 1.f;
    return make_float4(f,pdf);
}
