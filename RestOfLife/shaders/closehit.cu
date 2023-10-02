#include "appconfig.h"
#include <optix.h>

#include <vector_types.h>
#include <optix_types.h>
#include <sutil/vec_math.h>
#include "../lib/raydata.cuh"
#include "sysparameter.h"
#include "FunctionIdx.h"
#include "../pdf/pdf.cuh"

#include <cuda/random.h>
extern "C" __constant__ SysParamter Parameter;

// Returns true if ray is occluded, else false
static __forceinline__ __device__ bool traceOcclusion(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax
)
{
    unsigned int p0 = optixGetPayload_0();
    unsigned int p1 = optixGetPayload_1();
    // We are only casting probe rays so no shader invocation is needed
    optixTraverse(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax, 0.0f,                // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT /*| OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT*/,
        0,                         // SBT offset
        RAY_TYPE_COUNT,            // SBT stride
        0,                          // missSBTIndex
        p0,
        p1
    );
    return optixHitObjectIsHit();
}


extern "C" __global__ void __closesthit__radiance()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
    //thePrd->emitted = rt_data->color;
    const unsigned int thePrimitiveIndex = optixGetInstanceId();

    unsigned int n0 = optixGetAttribute_0(), n1 = optixGetAttribute_1(), n2 = optixGetAttribute_2();
    unsigned int u = optixGetAttribute_3(), v = optixGetAttribute_4();
    unsigned int p0 = optixGetAttribute_5(), p1 = optixGetAttribute_6(), p2 = optixGetAttribute_7();
    HitRecord hit = {};
    hit.normal = make_float3(__int_as_float(n0), __int_as_float(n1), __int_as_float(n2));
    hit.point = make_float3(__int_as_float(p0), __int_as_float(p1), __int_as_float(p2));
    hit.u = __int_as_float(u), hit.v = __int_as_float(v);
    float3 direction = optixGetWorldRayDirection();
    thePrd->distance = optixGetRayTmax();
    //thePrd->attenuation = make_float3(0.f);
    //thePrd->pdf = 0.f;
    MaterialParams const& parameters = Parameter.MatParams[thePrimitiveIndex];
    unsigned int callidx = parameters.matindex + NUM_CALLABLE_CAMERA;
    thePrd->radiance = make_float3(0.f);
    optixDirectCall<void, MaterialParams const&, PerRayData&, float3, HitRecord const&>(
        callidx, parameters, *thePrd, direction, hit);
    const int numLights = Parameter.numLights;

    if (thePrd->scatterEvent != Ray_Miss && thePrd->scatterEvent != Ray_Finish
        && thePrd->scatterEvent != Ray_Cancel && !thePrd->is_specular && numLights > 0 )
    {
        
        pdf_in in(thePrd->scattered_origin, thePrd->scattered_direction, thePrd->hit_normal);
        int pdfgenidx = Parameter.pdf.pdfGenIdx + NUM_CALLABLE_CAMERA +
            NUM_CALLABE_TEX_IDS + NUM_CALLABLE_MAT_IDS;
      /*  int pdfvalidx = Parameter.pdf.pdfValIdx + NUM_CALLABLE_CAMERA +
            NUM_CALLABE_TEX_IDS + NUM_CALLABLE_MAT_IDS;*/
        
        const int indexLight = (1 < numLights) ? clamp(static_cast<int>(floorf(rnd(thePrd->seed) * numLights)), 0, numLights - 1) : 0;
        
        LightSample lightsample = optixDirectCall<LightSample, LightDefinition const&,
            hitRectData const&, pdf_in&, uint32_t&>(pdfgenidx, Parameter.lights[indexLight],
                Parameter.pdf.pdfrect, in, thePrd->seed);

     //   float pdf_val = optixDirectCall<float, pdf_in&>(pdfvalidx, in);
        //thePrd->lightDirection = pdf_direction;
         
        if ( lightsample.pdf > 0 )
        {
            int reflect_callIdx = parameters.lightreflectIdx + +NUM_CALLABLE_CAMERA +
                NUM_CALLABE_TEX_IDS + NUM_CALLABLE_MAT_IDS;
           float4 reflectBsdf = optixDirectCall< float4, PerRayData&, const float3 > (reflect_callIdx,
                *thePrd, lightsample.direction);
           if (0.0f < reflectBsdf.w && isNotNull(make_float3(reflectBsdf)))
           {
               const bool occluded = traceOcclusion(
                   Parameter.handle, thePrd->scattered_origin,
                   lightsample.direction,
                   500*1.0e-7f, lightsample.distance - 500 * 1.0e-7f
               );

               if ( !occluded ) {
                   if (thePrd->materialType == Dielectric) // Supporting nested materials includes having lights inside a volume.
                   {
                       // Calculate the transmittance along the light sample's distance in case it's inside a volume.
                       // The light must be in the same volume or it would have been shadowed!
                       lightsample.emission *= expf(-lightsample.distance * thePrd->attenuation);
                   }

                  const float weight = powerHeuristic(lightsample.pdf, reflectBsdf.w);
                   thePrd->radiance += make_float3(reflectBsdf) * lightsample.emission *
                   (weight * dot(lightsample.direction, thePrd->hit_normal) / lightsample.pdf);
               }
                              
           }
        }
    }
    

}