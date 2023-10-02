#include <optix.h>
#include "../lib/raydata.cuh"
#include "../lib/sampling.cuh"
#include "../shaders/FunctionIdx.h"
#include "../shaders/sysparameter.h"
#include "../pdf/pdf.cuh"

extern "C" __constant__ SysParamter Parameter;
// Ray state variables
//rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
//rtDeclareVariable(PerRayData, thePrd, rtPayload,  );
//
//// "Global" variables
//rtDeclareVariable(rtObject, sysWorld, , );
//
//// The point and normal of intersection and UV parms
//rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );
//
///*! and finally - that particular material's parameters */
//rtDeclareVariable(rtCallableProgramId<float3(float, float, float3)>, sampleTexture, , );
//
//
/*! the actual scatter function - in Pete's reference code, that's a
  virtual function, but since we have a different function per program
  we do not need this here */
// inline __device__ bool scatter(const optix::Ray &ray_in,
//                                DRand48 &rndState,
//                                vec3f &scattered_origin,
//                                vec3f &scattered_direction,
//                                vec3f &attenuation) {
//   return false;
// }

//inline __device__ float3 emitted() {
//
//    if (dot(hitRecord.normal, theRay.direction) < 0.f) {
//        return sampleTexture(hitRecord.u, hitRecord.v, hitRecord.point);
//    }
//    else {
//        return make_float3(0.f);
//    }
//}

inline __device__ float scatteringPdf() {
    return false;
}

extern "C" __device__ void __direct_callable__diffuselight(MaterialParams const& matParam,
    PerRayData &thePrd, float3 direction, HitRecord const& hitRecord) 
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    hitRectData rec = rt_data->rectData;
    unsigned int idx = matParam.texArr->texCallidx + NUM_CALLABLE_CAMERA + NUM_CALLABLE_MAT_IDS;
    if (dot(hitRecord.normal, direction) < 0.f) {
        thePrd.emitted = optixDirectCall<float3, MaterialParams const&, float, float, float3 >(
            idx, matParam,
            hitRecord.u, hitRecord.v, hitRecord.point);
    }
    else
    {
        thePrd.emitted = make_float3(0.f);
    }
    thePrd.is_specular = false;
    thePrd.materialType = DiffuseLight;
    thePrd.hit_normal = hitRecord.normal;
    thePrd.scatterEvent = Ray_Cancel;
    thePrd.scattered_pdf = scatteringPdf();
    thePrd.radiance = thePrd.emitted;
}
