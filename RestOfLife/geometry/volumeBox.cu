#include <optix.h>


// defines FLT_
#include <cfloat>

#include "../lib/raydata.cuh"
#include "../lib/random.cuh"
#include "../shaders/sysparameter.h"
#include "sutil/vec_math.h"
/*! the parameters that describe each the box */
//rtDeclareVariable(float3, boxMin, , );
//rtDeclareVariable(float3, boxMax, , );
//rtDeclareVariable(float,  density, , );
//
//// The ray that will be intersected against
//rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
//rtDeclareVariable(PerRayData, thePrd, rtPayload,  );
//
//rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );

extern "C" __constant__ SysParamter Parameter;

constexpr int EPSILON = (0.0001f);

// Programs that performs the ray-box intersection
//

inline __device__ bool hit_boundary(const float3 direction, const float3 origin, 
     hitVolumeBoxdata const& hitBox, const float tMin, const float tMax, float &rec) {
    float3 t0 = (hitBox.boxMin - origin) / direction;
    float3 t1 = (hitBox.boxMax - origin) / direction;
    float temp1 = fmaxf(fminf(t0, t1));
    float temp2 = fminf(fmaxf(t0, t1));

    if(temp1 > temp2)
        return false;

    // if the first root was a hit,
    if (temp1 < tMax && temp1 > tMin) {
        rec = temp1;
        return true;
    }

    // if the second root was a hit,
    if (temp2 < tMax && temp2 > tMin){
        rec = temp2;
        return true;
    }

    return false;
}


extern "C" __global__ void __intersection__hitVolumebox() {
    float hitt1, hitt2;
    PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    hitVolumeBoxdata hitbox = rt_data->VolBoxData;
    float3 direction = optixGetObjectRayDirection();
    float3 origin = optixGetObjectRayOrigin();
    if(hit_boundary(direction,origin,hitbox,-FLT_MAX, FLT_MAX, hitt1)) {
        if(hit_boundary(direction, origin, hitbox, hitt1 + EPSILON, FLT_MAX, hitt2)){
            if(hitt1 < optixGetRayTmin())
                hitt1 = optixGetRayTmin();

            if(hitt2 > optixGetRayTmax())
                hitt2 = optixGetRayTmax();

            if(hitt1 >= hitt2)
                return;

            if(hitt1 < 0.f)
                hitt1 = 0.f;

            float distanceInsideBoundary = hitt2 - hitt1;
            distanceInsideBoundary *= length(direction);

            float hitDistance = -(1.f /hitbox. density) * logf(randf(thePrd->seed));
            float hitt = hitt1 + (hitDistance / length(direction));

            //if (rtPotentialIntersection(hitt)) 
            {
                float3 point = optixTransformPointFromObjectToWorldSpace(
                    origin + hitt*direction);

                float2 uv = make_float2(0.f);
               /* hitRecord.v = 0.f;*/

                float3 normal = make_float3(1.f, 0.f, 0.f);
                float3 shading_normal = normalize(optixTransformNormalFromObjectToWorldSpace(
                     normal));

                unsigned int p0, p1, p2;
                p0 = __float_as_uint(shading_normal.x);
                p1 = __float_as_uint(shading_normal.y);
                p2 = __float_as_uint(shading_normal.z);
                unsigned int u, v;
                u = __float_as_uint(uv.x);
                v = __float_as_uint(uv.y);
                unsigned int n0, n1, n2;
                n0 = __float_as_uint(point.x);
                n1 = __float_as_uint(point.y);
                n2 = __float_as_uint(point.z);
                optixReportIntersection(
                    hitt,      // t hit
                    0,          // user hit kind
                    p0, p1, p2, u, v, n0, n1, n2
                );
            }
        }
    }
}


/*! returns the bounding box of the pid'th primitive
  in this geometry. Since we handle multiple boxes by having a different
  geometry per box, the'pid' parameter is ignored */
//RT_PROGRAM void getBounds(int pid, float result[6]) {
//    optix::Aabb* aabb = (optix::Aabb*)result;
//    // rtPrintf("boxMin(%f,%f,%f)\n", boxMin.x, boxMin.y, boxMin.z);
//    // rtPrintf("boxMax(%f,%f,%f)\n", boxMax.x, boxMax.y, boxMax.z);
//    // NOTE: assume all components of boxMin are less than  boxMax
//    aabb->m_min = boxMin - make_float3(EPSILON);
//    aabb->m_max = boxMax + make_float3(EPSILON);
//}
