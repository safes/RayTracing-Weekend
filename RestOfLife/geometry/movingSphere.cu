#include <optix.h>

#include "sutil/vec_math.h"
#include "../lib/raydata.cuh"
#include "../shaders/sysparameter.h"

// Sphere variables
//rtDeclareVariable(float3, center0, , );
//rtDeclareVariable(float3, center1, , );
//rtDeclareVariable(float, radius, , );
//rtDeclareVariable(float, time0, , );
//rtDeclareVariable(float, time1, , );

// The ray that will be intersected against
//rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
//rtDeclareVariable(PerRayData, thePrd, rtPayload,  );
//
// The point and normal of intersection. and uv-space location
//   the "attribute" qualifier is used to communicate between intersection and shading programs
//   These may only be written between rtPotentialIntersection and rtReportIntersection
//rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );
extern "C" __constant__ SysParamter Parameter;
inline __device__ float2 get_sphere_uv(const float3 p) {
    float phi = atan2f(p.z, p.x);
    float theta = asinf(p.y);
    float2 uv;
    uv.x = 1 - (phi + CUDART_PI_F) / (2.f * CUDART_PI_F);
    uv.y = (theta + CUDART_PIO2_F) / CUDART_PI_F;
    return uv;

}

__device__ float3 center(float time, HitGroupData* hitRec) {
    if (hitRec->time0  == hitRec->time1)
        return hitRec->center;
    else
        return hitRec->center + ((time - hitRec->time0) / (hitRec->time1 - hitRec->time0)) 
        * (hitRec->center1 - hitRec->center);
}

// The sphere bounding box program
//   The pid parameter enables specifying a primitive withing this geometry
//   since there is only 1 primitive (the sphere), the pid is ignored here
//extern "C" __global__  void getBounds(int pid, float result[6])
//{
//    optix::Aabb* box0 = (optix::Aabb*)result;
//    box0->m_min = center(time0) - abs(radius);
//    box0->m_max = center(time0) + abs(radius);
//
//    optix::Aabb box1;
//    box1.m_min = center(time1) - abs(radius);
//    box1.m_max = center(time1) + abs(radius);
//
//    box0->include(box1);
//}


// The sphere intersection program
//   this function calls rtReportIntersection if an intersection occurs
//   As above, pid refers to a specific primitive, is ignored
extern "C" __global__ void __intersection__movingsphere()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    float3 direction = optixGetObjectRayDirection();
    PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
    float3 xcenter = center(thePrd->gatherTime, rt_data);
    float3 oc = optixGetObjectRayOrigin() - xcenter;
    float a = dot(direction, direction);
    float b = dot(oc, direction);
    float c = dot(oc, oc) - rt_data->radius* rt_data->radius;
    float discriminant = b*b - a*c;

    if (discriminant < 0.0f) return;

    float t = (-b - sqrtf(discriminant)) / a;
    if (t < optixGetRayTmax() && t > optixGetRayTmin())
        //if (rtPotentialIntersection(t))
        {
           // hitRecord.point = rtTransformPoint(RT_OBJECT_TO_WORLD,  theRay.origin + t*theRay.direction);
           // hitRecord.normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD,
           //                                                     (hitRecord.point - center(thePrd.gatherTime))/radius));
            float3 point = optixTransformPointFromObjectToWorldSpace(optixGetObjectRayOrigin() + t * direction);
            float3 shading_normal = optixTransformNormalFromObjectToWorldSpace(
                (point - xcenter) / rt_data->radius);


            float2 uv = get_sphere_uv(shading_normal);
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
                t,      // t hit
                0,          // user hit kind
                p0, p1, p2, u, v,n0,n1,n2);
        }

    t = (-b + sqrtf(discriminant)) / a;
    if (t < optixGetRayTmax() && t > optixGetRayTmin())
    {
        float3 point = optixTransformPointFromObjectToWorldSpace(
            optixGetObjectRayOrigin() + t * direction);
        // hitRecord.point = rtTransformPoint(RT_OBJECT_TO_WORLD,  theRay.origin + t*theRay.direction);
        // hitRecord.normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, (hitRecord.point - center)/radius));
        float3 shading_normal = optixTransformNormalFromObjectToWorldSpace(
            (point - xcenter) / rt_data->radius);


        float2 uv = get_sphere_uv(shading_normal);

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
            t,      // t hit
            0,          // user hit kind
            p0, p1, p2, u, v, n0, n1, n2
        );
    }
}
