#include <optix.h>
 
#include "sutil/vec_math.h"
#include "../lib/raydata.cuh"
#include "../shaders/sysparameter.h"

// Sphere variables
//rtDeclareVariable(float3, center, , );
//rtDeclareVariable(float, radius, , );
//
//// The ray that will be intersected against
//rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
//rtDeclareVariable(PerRayData, thePrd, rtPayload,  );
//
//// The point and normal of intersection. and uv-space location
////   the "attribute" qualifier is used to communicate between intersection and shading programs
////   These may only be written between rtPotentialIntersection and rtReportIntersection
//rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );
//
// assume p is normalized direction vector

extern "C" __constant__ SysParamter Parameter;

inline __device__ float2 get_sphere_uv(const float3 p) {
	float phi = atan2f(p.z, p.x);
	float theta = asinf(p.y);
    float2 uv;
	uv.x  = 1 - (phi + CUDART_PI_F) / (2.f * CUDART_PI_F);
	uv.y  = (theta + CUDART_PIO2_F) / CUDART_PI_F;
    return uv;

}

// The sphere bounding box program
//   The pid parameter enables specifying a primitive withing this geometry
//   since there is only 1 primitive (the sphere), the pid is ignored here
//RT_PROGRAM void getBounds(int pid, float result[6])
//{
//    optix::Aabb* aabb = (optix::Aabb*)result;
//    aabb->m_min = center - abs(radius);
//    aabb->m_max = center + abs(radius);
//}


// The sphere intersection program
//   this function calls rtReportIntersection if an intersection occurs
//   As above, pid refers to a specific primitive, is ignored
extern "C" __global__ void __intersection__sphere()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    float3 direction = optixGetObjectRayDirection();
    float3 oc = optixGetObjectRayOrigin() - rt_data->center;
    float a = dot(direction, direction);
    float b = dot(oc, direction);
    float c = dot(oc, oc) - rt_data->radius*rt_data->radius;
    float discriminant = b*b - a*c;

    if (discriminant < 0.0f) return;
   
    float t = (-b - sqrtf(discriminant)) / a;
    if (t < optixGetRayTmax() && t > optixGetRayTmin())
        {
           float3 point = optixTransformPointFromObjectToWorldSpace(optixGetObjectRayOrigin() + t * direction);
          //  hitRecord.point = rtTransformPoint(RT_OBJECT_TO_WORLD,  theRay.origin + t*theRay.direction);
         //   hitRecord.normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, (hitRecord.point - center)/radius));
           float3 shading_normal =  optixTransformNormalFromObjectToWorldSpace(
               (point - rt_data->center) / rt_data->radius);
            
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
               p0, p1, p2,u,v,n0,n1,n2
           );
        }

    t = (-b + sqrtf(discriminant)) / a;

    if(t < optixGetRayTmax() && t > optixGetRayTmin())
        {
            float3 point = optixTransformPointFromObjectToWorldSpace(
                optixGetObjectRayOrigin() + t * direction);
           // hitRecord.point = rtTransformPoint(RT_OBJECT_TO_WORLD,  theRay.origin + t*theRay.direction);
           // hitRecord.normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, (hitRecord.point - center)/radius));
            float3 shading_normal = optixTransformNormalFromObjectToWorldSpace(
                (point - rt_data->center) / rt_data->radius);

            
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
                p0, p1, p2, u, v,n0,n1,n2
            );
        }

}
