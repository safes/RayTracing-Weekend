#include <optix.h>
#include <cfloat>
#include "sutil/vec_math.h"
#include "../lib/raydata.cuh"
#include <cuda/random.h>
#include "../shaders/sysparameter.h"
// Sphere variables
//rtDeclareVariable(float3, center, , );
//rtDeclareVariable(float, radius, , );
//rtDeclareVariable(float, density, , );

// The ray that will be intersected against
//rtDeclareVariable(optix::Ray, theRay, rtCurrentRay, );
//rtDeclareVariable(PerRayData, thePrd, rtPayload,  );
//
// The point and normal of intersection. and uv-space location
//   the "attribute" qualifier is used to communicate between intersection and shading programs
//   These may only be written between rtPotentialIntersection and rtReportIntersection
//rtDeclareVariable(HitRecord, hitRecord, attribute hitRecord, );
extern "C" __constant__ SysParamter Parameter;

constexpr int EPSILON = (0.0001f);

// assume p is normalized direction vector
//inline __device__ void get_sphere_uv(const float3 p) {
//	float phi = atan2(p.z, p.x);
//	float theta = asin(p.y);
//
//	hitRecord.u = 1 - (phi + CUDART_PI_F) / (2.f * CUDART_PI_F);
//	hitRecord.v = (theta + CUDART_PIO2_F) / CUDART_PI_F;
//}


// The sphere intersection program
//   this function calls rtReportIntersection if an intersection occurs
//   As above, pid refers to a specific primitive, is ignored
inline __device__ bool hit_boundary(const float3 direction,const float3 origin,const float3 center,
    const float radius, const float tmin, const float tmax, float &rec)
{
    float3 oc = origin - center;
    float a = dot(direction, direction);
    float b = dot(oc, direction);
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;

    if (discriminant < 0.f) return false;

    float t = (-b - sqrtf(discriminant)) / a;
    if (t < tmax && t > tmin) {
        rec = t;
        return true;
    }

    t       = (-b + sqrtf(discriminant)) / a;
    if (t < tmax && t > tmin) {
        rec = t;
        return true;
    }

    return false;
}


// The sphere intersection program
//   this function calls rtReportIntersection if an intersection occurs
//   As above, pid refers to a specific primitive, is ignored
extern "C" __global__ void __intersection__hitVolumesphere()
{
    float hitt1, hitt2;
    PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    float3 direction = optixGetObjectRayDirection();
    float3 origin = optixGetObjectRayOrigin();
    if(hit_boundary(direction, origin, rt_data->center,rt_data->radius, -FLT_MAX, FLT_MAX, hitt1)) {
        if(hit_boundary(direction, origin, rt_data->center, rt_data->radius, hitt1 + EPSILON, FLT_MAX, hitt2)) {
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

            float hitDistance = -(1.f / rt_data->density) * logf(rnd(thePrd->seed));
            float hitt = hitt1 + (hitDistance / length(direction));

            /*if (rtPotentialIntersection(hitt))*/ {
                float3 point = optixTransformPointFromObjectToWorldSpace(origin + hitt * direction);
              //  hitRecord.point = rtTransformPoint(RT_OBJECT_TO_WORLD, theRay.origin + hitt*theRay.direction);

                float ux = 0.f;
                float vy = 0.f;

                float3 normal = make_float3(1.f, 0.f, 0.f);
                float3 shading_normal = normalize(optixTransformNormalFromObjectToWorldSpace(
                     normal));

                unsigned int p0, p1, p2;
                p0 = __float_as_uint(shading_normal.x);
                p1 = __float_as_uint(shading_normal.y);
                p2 = __float_as_uint(shading_normal.z);
                unsigned int u, v;
                u = __float_as_uint(ux);
                v = __float_as_uint(vy);
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

// The sphere bounding box program
//   The pid parameter enables specifying a primitive withing this geometry
//   since there is only 1 primitive (the sphere), the pid is ignored here
