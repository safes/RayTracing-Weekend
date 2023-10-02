#include <optix.h>

#include "sutil/vec_math.h"
#include "../lib/raydata.cuh"
#include "../shaders/sysparameter.h"
extern "C" __constant__ SysParamter Parameter;

extern "C" __global__ void __intersection__hitRectY() {

    HitGroupData* hitrec = (HitGroupData*)optixGetSbtDataPointer();
    hitRectData* hitrc = &hitrec->rectData;
    float3 origin = optixGetObjectRayOrigin();
    float3 direction = optixGetObjectRayDirection();
    float t = (hitrc->k - origin.y) / direction.y;
    if (t > optixGetRayTmax() || t < optixGetRayTmin())
        return;

    float a = origin.x + t * direction.x;
    float b = origin.z + t * direction.z;
    if (a < hitrc->a0 || a > hitrc->a1 || b < hitrc->b0 || b > hitrc->b1)
        return;

    {
        float3 point = optixTransformPointFromObjectToWorldSpace(origin + t * direction);

        float3 normal = make_float3(0.f, 1.f, 0.f);
        if (hitrc->flip)//(0.f < dot(normal, direction))
            normal = -normal;

        float3 shade_normal = normalize(optixTransformNormalFromObjectToWorldSpace( normal));

        float ux = (a - hitrc->a0) / (hitrc->a1 - hitrc->a0);
        float vy = (b - hitrc->b0) / (hitrc->b1 - hitrc->b0);
        unsigned int p0, p1, p2;
        p0 = __float_as_uint(point.x);
        p1 = __float_as_uint(point.y);
        p2 = __float_as_uint(point.z);
        unsigned int u, v;
        u = __float_as_uint(ux);
        v = __float_as_uint(vy);
        unsigned int n0, n1, n2;
        n0 = __float_as_uint(shade_normal.x);
        n1 = __float_as_uint(shade_normal.y);
        n2 = __float_as_uint(shade_normal.z);
        optixReportIntersection(
            t, 0,n0, n1, n2
            , u, v,p0, p1, p2 
        );
    }
}