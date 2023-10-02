#ifndef RAYDATA_CUH
#define RAYDATA_CUH

#include <optix.h>
#include <vector_types.h>
#include <texture_types.h>
// defines CUDART_PI_F
#include "math_constants.h"
enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_COUNT
};
typedef enum
{
    Ray_Miss,
    Ray_Hit,
    Ray_Finish,
    Ray_Cancel
} ScatterEvent;

typedef enum {
    Lambertian,
    DiffuseLight,
    Metallic,
    Dielectric,
    Isotropic,
    Normalic,
} MaterialType;

struct LightDefinition
{
  //  LightType type; // Constant or spherical environment, rectangle (parallelogram).

    // Rectangle lights are defined in world coordinates as footpoint and two vectors spanning a parallelogram.
    // All in world coordinates with no scaling.
    float3 position;
    float3 vecU;
    float3 vecV;
    float3 normal;
    float  area;
    float3 emission;

    // Manual padding to float4 alignment goes here.
    float unused0;
    float unused1;
    float unused2;
};

struct LightSample
{
    float3 position;
    float  distance;
    float3 direction;
    float3 emission;
    float  pdf;
};

struct PerRayData
{
    float3 attenuation;         // 12 bytes
    float3 scattered_origin;    // 12 bytes
    float3 scattered_direction; // 12 bytes // obviated by PDF importance sampling for non-specular materials
    float3 emitted;             // 12 bytes
    float3 radiance;
    float  distance;
    float3 hit_normal;          // 12 bytes - Need to save per ray?
    ScatterEvent scatterEvent;  //  4 bytes
    float  pdf;                 //  4 bytes
    float3 lightDirection;      //  light direction
    MaterialType materialType;  //  4 bytes
    float  scattered_pdf;       //  4 bytes
    float  gatherTime;          //  4 bytes
    unsigned int   seed;                //  4 bytes
    int            depth;
    bool   is_specular;         //  1 byte?

};
struct hitRectData
{
    float a0;
    float a1;
    float b0;
    float b1;
    float k;
    int   flip;
};

struct hitVolumeBoxdata
{
     float3 boxMin;
     float3 boxMax;
     float density;

};

struct HitGroupData
{
    float3     color;

    // For spheres.  In real use case, we would have an abstraction for geom data/ material data
    float3     center;
    float      radius;

    float3 center1;
    float time0;
    float time1;
    int   bMotion;
    float density;  //only for volume sphere.
    hitRectData rectData;
    hitVolumeBoxdata VolBoxData;
    
    unsigned int pad;

};


struct HitRecord
{
    float3 point;   // 12 bytes
    float3 normal;  // 12 bytes
    float distance; //  4 bytes
    float u;        //  4 bytes
    float v;        //  4 bytes
};

struct textureParam
{
    int texCallidx;

    float3 color; // for contant tex
    int odd, even; // for odd even tex index
    float scale;
    CUdeviceptr randvec;
    CUdeviceptr permX, permY, permZ;
    cudaTextureObject_t imagetex;
    
};


typedef union
{
    PerRayData* ptr;
    uint2       dat;
} Payload;

__forceinline__ __device__ uint2 splitPointer(PerRayData* ptr)
{
    Payload payload;

    payload.ptr = ptr;

    return payload.dat;
}

__forceinline__ __device__ PerRayData* mergePointer(unsigned int p0, unsigned int p1)
{
    Payload payload;

    payload.dat.x = p0;
    payload.dat.y = p1;

    return payload.ptr;
}

// Used for Multiple Importance Sampling.
__forceinline__ __host__ __device__ float powerHeuristic(const float a, const float b)
{
    const float t = a * a;
    return t / (t + b * b);
}

__forceinline__ __host__ __device__ bool isNull(const float3& v)
{
    return (v.x == 0.0f && v.y == 0.0f && v.z == 0.0f);
}

__forceinline__ __host__ __device__ bool isNotNull(const float3& v)
{
    return (v.x != 0.0f || v.y != 0.0f || v.z != 0.0f);
}
#endif //!RAYDATA_CUH
