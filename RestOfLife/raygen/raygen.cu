#include <optix.h>
#include "../shaders/appconfig.h"
#include "../lib/raydata.cuh"

//#include "../scene/camera.cuh"
#include <vector_types.h>
#include <sutil/vec_math.h>
#include "../shaders/sysparameter.h"
#include <cuda/random.h>
#include <cuda/helpers.h>
#include "../shaders/FunctionIdx.h"
#include "../pdf/pdf.cuh"

extern "C" __constant__ SysParamter Parameter;


inline __device__ float3 removeNaNs(float3 radiance)
{
    float3 r = radiance;
    if(!(r.x == r.x)) r.x = 0.0f;
    if(!(r.y == r.y)) r.y = 0.0f;
    if(!(r.z == r.z)) r.z = 0.0f;
    return r;
}



inline __device__ float3 rayColor(PerRayData& thePrd, float3 origin, float3 direction,int depth)
{
    unsigned int seed = thePrd.seed;
 /*   float3 origin = thePrd.scattered_origin;
    float3 direction = thePrd.scattered_direction;*/
    float3 throughput = make_float3(1.0f);
    float3 sampleRadiance = make_float3(0.f);
    //PerRayData *raydatas = new  PerRayData[Parameter.maxRayDepth];
    while (depth < Parameter.maxRayDepth)
    {
        // rtTrace(sysWorld, theRay, thePrd);
        thePrd.lightDirection = -direction;
        uint2 data = splitPointer(&thePrd);
        optixTraverse(
            OPTIX_PAYLOAD_TYPE_ID_0,
            Parameter.handle,
            origin,
            direction,
            1e-6f,                     // tmin
            RT_DEFAULT_MAX,                    // tmax
            rnd(seed),
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_RADIANCE,        // SBT offset
            RAY_TYPE_COUNT,           // SBT stride
            RAY_TYPE_RADIANCE,        // missSBTIndex
            data.x, data.y);
        optixReorder(
            // Application specific coherence hints could be passed in here
        );

        optixInvoke(OPTIX_PAYLOAD_TYPE_ID_0, data.x, data.y);
        sampleRadiance += thePrd.radiance * throughput;
        if (thePrd.scatterEvent == Ray_Miss || thePrd.scatterEvent == Ray_Finish ||
            thePrd.scatterEvent == Ray_Cancel)
        {
            //sampleRadiance += thePrd.emitted;
            
            break;
        }
        else { // ray is still alive, and got properly bounced
                origin = thePrd.scattered_origin;
                direction = thePrd.scattered_direction;
                throughput *= thePrd.attenuation;
                      
        }
        if (2 <= depth) // Start termination after a minimum number of bounces.
        {
            const float probability = fmaxf(throughput); // Other options: // intensity(throughput); // fminf(0.5f, intensity(throughput));
            if (probability < rnd(thePrd.seed)) // Paths with lower probability to continue are terminated earlier.
            {
                break;
            }
            throughput /= probability; // Path isn't terminated. Adjust the throughput so that the average is right again.
        }
        depth++;
    }
    
    return sampleRadiance;
}

inline __device__ float3 color(PerRayData& thePrd, unsigned int& seed)
{
  //  PerRayData thePrd;
    thePrd.seed = seed;
    float3 sampleRadiance = make_float3(1.0f, 1.0f, 1.0f);
    thePrd.gatherTime = Parameter.cameraTime0 + 
        rnd(seed)*(Parameter.cameraTime1 - Parameter.cameraTime0);

    float3 origin = thePrd.scattered_origin;
    float3 direction = thePrd.scattered_direction;
    int depth = 0;
    sampleRadiance = rayColor(thePrd, origin, direction, depth);
    
    seed = thePrd.seed;
    return sampleRadiance;
   
}

inline __device__ PerRayData generateRay(float s, float t, unsigned int& seed)
{
    float3 initialOrigin, initialDirection;
    PerRayData theRay = {};
    
    optixDirectCall<void, const float, const float, unsigned int&,
        float3&, float3& >(Parameter.cameraType,
            s, t, seed, initialOrigin, initialDirection);
        theRay.scattered_origin = initialOrigin;
        theRay.scattered_direction = initialDirection;
        
        return theRay;
    
}


extern "C" __global__ void __raygen__Program()
{
    const uint3 theLaunchDim = optixGetLaunchDimensions();
    const uint3 theLaunchIndex = optixGetLaunchIndex();
    const int    w = Parameter.m_Nx;
    //const int    h = Parameter.m_Ny;// .height;
    unsigned int seed = tea<64>(theLaunchDim.x * theLaunchIndex.y + theLaunchIndex.x, 0);

    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
   
    {
        float s = float(theLaunchIndex.x+rnd(seed)) / float(theLaunchDim.x);
        float t = float(theLaunchIndex.y+rnd(seed)) / float(theLaunchDim.y);

        // generateRay is found in scene/camera.cuh
       
        PerRayData theRay = generateRay(s, t, seed);

        float3 sampleRadiance = color(theRay, seed);

        // Remove NaNs - should also remove from sample count? as this is a "bad" sample
        sampleRadiance = removeNaNs(sampleRadiance);

        radiance += sampleRadiance;
    }
   

    // gamma correction (2)
    radiance = make_float3(
        sqrtf(radiance.x),
        sqrtf(radiance.y),
        sqrtf(radiance.z)
    );
    const int image_index = theLaunchIndex.y * w + theLaunchIndex.x;
    float4 accum = make_float4(radiance, 1.f);
    Parameter.frame_buffer[image_index] = accum;//make_color(radiance);
}
