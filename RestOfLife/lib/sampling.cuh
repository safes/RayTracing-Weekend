#ifndef SAMPLING_CUH
#define SAMPLING_CUH

#include <optix.h>
//#include <optix_world.h>
//#include <optixu/optixu_math_namespace.h>
#include <vector_types.h>
#include "sutil/vec_math.h"
// defines CUDART_PI_F
#include "math_constants.h"

#include <cuda/random.h>


inline __device__ float3 random_in_unit_disk(uint32_t& seed) {
    float a = rnd(seed) * 2.0f * CUDART_PI_F;

    float3 xy = make_float3(sinf(a), cosf(a), 0);
    xy *= sqrtf(rnd(seed));

  return xy;
}


inline __device__ float3 randomInUnitSphere(uint32_t& seed)
{
    float3 p;
    do
    {
        p = 2.0f*make_float3(rnd(seed), rnd(seed), rnd(seed))
            - make_float3(1.0f, 1.0f, 1.0f);
    } while(dot(p,p) >= 1.0f);
    return p;
}

// inline __device__ float3 r_andomInUnitSphere(uint32_t& seed) {
//     float z = randf(seed)*2.0f - 1.0f;
// 	float t = randf(seed) * (2.0f * CUDART_PI_F);
// 	float r = sqrtf((0.0f > (1.0f - z*z) ? 0.0f : (1.0f - z*z)));
// 	float x = r * cosf(t);
// 	float y = r * sinf(t);

// 	float3 res = make_float3(x, y, z);
// 	res *= powf(randf(seed), 1.0f / 3.0f);

//   return res;
// }

inline __device__ float3 randomCosineDirection(uint32_t& seed) {
	float r1 = rnd(seed);
	float r2 = rnd(seed);

	float phi = 2.f * CUDART_PI_F * r1;

	float x = cosf(phi) * 2 * sqrtf(r2);
	float y = sinf(phi) * 2 * sqrtf(r2);
	float z = sqrtf(1 - r2);

	return make_float3(x, y, z);
}




#endif //!SAMPLING_CUH
