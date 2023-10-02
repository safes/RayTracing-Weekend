#ifndef ONB_CUH
#define ONB_CUH

// optix code
#include <optix.h>
#include "sutil/vec_math.h"


struct onb {
    // __device__ onb() {}

     float3 __device__ local(float a, float b, float c) const {
        return a*u + b*v + c*w;
    }

     float3 __device__ local(const float3 &a) const {
        return a.x*u + a.y*v + a.z*w;
    }

     void __device__ buildFromW(const float3& n){
        w = normalize(n);

        float3 a;
        // equivalent to if(fabsf(w.x) > 0.9f)
        if((w.x > 0.9f) || (w.x < -0.9f))
            a = make_float3(0.f, 1.f, 0.f);
        else
            a = make_float3(1.f, 0.f, 0.f);

        v = normalize(cross(w, a));
        u = cross(w, v);
    }

    float3 u, v, w;
};


#endif //!ONB_CUH
