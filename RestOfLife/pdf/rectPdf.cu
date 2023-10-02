#include <optix.h>
#include "pdf.cuh"
#include "../lib/raydata.cuh"
#include <cfloat>
#include <cuda/random.h>
#include "../shaders/sysparameter.h"
//rtDeclareVariable(float,  a0, , );
//rtDeclareVariable(float,  a1, , );
//rtDeclareVariable(float,  b0, , );
//rtDeclareVariable(float,  b1, , );
//rtDeclareVariable(float,  k, , );
//rtDeclareVariable(int,  flip, , );

#define DENOMINATOR_EPSILON 1.0e-6f
extern "C" __constant__ SysParamter Parameter;

//inline __device__ bool hit_x(pdf_in &in, const float tmin, const float tmax, pdf_rec &rec) {
//    hitRectData &pdfrect = Parameter.pdf.pdfrect;
//    float t = (pdfrect.k - in.origin.x) / in.light_direction.x;
//
//    float a = in.origin.y + t * in.light_direction.y;
//    float b = in.origin.z + t * in.light_direction.z;
//    if (a < pdfrect.a0 || a > pdfrect.a1 ||
//        b < pdfrect.b0 || b > pdfrect.b1)
//        return false;
//
//    if (t < tmax && t > tmin){
//        rec.normal = make_float3(1.f, 0.f, 0.f);
//        rec.distance = t;
//        return true;
//    }
//
//    return false;
//}
//
//inline __device__ bool hit_y(pdf_in &in, const float tmin, const float tmax, pdf_rec &rec) {
//
//    hitRectData& pdfrect = Parameter.pdf.pdfrect;
//    float t = (pdfrect.k - in.origin.y) / in.light_direction.y;
//
//    float a = in.origin.x + t * in.light_direction.x;
//    float b = in.origin.z + t * in.light_direction.z;
//    if (a < pdfrect.a0 || a > pdfrect.a1 || b < pdfrect.b0 || b > pdfrect.b1)
//        return false;
//
//    if (t < tmax && t > tmin){
//        rec.normal = make_float3(0.f, 1.f, 0.f);
//        rec.distance = t;
//        return true;
//    }
//
//    return false;
//}
//
//inline __device__ bool hit_z(pdf_in &in, const float tmin, const float tmax, pdf_rec &rec) {
//    hitRectData& pdfrect = Parameter.pdf.pdfrect;
//    float t = (pdfrect.k - in.origin.z) / in.light_direction.z;
//
//    float a = in.origin.x + t * in.light_direction.x;
//    float b = in.origin.y + t * in.light_direction.y;
//    if (a < pdfrect.a0 || a > pdfrect.a1 ||
//        b < pdfrect.b0 || b > pdfrect.b1)
//        return false;
//
//    if (t < tmax && t > tmin){
//        rec.normal = make_float3(0.f, 0.f, 1.f);
//        rec.distance = t;
//        return true;
//    }
//
//    return false;
//}


extern "C" __device__ float __direct_callable__rect_x_value(pdf_in &in) {
    pdf_rec rec;
    hitRectData& pdfrect = Parameter.pdf.pdfrect;
   /* if(hit_x(in, 0.001f, FLT_MAX, rec)){
        float area = (pdfrect.a1 - pdfrect.a0) * 
            (pdfrect.b1 - pdfrect.b0);
        float distance_squared = rec.distance * rec.distance * 
            dot(in.light_direction, in.light_direction);
        float cosine = fabs(dot(in.light_direction, rec.normal)) / length(in.light_direction);
        in.cosin_pdf = fabs(dot(in.light_direction, make_float3(1.f, 0, 0.f))) / CUDART_PI_F;
        return distance_squared / (cosine * area);
    }
    else*/
        return 0.0000001f;
}

extern "C" __device__ float __direct_callable__rect_y_value(pdf_in &in) {
    pdf_rec rec;
    hitRectData const& pdfrect = Parameter.pdf.pdfrect;
   // HitGroupData* hitData = (HitGroupData*)optixGetSbtDataPointer();
    
    /*if(hit_y(in, 0.001f, FLT_MAX, rec)){
        float area = (pdfrect.a1 - pdfrect.a0) * 
            (pdfrect.b1 - pdfrect.b0);
        float distance_squared = rec.distance * rec.distance * dot(in.light_direction, in.light_direction);
        float cosine = fabs(dot(in.light_direction, rec.normal)) / length(in.light_direction);
        in.cosin_pdf = fabs(dot(in.light_direction, make_float3(0, 1.f, 0.f))) / CUDART_PI_F;
        in.light_dist = rec.distance;
        return distance_squared / (cosine * area);
    }
    else*/
        return 0.0000001f;
}

extern "C" __device__ float __direct_callable__rect_z_value(pdf_in &in) {
    pdf_rec rec;
    hitRectData& pdfrect = Parameter.pdf.pdfrect;
    /*if(hit_z(in, 0.001f, FLT_MAX, rec)){
        float area = (pdfrect.a1 - pdfrect.a0) * 
            (pdfrect.b1 - pdfrect.b0);
        float distance_squared = rec.distance * rec.distance * dot(in.light_direction, in.light_direction);
        float cosine = fabs(dot(in.light_direction, rec.normal)) / length(in.light_direction);
        in.cosin_pdf = fabs(dot(in.light_direction, make_float3(0, 0, 1.f)))/ CUDART_PI_F;
        return distance_squared / (cosine * area);
    }
    else*/
        return 0.00000001f;
}

extern "C" __device__ LightSample __direct_callable__rect_x_generate(LightDefinition const& light,
    hitRectData const& pdfrect, pdf_in &in, uint32_t& seed) {
    LightSample lightSample;
  //  hitRectData& pdfrect = Parameter.pdf.pdfrect;
    float3 random_point = make_float3(pdfrect.k, pdfrect.a0 + rnd(seed)
        * (pdfrect.a1 -pdfrect.a0), pdfrect.b0 + 
        rnd(seed) * (pdfrect.b1 - pdfrect.b0));
    lightSample.direction = random_point - in.origin;
    lightSample.position = random_point;
    lightSample.distance = length(lightSample.direction);
    lightSample.pdf = 0.f;
    if (lightSample.distance > DENOMINATOR_EPSILON) {
        lightSample.direction /= lightSample.distance;
        const float costa = dot(-lightSample.direction, light.normal);
        if (costa > DENOMINATOR_EPSILON)
        {
            lightSample.emission = light.emission * Parameter.numLights;
            lightSample.pdf = lightSample.distance * lightSample.distance / (light.area * costa);
        }
    }
    return lightSample;
}

extern "C" __device__ LightSample __direct_callable__rect_y_generate(LightDefinition const& light,
    hitRectData const &pdfrect, pdf_in &in, uint32_t& seed) {
    
    float3 random_point = make_float3(pdfrect.a0 + rnd(seed) * 
        (pdfrect.a1 - pdfrect.a0), pdfrect.k, 
        pdfrect.b0 + rnd(seed) * (pdfrect.b1 -pdfrect.b0));
    LightSample lightSample;
    lightSample.direction = random_point - in.origin;
    lightSample.position = random_point;
    lightSample.distance = length(lightSample.direction);
    lightSample.pdf = 0.f;
    if (lightSample.distance > DENOMINATOR_EPSILON) {
        lightSample.direction /= lightSample.distance;
        const float costa = dot(-lightSample.direction, light.normal);
        if (costa > DENOMINATOR_EPSILON)
        {
            lightSample.emission = light.emission * Parameter.numLights;
            lightSample.pdf = lightSample.distance * lightSample.distance / (light.area * costa);
        }
    }
    return lightSample;
   
}

extern "C" __device__ LightSample __direct_callable__rect_z_generate(LightDefinition const& light,
    hitRectData const& pdfrect, pdf_in &in, uint32_t& seed) {
   // hitRectData& pdfrect = Parameter.pdf.pdfrect;
    float3 random_point = make_float3(pdfrect.a0 + rnd(seed) *
        (pdfrect.a1 -pdfrect.a0), pdfrect.b0 + 
        rnd(seed) * (pdfrect.b1 - pdfrect.b0), pdfrect.k);
    LightSample lightSample;
    lightSample.direction = random_point - in.origin;
    lightSample.position = random_point;
    lightSample.distance = length(lightSample.direction);
    lightSample.pdf = 0.f;
    if (lightSample.distance > DENOMINATOR_EPSILON) {
        lightSample.direction /= lightSample.distance;
        const float costa = dot(-lightSample.direction, light.normal);
        if (costa > DENOMINATOR_EPSILON)
        {
            lightSample.emission = light.emission * Parameter.numLights;
            lightSample.pdf = lightSample.distance * lightSample.distance / (light.area * costa);
        }
    }
    return lightSample;
    
}
