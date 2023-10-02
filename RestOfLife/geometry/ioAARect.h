#ifndef IO_AA_RECT_H
#define IO_AA_RECT_H

#include "ioGeometry.h"

#include <optix.h>
#include "sutil/Exception.h"
#include <vector_types.h>

enum Axis { X_AXIS, Y_AXIS, Z_AXIS };

class ioAARect : public ioGeometry
{
public:
    ioAARect() {}

    ioAARect(const float a0, const float a1, const float b0, const float b1, const float k, const bool flip, const Axis orientation)
        {
            m_a0 = a0;
            m_a1 = a1;
            m_b0 = b0;
            m_b1 = b1;
            m_k  = k;
            m_flip = flip;
            kind = orientation;
        }

    virtual std::array<float, 12> getTransform() {
        std::array<float, 12> transArr = { 1.f,0.f,0.f,0,
                                         0.f,1.f,0.f,0,
                                         0.f,0.f,1.f,0 };
        return transArr;
    }
    virtual float3  getCenter() {
        return make_float3(0.f, 0.f, 0.f);
    }
    virtual float   getR() {
        return 0;
    }

    void init(OptixDeviceContext& context) {

        OptixAabb aabb;
        switch (kind) {
        case X_AXIS:
            getBoundsX(&aabb);
            break;
        case Y_AXIS:
            getBoundsY(&aabb);
            break;
        case Z_AXIS:
            getBoundsZ(&aabb);
            break;
        }
        
        CUdeviceptr d_aabb_buffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer), sizeof(OptixAabb)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_aabb_buffer),
            &aabb,
            sizeof(OptixAabb),
            cudaMemcpyHostToDevice
        ));
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        uint32_t sphere_input_flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        OptixBuildInput sphere_input = {};
        sphere_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        sphere_input.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
        sphere_input.customPrimitiveArray.numPrimitives = 1;
        sphere_input.customPrimitiveArray.flags = &sphere_input_flag;
        sphere_input.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options,
            &sphere_input,
            1,  // num_build_inputs
            &gas_buffer_sizes));

        CUdeviceptr d_temp_buffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

        // non-compacted output
        CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
        size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
            compactedSizeOffset + 8
        ));


        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);
        OPTIX_CHECK(optixAccelBuild(context, 0, &accel_options,
            &sphere_input, 1, d_temp_buffer, gas_buffer_sizes.tempSizeInBytes,
            d_buffer_temp_output_gas_and_compacted_size, compactedSizeOffset + 8,
            &sphere_gas_handle, &emitProperty, 1));


        //m_geo = context->createGeometry();
        //m_geo->setPrimitiveCount(1);

        //if (kind == X_AXIS) {
        //    m_geo->setBoundingBoxProgram(context->createProgramFromPTXString(aarect_ptx_c, "getBoundsX"));
        //    m_geo->setIntersectionProgram(context->createProgramFromPTXString(aarect_ptx_c, "hitRectX"));
        //} else if  (kind == Y_AXIS) {
        //    m_geo->setBoundingBoxProgram(context->createProgramFromPTXString(aarect_ptx_c, "getBoundsY"));
        //    m_geo->setIntersectionProgram(context->createProgramFromPTXString(aarect_ptx_c, "hitRectY"));
        //} else if  (kind == Z_AXIS) {
        //    m_geo->setBoundingBoxProgram(context->createProgramFromPTXString(aarect_ptx_c, "getBoundsZ"));
        //    m_geo->setIntersectionProgram(context->createProgramFromPTXString(aarect_ptx_c, "hitRectZ"));
        //} else { // should never reach this branch
        //    m_geo->setBoundingBoxProgram(context->createProgramFromPTXString(aarect_ptx_c, "getBoundsY"));
        //    m_geo->setIntersectionProgram(context->createProgramFromPTXString(aarect_ptx_c, "hitRectY"));
        //}
        //_init();
    }

    void _init() {
      /*  m_geo["a0"]->setFloat(m_a0);
        m_geo["a1"]->setFloat(m_a1);
        m_geo["b0"]->setFloat(m_b0);
        m_geo["b1"]->setFloat(m_b1);
        m_geo["k"]->setFloat(m_k);*/
    }
     void getBoundsX(OptixAabb* result) {
        OptixAabb* aabb = (OptixAabb*)result;

        float3 minbox  = make_float3(m_k - 0.0001f, m_a0, m_b0);
        float3 maxbox = make_float3(m_k + 0.0001f, m_a1, m_b1);
        aabb->minX = minbox.x, aabb->minY = minbox.y, aabb->minZ = minbox.z;
        aabb->maxX = maxbox.x, aabb->maxY = maxbox.y, aabb->maxZ = maxbox.z;
    }

    void getBoundsY(OptixAabb* result) {
        OptixAabb* aabb = (OptixAabb*)result;

        float3 minbox = make_float3(m_a0, m_k - 0.0001f, m_b0);
        float3 maxbox = make_float3(m_a1, m_k + 0.0001f, m_b1);

        aabb->minX = minbox.x, aabb->minY = minbox.y, aabb->minZ = minbox.z;
        aabb->maxX = maxbox.x, aabb->maxY = maxbox.y, aabb->maxZ = maxbox.z;
    }

    void getBoundsZ(OptixAabb *result) {
        OptixAabb* aabb = (OptixAabb*)result;

        float3 minbox = make_float3(m_a0, m_b0, m_k - 0.0001f);
        float3 maxbox = make_float3(m_a1, m_b1, m_k + 0.0001f);
        aabb->minX = minbox.x, aabb->minY = minbox.y, aabb->minZ = minbox.z;
        aabb->maxX = maxbox.x, aabb->maxY = maxbox.y, aabb->maxZ = maxbox.z;
    }
public:
    float m_a0;
    float m_a1;
    float m_b0;
    float m_b1;
    float m_k;
    bool  m_flip;
    Axis kind;
};

#endif //!IO_AA_RECT_H
