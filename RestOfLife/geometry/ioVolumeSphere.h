#ifndef IO_VOLUME_SPHERE_H
#define IO_VOLUME_SPHERE_H

#include "ioGeometry.h"

#include <optix.h>
#include "sutil/Aabb.h"
#include "sutil/vec_math.h"
#include "optix_types.h"
extern "C" const char volume_sphere_ptx_c[];

class ioVolumeSphere : public ioGeometry
{
public:
    ioVolumeSphere() {
        m_cx0 = 0.0f;
        m_cy0 = 0.0f;
        m_cz0 = 0.0f;
        m_r = 0.0f;
        m_density = 0.0f;
    }

    ioVolumeSphere(const float x, const float y, const float z, const float r, const float density)
        : m_cx0(x), m_cy0(y), m_cz0(z), m_r(r), m_density(density) {}

    virtual void init(OptixDeviceContext& context) {
        sutil::Aabb aabb;
        getBounds(&aabb);
        CUdeviceptr d_aabb_buffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer), sizeof(sutil::Aabb)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_aabb_buffer),
            &aabb,
            sizeof(sutil::Aabb),
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

    }

    void getBounds(sutil::Aabb* result)
    {
        sutil::Aabb* aabb = (sutil::Aabb*)result;
        float3 center = make_float3(m_cx0, m_cy0, m_cz0);
        aabb->m_min = center - abs(m_r);
        aabb->m_max = center + abs(m_r);
    }
    virtual std::array<float, 12> getTransform()
    {
        std::array<float, 12> transArr = { 1.f,0.f,0.f,0,
                                         0.f,1.f,0.f,0,
                                         0.f,0.f,1.f,0 };
        return transArr;
    }
    virtual float3  getCenter()
    {
        return make_float3(m_cx0,m_cy0,m_cz0);
    }
    virtual float   getR()
    {
        return m_r;
    }
public:
    float m_cx0;
    float m_cy0;
    float m_cz0;
    float m_r;
    float m_density;
};

#endif //!IO_VOLUME_SPHERE_H
