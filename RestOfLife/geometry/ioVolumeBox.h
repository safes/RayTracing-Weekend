#ifndef IO_VOLUME_BOX_H
#define IO_VOLUME_BOX_H

#include <iostream>

#include "ioGeometry.h"
#include "../material/ioMaterial.h"

#include <optix.h>
#include <vector_types.h>
#include <sutil/Aabb.h>

extern "C" const char volume_box_ptx_c[];
#define EPSILON (0.0001f)

class ioVolumeBox : public ioGeometry
{
public:

    ioVolumeBox(const float3 &p0, const float3 &p1, const float density)
        : m_boxMin(p0), m_boxMax(p1), m_density(density) {
        // std::cerr  << "boxMin(" << m_boxMin.x << ',' << m_boxMin.y << ',' << m_boxMin.z  << ')' << " "
        //            << "boxMax(" << m_boxMax.x << ',' << m_boxMax.y << ',' << m_boxMax.z  << ')' << std::endl;
    }

    void init(OptixDeviceContext& context) {
        CUdeviceptr d_aabb_buffer;
        sutil::Aabb aabb;
        getBounds(&aabb);
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
        // rtPrintf("boxMin(%f,%f,%f)\n", boxMin.x, boxMin.y, boxMin.z);
        // rtPrintf("boxMax(%f,%f,%f)\n", boxMax.x, boxMax.y, boxMax.z);
        // NOTE: assume all components of boxMin are less than  boxMax
        aabb->m_min = m_boxMin - make_float3(EPSILON);
        aabb->m_max = m_boxMax + make_float3(EPSILON);
    }

    void _init() {
    /*    m_geo["boxMin"]->setFloat(m_boxMin);
        m_geo["boxMax"]->setFloat(m_boxMax);
        m_geo["density"]->setFloat(m_density);*/
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
        return make_float3(0.f);
    }
    virtual float   getR()
    {
        return 0.f;
    }

public:
    const float3 m_boxMin;
    const float3 m_boxMax;
    const float m_density;
};

#endif //!IO_VOLUME_BOX_H
