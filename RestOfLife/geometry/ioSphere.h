#ifndef IO_SPHERE_H
#define IO_SPHERE_H

#include "ioGeometry.h"

#include <optix.h>
#include <sutil/vec_math.h>
#include <vector_types.h>
#include "sutil/Exception.h"

extern "C" const char sphere_ptx_c[];

class ioSphere : public ioGeometry
{
public:

  ioSphere(const float x, const float y, const float z, const float r)
    {
      m_cx = x;
      m_cy = y;
      m_cz = z;
      m_r = r;
    }

 void getBounds(OptixAabb* result)
{
    OptixAabb* aabb = (OptixAabb*)result;
    float3 center = { m_cx,m_cy,m_cz };
    float3 normalc = normalize(center) * m_r;
    float3 minbox = (center - m_r);
    float3 maxbox = center + m_r;
    aabb->minX = minbox.x, aabb->minY = minbox.y, aabb->minZ = minbox.z;
    aabb->maxX = maxbox.x, aabb->maxY = maxbox.y, aabb->maxZ = maxbox.z;
}

  std::array<float,12> getTransform()
 {
      std::array<float, 12> transArr = { 1.f,0.f,0.f,0,
                                         0.f,1.f,0.f,0,
                                         0.f,0.f,1.f,0 };
      return transArr;
      
 }

  virtual void init(OptixDeviceContext& context)
  {
      OptixAabb aabb;
      getBounds(&aabb);
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


   /*   m_geo = context->createGeometry();
      m_geo->setPrimitiveCount(1);
      m_geo->setBoundingBoxProgram(
        context->createProgramFromPTXString(sphere_ptx_c, "getBounds")
        );
      m_geo->setIntersectionProgram(
        context->createProgramFromPTXString(sphere_ptx_c, "intersection")
        );
      m_geo["center"]->setFloat(m_cx, m_cy, m_cz);
      m_geo["radius"]->setFloat(m_r);*/

  }
  virtual float3  getCenter()
  {
      return make_float3(m_cx, m_cy, m_cz);
  }
  virtual float   getR()
  {
      return m_r;
  }
private:
  float m_cx;
  float m_cy;
  float m_cz;
  float m_r;
};

#endif //!IO_SPHERE_H
