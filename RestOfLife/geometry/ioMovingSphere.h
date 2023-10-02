#ifndef IO_MOVING_SPHERE_H
#define IO_MOVING_SPHERE_H

#include "ioGeometry.h"
#include "../material/ioMaterial.h"

#include <optix.h>
#include <vector_types.h>
#include <optix_types.h>
#include <sutil/vec_math.h>
#include <sutil/Exception.h>
extern "C" const char moving_sphere_ptx_c[];

class ioMovingSphere : public ioGeometry
{
public:
  ioMovingSphere()
    {
      m_cx0 = 0.0f;
      m_cy0 = 0.0f;
      m_cz0 = 0.0f;
      m_cx1 = 0.0f;
      m_cy1 = 0.0f;
      m_cz1 = 0.0f;
      m_r = 0.0f;
      m_t0 = 0.0f;
      m_t1 = 0.0f;
    }

  ioMovingSphere(const float x, const float y, const float z,
                 const float x1, const float y1, const float z1,
                 const float r,
                 const float t0 = 0.f, const float t1 = 1.f)
    {
      m_cx0 = x;
      m_cy0 = y;
      m_cz0 = z;
      m_cx1 = x1;
      m_cy1 = y1;
      m_cz1 = z1;
      m_r = r;
      m_t0 = t0;
      m_t1 = t1;
    }

   void getBounds(OptixAabb* result)
  {
      OptixAabb* box0 = (OptixAabb*)result;
      float3 center0 = { m_cx0,m_cy0,m_cz0 };
      float3 normalc =   normalize(center0)*m_r;
      float3 minbox = center0 - m_r;
      float3 maxbox = (center0 +  m_r);
      box0->minX = minbox.x, box0->minY = minbox.y, box0->minZ = minbox.z;
      box0->maxX = maxbox.x, box0->maxY = maxbox.y, box0->maxZ = maxbox.z;

      OptixAabb box1;
      float3 center1 = { m_cx1,m_cy1,m_cz1 };
      normalc = normalize(center1) * m_r;
      minbox = center1 - m_r;
      maxbox = center1 + m_r;

      box1.minX = minbox.x, box1.minY = minbox.y, box1.minZ = minbox.z;
      box1.maxX = maxbox.x, box1.maxY = maxbox.y, box1.maxZ = maxbox.z;
      
      box0->minX = fmin(box0->minX, box1.minX);
      box0->minY = fmin(box0->minY, box1.minY);
      box0->minZ = fmin(box0->minZ, box1.minZ);
      box0->maxX = fmax(box0->maxX, box1.maxX);
      box0->maxY = fmax(box0->maxY, box1.maxY);
      box0->maxZ = fmax(box0->maxZ, box1.maxZ);
      
  }
   std::array<float, 12> getTransform()
   {
       std::array<float, 12> transArr = { 1.f,0.f,0.f,0,
                                          0.f,1.f,0.f,0,
                                          0.f,0.f,1.f,0 };
       return transArr;

   }

   virtual float3  getCenter()
   {
       return make_float3(m_cx0, m_cy0, m_cz0);
   }
   virtual float   getR()
   {
       return m_r;
   }
  virtual void init(OptixDeviceContext& context)
    {
      // create box accel 
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

    //  OptixTraversableHandle sphere_gas_handle;
      OptixAccelEmitDesc emitProperty = {};
      emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
      emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);
      OPTIX_CHECK(optixAccelBuild(context, 0, &accel_options,
          &sphere_input, 1, d_temp_buffer, gas_buffer_sizes.tempSizeInBytes,
          d_buffer_temp_output_gas_and_compacted_size, compactedSizeOffset + 8,
          &sphere_gas_handle, &emitProperty, 1));

      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_aabb_buffer)));

      size_t compacted_gas_size;
      CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));
      CUdeviceptr d_sphere_gas_output_buffer;
      if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
      {
          CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sphere_gas_output_buffer), compacted_gas_size));

          // use handle as input and output
          OPTIX_CHECK(optixAccelCompact(context, 0, sphere_gas_handle, d_sphere_gas_output_buffer,
              compacted_gas_size, &sphere_gas_handle));

          CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
      }
      else
      {
          d_sphere_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
      }

      {
          const float motion_matrix_keys[2][12] =
          {
              {
                  1.0f, 0.0f, 0.0f, m_cx0,
                  0.0f, 1.0f, 0.0f, m_cy0,
                  0.0f, 0.0f, 1.0f, m_cz0
              },
              {
                  1.0f, 0.0f, 0.0f, m_cx1,
                  0.0f, 1.0f, 0.0f, m_cy1,
                  0.0f, 0.0f, 1.0f, m_cz1
              }
          };

          OptixMatrixMotionTransform motion_transform = {};
          motion_transform.child = sphere_gas_handle;
          motion_transform.motionOptions.numKeys = 2;
          motion_transform.motionOptions.timeBegin = 0.0f;
          motion_transform.motionOptions.timeEnd = 1.0f;
          motion_transform.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
          memcpy(motion_transform.transform, motion_matrix_keys, 2 * 12 * sizeof(float));
          
          CUdeviceptr d_sphere_motion_transform;
          CUDA_CHECK(cudaMalloc(
              reinterpret_cast<void**>(&d_sphere_motion_transform),
              sizeof(OptixMatrixMotionTransform)
          ));

          CUDA_CHECK(cudaMemcpy(
              reinterpret_cast<void*>(d_sphere_motion_transform),
              &motion_transform,
              sizeof(OptixMatrixMotionTransform),
              cudaMemcpyHostToDevice
          ));

          OPTIX_CHECK(optixConvertPointerToTraversableHandle(
              context,
              d_sphere_motion_transform,
              OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM,
              &sphere_motion_transform_handle
          ));
      }

   /*   m_geo = context->createGeometry();
      m_geo->setPrimitiveCount(1);
      m_geo->setBoundingBoxProgram(
        context->createProgramFromPTXString(moving_sphere_ptx_c, "getBounds")
        );
      m_geo->setIntersectionProgram(
        context->createProgramFromPTXString(moving_sphere_ptx_c, "intersection")
        );
      m_geo["center0"]->setFloat(m_cx0, m_cy0, m_cz0);
      m_geo["center1"]->setFloat(m_cx1, m_cy1, m_cz1);
      m_geo["radius"]->setFloat(m_r);
      m_geo["time0"]->setFloat(m_t0);
      m_geo["time1"]->setFloat(m_t1);*/
    }

    virtual OptixTraversableHandle get()
    {
        return sphere_motion_transform_handle;
    }

  float3 getCenter1()
  {
      return make_float3(m_cx1, m_cy1, m_cz1);
   }

  void getTimeVal(float& t1, float& t2)
  {
      t1 = m_t0;
      t2 = m_t1;

  }

private:
  float m_cx0;
  float m_cy0;
  float m_cz0;
  float m_cx1;
  float m_cy1;
  float m_cz1;
  float m_r;
  float m_t0;
  float m_t1;
};

#endif //!IO_MOVING_SPHERE_H
