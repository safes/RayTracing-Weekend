#ifndef IO_GEOMETRY_GROUP_H
#define IO_GEOMETRY_GROUP_H

#include <optix.h>
#include "sutil/Aabb.h"
#include "geometry/ioAARect.h"
#include "geometry/ioGeometryInstance.h"
#include <initializer_list>

class ioGeometryGroup
{
public:
  ioGeometryGroup() { }

  void init(OptixDeviceContext& context)
    {
      Context = context;
      //m_gg = context->createGeometryGroup();
      //// NoAccel, Bvh, Sbvh, Trbvh
      ////m_gg->setAcceleration(context->createAcceleration("NoAccel")); // faster than Trbvh with early empty Cornell box
      //m_gg->setAcceleration(context->createAcceleration("Trbvh"));
      //m_gg->setChildCount(0);
    }


    // Utility function - box made of rectangle primitives
   static OptixTraversableHandle createBox( int rectbaseidx, const float3& p0, const float3& p1, ioMaterial* material,
       OptixDeviceContext &context, std::vector<ioGeometry*>& IngeometryList)
   {
        
      //  std::vector<ioGeometryInstance> geoInstList;
     //   std::vector<ioGeometry*> geometryList;
       IngeometryList.push_back(new ioAARect(p0.x, p1.x, p0.y, p1.y, p0.z, true,  Z_AXIS)); // left wall
       IngeometryList.push_back(new ioAARect(p0.x, p1.x, p0.y, p1.y, p1.z, false, Z_AXIS)); // right wall

       IngeometryList.push_back(new ioAARect(p0.x, p1.x, p0.z, p1.z, p0.y, true,  Y_AXIS)); // roof
       IngeometryList.push_back(new ioAARect(p0.x, p1.x, p0.z, p1.z, p1.y, false, Y_AXIS)); // floor

       IngeometryList.push_back(new ioAARect(p0.y, p1.y, p0.z, p1.z, p0.x, true,  X_AXIS)); // back wall
       IngeometryList.push_back(new ioAARect(p0.y, p1.y, p0.z, p1.z, p1.x, false, X_AXIS)); // front wall

    /*    for (auto& rc : geometryList)
        {
            rc->init( context);
        }*/

  /*      OptixAabb aabb;
        sutil::Aabb ab;
        for (int i = 0; i < 6; ++i) {
            ioAARect* rc =(ioAARect*) geometryList[i];;
            switch (rc->kind) {
            case X_AXIS:
                rc->getBoundsX(&aabb);
                break;
            case Y_AXIS:
                rc->getBoundsY(&aabb);
                break;
            case Z_AXIS:
                rc->getBoundsZ(&aabb);
                break;
            }
            ab.include(make_float3(aabb.minX, aabb.minY, aabb.minZ), make_float3(aabb.maxX,
                aabb.maxY, aabb.maxZ));
        }*/
     /*   IngeometryList.insert(IngeometryList.end(), std::initializer_list<ioGeometry*>(&geometryList[0],
            ( & geometryList[0])+6));*/
#if 0
        CUdeviceptr d_aabb_buffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer), sizeof(sutil::Aabb)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_aabb_buffer),
            &ab,
            sizeof(sutil::Aabb),
            cudaMemcpyHostToDevice
        ));

        CUdeviceptr sbtoffsetbuffer;
        static int offsetSbt = 6;
       /* std::array <int, 6> offsetbuffer = { rectbaseidx++, rectbaseidx++,rectbaseidx++,
            rectbaseidx++, rectbaseidx++, rectbaseidx++ };*/
        std::array <int, 6> offsetbuffer = { 0, 1,2,3, 4, 5 };
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbtoffsetbuffer), sizeof(int)*6));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(sbtoffsetbuffer),
           (void*)offsetbuffer.data(),
            sizeof(int) * 6,
            cudaMemcpyHostToDevice
        ));
        
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        uint32_t sphere_input_flag[6] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT ,OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT ,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT ,OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT ,OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT ,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
        
        OptixBuildInput sphere_input = {};
        sphere_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        sphere_input.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
        sphere_input.customPrimitiveArray.numPrimitives = 6;
        sphere_input.customPrimitiveArray.primitiveIndexOffset = rectbaseidx;
        sphere_input.customPrimitiveArray.flags = sphere_input_flag;
        sphere_input.customPrimitiveArray.numSbtRecords = 6;
        sphere_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 4;
        sphere_input.customPrimitiveArray.sbtIndexOffsetBuffer = sbtoffsetbuffer;

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
        OptixTraversableHandle sphere_gas_handle;

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);
        OPTIX_CHECK(optixAccelBuild(context, 0, &accel_options,
            &sphere_input, 1, d_temp_buffer, gas_buffer_sizes.tempSizeInBytes,
            d_buffer_temp_output_gas_and_compacted_size, compactedSizeOffset + 8,
            &sphere_gas_handle, &emitProperty, 1));
#endif 
        
        // init all geometry
     /*   for(int i = 0; i < geometryList.size(); i++) {
            geometryList[i]->init(context);
        }*/
        // GeometryInstance
        //geoInstList.resize(1);
        //for (int i = 0; i < geoInstList.size(); i++)
        //{
        //    //std::cerr << i << std::endl;
        //    geoInstList[i] = ioGeometryInstance();
        //    geoInstList[i].init(context,i,rectbaseidx +i);
        //    geoInstList[i].get().traversableHandle = sphere_gas_handle;

        // //   material->assignTo(geoInstList[i].get(), context);
        //}
        //ioGeometryGroup geometryGroup;
        //geometryGroup.init(context);  // init() sets acceleration
        //for (int i = 0; i < geoInstList.size(); i++)
        //    geometryGroup.addChild(geoInstList[i]);

        //OptixTraversableHandle root = geometryGroup.buildInstanceAccel();
        return NULL;/*sphere_gas_handle;*/

    }


  OptixTraversableHandle buildInstanceAccel()
  {
      const size_t instance_size_in_bytes = sizeof(OptixInstance) * m_gg.size();
      CUdeviceptr  d_instances;
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), instance_size_in_bytes));
      CUDA_CHECK(cudaMemcpy(
          reinterpret_cast<void*>(d_instances),
          m_gg.data(),
          instance_size_in_bytes,
          cudaMemcpyHostToDevice
      ));

      OptixBuildInput instance_input = {};
      instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
      instance_input.instanceArray.instances = d_instances;
      instance_input.instanceArray.numInstances = m_gg.size();

      OptixAccelBuildOptions accel_options = {};
      accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
      accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

      accel_options.motionOptions.numKeys = 2;
      accel_options.motionOptions.timeBegin = 0.0f;
      accel_options.motionOptions.timeEnd = 1.0f;
      accel_options.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;

      OptixAccelBufferSizes ias_buffer_sizes;
      OPTIX_CHECK(optixAccelComputeMemoryUsage(
          Context,
          &accel_options,
          &instance_input,
          1, // num build inputs
          &ias_buffer_sizes
      ));

      CUdeviceptr d_temp_buffer;
      CUDA_CHECK(cudaMalloc(
          reinterpret_cast<void**>(&d_temp_buffer),
          ias_buffer_sizes.tempSizeInBytes
      ));

      
      CUDA_CHECK(cudaMalloc(
          reinterpret_cast<void**>(&d_ias_output_buffer),
          ias_buffer_sizes.outputSizeInBytes
      ));
      
      OPTIX_CHECK(optixAccelBuild(
          Context,
          0,                  // CUDA stream
          &accel_options,
          &instance_input,
          1,                  // num build inputs
          d_temp_buffer,
          ias_buffer_sizes.tempSizeInBytes,
          d_ias_output_buffer,
          ias_buffer_sizes.outputSizeInBytes,
          &ias_handle,
          nullptr,            // emitted property list
          0                   // num emitted properties
      ));

      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_instances)));
      return ias_handle;
  }

  void destroy()
    {
      m_gg.clear();
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ias_output_buffer)));
    }

  void addChild(ioGeometryInstance& gi)
    {
      m_gg.push_back(gi.get());
    }

  std::vector<OptixInstance>& get()
    {
      return m_gg;
    }

private:
  std::vector<OptixInstance> m_gg;
  OptixDeviceContext Context;
  CUdeviceptr d_ias_output_buffer;
  OptixTraversableHandle ias_handle;
};

#endif //!IO_GEOMETRY_GROUP_H
