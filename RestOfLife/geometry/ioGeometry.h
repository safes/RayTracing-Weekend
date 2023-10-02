#ifndef IO_GEOMETRY_H
#define IO_GEOMETRY_H

#include <optix.h>
#include <array>

class ioGeometry
{
public:
  ioGeometry() {
       sphere_gas_handle = 0;
       sphere_motion_transform_handle = 0;
  }
  virtual ~ioGeometry() {}
  virtual void init(OptixDeviceContext& context) = 0;

  virtual void destroy()
    {
      //m_geo->destroy();
      
    }

 virtual OptixTraversableHandle get()
    {
      return sphere_gas_handle;
    }

  virtual std::array<float,12> getTransform() = 0;
  virtual float3  getCenter() = 0;
  virtual float   getR() = 0;
protected:
  //OptixGeometry m_geo;
  OptixTraversableHandle sphere_gas_handle;
  OptixTraversableHandle sphere_motion_transform_handle;

};

#endif //!IO_GEOMETRY_H
