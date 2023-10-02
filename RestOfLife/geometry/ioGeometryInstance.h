#ifndef IO_GEOMETRY_INSTANCE_H
#define IO_GEOMETRY_INSTANCE_H

#include <optix.h>
#include <vector_types.h>

#include "../geometry/ioGeometry.h"
#include "../geometry/ioSphere.h"
#include "geometry/ioVolumeBox.h"
#include "geometry/ioVolumeSphere.h"
#include "../material/ioMaterial.h"
#include "../texture/ioTexture.h"
#include <algorithm>

class ioGeometryInstance
{
public:
    ioGeometryInstance() { }

    void init(OptixDeviceContext& context,unsigned int id, unsigned int sbtidx)
        {
        m_gi.instanceId = id;
        m_gi.visibilityMask = 255;
        m_gi.sbtOffset = sbtidx ; // This controls the SBT instance offset!
        m_gi.flags = OPTIX_INSTANCE_FLAG_NONE;
        }

    static OptixInstance createSphere(const int insIdx, const int sbtIdx, const float3& p0, const float radius,
                                       ioMaterial* material, OptixDeviceContext &context,
        std::vector<ioGeometry*>& geometryList) {
        ioGeometry* theSphereShape = new ioSphere(p0.x, p0.y, p0.z, radius);
        geometryList.push_back(theSphereShape);
        theSphereShape->init(context);
        ioGeometryInstance gi = ioGeometryInstance();
        gi.init(context,insIdx,sbtIdx);
        gi.setGeometry(*theSphereShape);
        //material->assignTo(gi.get(), context);
        return gi.get();
    }


    static OptixInstance createVolumeBox(const int insIdx,const int sbtIdx, const float3& p0, const float3& p1,
                                                   const float density,
                                                   ioMaterial* material, 
        OptixDeviceContext &context, std::vector<ioGeometry*>& geometryList) {
        ioGeometry* theBoxShape = new ioVolumeBox(p0, p1, density);
        geometryList.push_back(theBoxShape);
        theBoxShape->init(context);
        ioGeometryInstance gi = ioGeometryInstance();
        gi.init(context,insIdx,sbtIdx);
        gi.setGeometry(*theBoxShape);
     //   material->assignTo(gi.get(), context);

        return gi.get();
    }

    static OptixInstance createVolumeSphere(const int insIdx, const int sbtIdx,const float3& p0, const float radius,
                                                      const float density,
                                                      ioMaterial* material, 
        OptixDeviceContext &context, std::vector<ioGeometry*>& geometryList) {
        ioGeometry* theSphereShape = new ioVolumeSphere(p0.x, p0.y, p0.z,
                                                        radius, density);
        theSphereShape->init(context);
        geometryList.push_back(theSphereShape);
       
        ioGeometryInstance gi = ioGeometryInstance();
        gi.init(context,insIdx,sbtIdx);
        gi.setGeometry(*theSphereShape);
      //  material->assignTo(gi.get(), context);

        return gi.get();
    }

    void destroy()
        {
            //m_gi->destroy();
        }

    OptixInstance& get()
        {
            return m_gi;
        }

    void setGeometry(ioGeometry& geo)
        {
            m_gi.traversableHandle = (geo.get());
            std::swap_ranges(&m_gi.transform[0], m_gi.transform + 12, geo.getTransform().begin());
        }

private:
    OptixInstance m_gi;
};

#endif //!IO_GEOMETRY_INSTANCE_H
