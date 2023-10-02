#ifndef IO_MATERIAL_H
#define IO_MATERIAL_H

#include <optix.h>
#include "../lib/raydata.cuh"
#include "../shaders/sysparameter.h"
#include "../texture/ioTexture.h"
#include <vector>
#include "../shaders/FunctionIdx.h"

class ioMaterial
{
public:
    ioMaterial() { }

    virtual void destroy() {
        // in case materials are re-used, only need to be destroyed once
   /*     if (m_mat) {
            m_mat->destroy();
            m_mat = nullptr;
        }*/
    }

 /*   optix::Material get() {
        return m_mat;
    }*/

    virtual void assignTo( MaterialParams &matParam) = 0;
     std::vector<textureParam> TexArray;
protected:
 //   OptixMaterial m_mat;
    float3 albedo;
};

#endif //!IO_MATERIAL_H
