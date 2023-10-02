#ifndef IO_DIELECTRIC_MATERIAL_H
#define IO_DIELECTRIC_MATERIAL_H

#include "ioMaterial.h"
#include <optix.h>

//extern "C" const char dielectric_material_ptx_c[];

class ioDielectricMaterial : public ioMaterial
{
public:
  ioDielectricMaterial() { }

  ioDielectricMaterial(float eta) : m_eta(eta) { }

virtual void assignTo(MaterialParams& matParam)  override
    {
        matParam.eta = m_eta;
        matParam.matindex = CALLABLE_ID_DIELECTRIC;
        matParam.lightreflectIdx = CALLABLE_ID_LIGHT_SAMPLE_DIELECT_PDF;
    }

private:
  float m_eta;
};

#endif //!IO_DIELECTRIC_MATERIAL_H
