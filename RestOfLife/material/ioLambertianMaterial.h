#ifndef IO_LAMBERTIAN_MATERIAL_H
#define IO_LAMBERTIAN_MATERIAL_H

#include "ioMaterial.h"
#include "../texture/ioTexture.h"

#include <optix.h>
#include "../shaders/FunctionIdx.h"

extern "C" const char lambertian_material_ptx_c[];

class ioLambertianMaterial : public ioMaterial
{
public:
  ioLambertianMaterial() { }

  ioLambertianMaterial(const ioTexture* t) : texture(t) {}

  virtual void assignTo(MaterialParams& matParam)  override
    {
      TexArray.push_back(texture->getTexRec());
      matParam.matindex = CALLABLE_ID_LAMBERTIAN;
      matParam.lightreflectIdx = CALLABLE_ID_LIGHT_SAMPLE_PDF;
      
      //m_mat = context->createMaterial();
      //m_mat->setClosestHitProgram(0, context->createProgramFromPTXString
      //(lambertian_material_ptx_c, "closestHit"));
      //
      //gi->setMaterialCount(1);
      //gi->setMaterial(/*ray type:*/0, m_mat);
      //texture->assignTo(gi, context);
    }


private:
  const ioTexture* texture;

};

#endif //!IO_LAMBERTIAN_MATERIAL_H
