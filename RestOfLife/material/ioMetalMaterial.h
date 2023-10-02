#ifndef IO_METAL_MATERIAL_H
#define IO_METAL_MATERIAL_H

#include "ioMaterial.h"
#include "../texture/ioTexture.h"

#include <optix.h>


extern "C" const char metal_material_ptx_c[];

class ioMetalMaterial : public ioMaterial
{
public:
    ioMetalMaterial() { fuzz = 0; }

  ioMetalMaterial(const ioTexture *t,  float fuzz) : texture(t), fuzz(fuzz) {}

  virtual void assignTo(MaterialParams& matParam)  override
    {
     // matParam.texindex = texture->texIdx;
      
      matParam.matindex = CALLABLE_ID_METAL;
      TexArray.push_back(texture->getTexRec());
      matParam.lightreflectIdx = CALLABLE_ID_LIGHT_SAMPLE_METAL_PDF;
      //m_mat = context->createMaterial();
      //m_mat->setClosestHitProgram(0, context->createProgramFromPTXString
      //(metal_material_ptx_c, "closestHit"));
      //
      //gi->setMaterialCount(1);
      //gi->setMaterial(/*ray type:*/0, m_mat);
      //texture->assignTo(gi, context);

      if (fuzz < 1.f) {
      matParam.fuzz = fuzz;
      } else {
      matParam.fuzz = 1.f;
      }
    }

private:
   const ioTexture* texture;
   float fuzz;
};

#endif //!IO_METAL_MATERIAL_H
