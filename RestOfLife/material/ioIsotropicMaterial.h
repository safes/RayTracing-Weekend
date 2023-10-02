#ifndef IO_ISOTROPIC_MATERIAL_H
#define IO_ISOTROPIC_MATERIAL_H

#include "ioMaterial.h"
#include "../texture/ioTexture.h"

#include <optix.h>


extern "C" const char isotropic_material_ptx_c[];

class ioIsotropicMaterial : public ioMaterial
{
public:
    ioIsotropicMaterial() { }

    ioIsotropicMaterial(const ioTexture *t) : texture(t) {}

    virtual void assignTo(MaterialParams& matParam)  override 
    {
        matParam.matindex = CALLABLE_ID_ISOTROPIC;
       // matParam.texindex = texture->texIdx;
        TexArray.push_back(texture->getTexRec());
    }

private:
    const ioTexture* texture;
};

#endif //!IO_ISOTROPIC_MATERIAL_H
