#include <optix.h>
#include "pdf.cuh"
#include "../shaders/sysparameter.h"
#include "../shaders/FunctionIdx.h"
#include <cuda/random.h>
//rtDeclareVariable(rtCallableProgramId<float(pdf_in&)>, p0_value, , );
//rtDeclareVariable(rtCallableProgramId<float(pdf_in&)>, p1_value, , );
extern "C" __constant__ SysParamter Parameter;

extern "C" __device__ float __direct_callable__mixture_value(pdf_in &in) {
    pdfCallfun* p0 = (pdfCallfun*)Parameter.pdf.p0;
    int p0_value = p0->pdfValIdx + NUM_CALLABLE_CAMERA +
        NUM_CALLABE_TEX_IDS + NUM_CALLABLE_MAT_IDS;
    pdfCallfun* p1 = (pdfCallfun*)Parameter.pdf.p1;
    int p1_value = p1->pdfValIdx + NUM_CALLABLE_CAMERA +
        NUM_CALLABE_TEX_IDS + NUM_CALLABLE_MAT_IDS;
    //float val0 = optixDirectCall<float, pdf_in&>(p0_value, in);
    float val1 = optixDirectCall<float, pdf_in&>(p1_value, in);
    return /*0.5f * val0 + 0.5f * */val1;
}

//rtDeclareVariable(rtCallableProgramId<float3(pdf_in&, uint32_t&)>, p0_generate, , );
//rtDeclareVariable(rtCallableProgramId<float3(pdf_in&, uint32_t&)>, p1_generate, , );

extern "C" __device__ LightSample __direct_callable__mixture_generate(LightDefinition const& light,
    hitRectData const& pdfrect, pdf_in &in, uint32_t& seed) {
    pdfCallfun* p0 = (pdfCallfun*)Parameter.pdf.p0;
    int p0_generate = p0->pdfGenIdx + NUM_CALLABLE_CAMERA +
        NUM_CALLABE_TEX_IDS + NUM_CALLABLE_MAT_IDS;
    pdfCallfun* p1 = (pdfCallfun*)Parameter.pdf.p1;
    int p1_generate = p1->pdfGenIdx + NUM_CALLABLE_CAMERA +
        NUM_CALLABE_TEX_IDS + NUM_CALLABLE_MAT_IDS;
 /*   if (rnd(seed) < 0.5f)
        return optixDirectCall<float3, pdf_in&, uint32_t&>(p0_generate, in, seed);
    else*/
        return optixDirectCall<LightSample,LightDefinition const& , hitRectData const& ,
            pdf_in&, uint32_t&>(p1_generate,light,Parameter.pdf.p1->pdfrect, in, seed);
}
