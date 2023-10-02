#include <iostream>
#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <sutil/Exception.h>
#include <sutil/CUDAOutputBuffer.h>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

//#include <optixu/optixpp.h>

#include "Director.h"

// extern "C" const char exception_ptx_c[];
//std::vector<textureParam> ioMaterial::TexArray;

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
        << message << "\n";
}


void Director::init(unsigned int width, unsigned int height,
                    unsigned int samples)
{
    Params = {};
    
    Params.m_Nx = m_Nx = width;
    Params.m_Ny = m_Ny = height;
    Params.numSamples = samples;

    Params.maxRayDepth = 20;
    Params.cameraType = CALLABLE_ID_PINHOLE;
    m_maxRayDepth = 20;
    m_Ns = samples;
    m_sbt = {};
    pipeline_compile_options = {};

    m_denoiser = nullptr;
    m_d_stateDenoiser = 0;
    m_d_scratchDenoiser = 0;
    m_d_denoisedBuffer = 0;

    initContext();
    createModule();
   // m_context->setEntryPointCount(1);
    initRayGenProgram();
    initMissProgram();

    initOutputBuffer();
    initDenoiser();

  //  m_context["sysOutputBuffer"]->set(m_outputBuffer);
}

void Director::destroy()
{
    m_scene.destroy();
      
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
 /*   OPTIX_CHECK(optixProgramGroupDestroy(m_rayGenProgram));
    OPTIX_CHECK(optixProgramGroupDestroy(m_missProgram));
    OPTIX_CHECK(optixProgramGroupDestroy(m_hitRadianceProg));*/
    //
    for (int i = 0; i < NUM_MODULE_IDENTIFIERS; ++i) {
        OPTIX_CHECK(optixModuleDestroy(ptxModules[i]));
    }
    for (int i = 0; i < programGroups.size(); ++i) {
        OPTIX_CHECK(optixProgramGroupDestroy(programGroups[i]));
    }

    CUDA_CHECK(cudaFree((void*)m_d_denoisedBuffer));
    CUDA_CHECK(cudaFree((void*)m_d_scratchDenoiser));
    CUDA_CHECK(cudaFree((void*)m_d_stateDenoiser));
    delete[] hostBuffer;
    OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));

    
    OPTIX_CHECK(optixDeviceContextDestroy(m_context));
    CUDA_CHECK(cudaStreamDestroy(m_custream));
    delete m_outputBuffer;
    m_outputBuffer = nullptr;
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.callablesRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(Params.lights)));
    /*   CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_tri_gas_output_buffer)));
       CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_sphere_gas_output_buffer)));
       CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.accum_buffer)));*/
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
 //   m_outputBuffer.deletePBO();
 //   m_context.destroy();
}

void Director::initContext()
{

    CUDA_CHECK(cudaFree(0));

    OptixDeviceContext context;
    CUcontext          cuCtx = 0;  // zero means take the current context
    CUDA_CHECK(cudaStreamCreate(&m_custream));
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

    m_context = context;

}

void Director::initOutputBuffer()
{
    m_outputBuffer = new sutil::CUDAOutputBuffer<float4>(sutil::CUDAOutputBufferType::ZERO_COPY,
        m_Nx, m_Ny);
    m_outputBuffer->setStream(m_custream);
        // m_context->createBuffer(RT_BUFFER_OUTPUT);
    hostBuffer = new float4[m_Nx * m_Ny];
  
}
static std::vector<char> readData(std::string const& filename)
{
    std::ifstream inputData(filename, std::ios::binary);

    if (inputData.fail())
    {
        std::cerr << "ERROR: readData() Failed to open file " << filename << '\n';
        return std::vector<char>();
    }

    // Copy the input buffer to a char vector.
    std::vector<char> data(std::istreambuf_iterator<char>(inputData), {});

    if (inputData.fail())
    {
        std::cerr << "ERROR: readData() Failed to read file " << filename << '\n';
        return std::vector<char>();
    }

    return data;
}

static const std::string modulePath = "C:\\Developments\\weeker_raytracer\\src\\OptiX_ex\\RestOfLife\\RestOfLife\\RestOfLife\\gpu-code\\";
void Director::createModule()
{
    OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY; // IMPORTANT: if not set to 'ANY', instance traversables will not work
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.numAttributeValues = 8;
    pipeline_compile_options.usesMotionBlur = true;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE; // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "Parameter";

    std::vector<std::string> moduleFiles = { "raygen.cu", "miss.cu", "closehit.cu",
                       "sphere.cu","movingSphere.cu","aarectx.cu","aarecty.cu","aarectz.cu",
        "volumeBox.cu","volumeSphere.cu","camera.cu","diffuseLight.cu",
        "dielectricMaterial.cu","isotropicMaterial.cu",
        "lambertianMaterial.cu","metalmaterial.cu","normalmaterial.cu", 
        "checkeredTexture.cu","constantTexture.cu","imageTexture.cu",
        "noiseTexture.cu","nullTexture.cu", "cosinePdf.cu", "mixtureBiasPdf.cu", 
        "mixturePdf.cu", "rectPdf.cu"};
    std::vector< char> input;
    for (int i = 0; i < moduleFiles.size(); ++i)
    {
        size_t      inputSize = 0;
        input = readData(modulePath + moduleFiles[i] + ".optixir");
        inputSize = input.size();
        OPTIX_CHECK_LOG(optixModuleCreate(
            m_context,
            &module_compile_options,
            &pipeline_compile_options,
            input.data(),
            inputSize,
            LOG, &LOG_SIZE,
            &ptxModules[i]
        ));
    }

        
}



void Director::initRayGenProgram()
{
    OptixProgramGroupOptions program_group_options = {};
    std::vector<OptixProgramGroupDesc> programGroupDescriptions(NUM_PROGRAM_IDENTIFIERS);
    OptixProgramGroupDesc *prog_group_desc = &programGroupDescriptions[PROGRAM_ID_RAYGENERATION];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    prog_group_desc->raygen.module = ptxModules[MODULE_ID_RAYGENERATION];
    prog_group_desc->raygen.entryFunctionName = "__raygen__Program";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_MISS_RADIANCE];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    prog_group_desc->miss.module = ptxModules[MODULE_ID_MISS];
    prog_group_desc->miss.entryFunctionName = "__miss__Program";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
   
    /*prog_group_desc = &programGroupDescriptions[PROGRAM_ID_HIT_RADIANCE];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  
    prog_group_desc->hitgroup.moduleCH = ptxModules[MODULE_ID_CLOSESTHIT];
    prog_group_desc->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;*/
   
    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_IS_PHERE];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
   
    prog_group_desc->hitgroup.moduleCH = ptxModules[MODULE_ID_CLOSESTHIT];
    prog_group_desc->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    prog_group_desc->hitgroup.moduleIS = ptxModules[MODULE_ID_IS_HIT];
    prog_group_desc->hitgroup.entryFunctionNameIS = "__intersection__sphere";
    prog_group_desc->hitgroup.moduleAH = nullptr;
    prog_group_desc->hitgroup.entryFunctionNameAH = nullptr;
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
   
    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_IS_MOVING_PHERE];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    prog_group_desc->hitgroup.moduleCH = ptxModules[MODULE_ID_CLOSESTHIT];
    prog_group_desc->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    prog_group_desc->hitgroup.moduleIS = ptxModules[MODULE_ID_IS_MOVE_HIT];
    prog_group_desc->hitgroup.entryFunctionNameIS = "__intersection__movingsphere";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    prog_group_desc->hitgroup.entryFunctionNameAH = nullptr;
    prog_group_desc->hitgroup.moduleAH = nullptr;
   
    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_IS_AARECT_X];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    prog_group_desc->hitgroup.moduleCH = ptxModules[MODULE_ID_CLOSESTHIT];
    prog_group_desc->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    prog_group_desc->hitgroup.moduleIS = ptxModules[MODULE_ID_IS_AARECT_X];
    prog_group_desc->hitgroup.entryFunctionNameIS = "__intersection__hitRectX";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    prog_group_desc->hitgroup.entryFunctionNameAH = nullptr;
    prog_group_desc->hitgroup.moduleAH = nullptr;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_IS_AARECT_Y];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    prog_group_desc->hitgroup.moduleCH = ptxModules[MODULE_ID_CLOSESTHIT];
    prog_group_desc->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    prog_group_desc->hitgroup.moduleIS = ptxModules[MODULE_ID_IS_AARECT_Y];
    prog_group_desc->hitgroup.entryFunctionNameIS = "__intersection__hitRectY";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    prog_group_desc->hitgroup.entryFunctionNameAH = nullptr;
    prog_group_desc->hitgroup.moduleAH = nullptr;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_IS_AARECT_Z];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    prog_group_desc->hitgroup.moduleCH = ptxModules[MODULE_ID_CLOSESTHIT];
    prog_group_desc->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    prog_group_desc->hitgroup.moduleIS = ptxModules[MODULE_ID_IS_AARECT_Z];
    prog_group_desc->hitgroup.entryFunctionNameIS = "__intersection__hitRectZ";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    prog_group_desc->hitgroup.entryFunctionNameAH = nullptr;
    prog_group_desc->hitgroup.moduleAH = nullptr;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_IS_VOLUMEBOX];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    prog_group_desc->hitgroup.moduleCH = ptxModules[MODULE_ID_CLOSESTHIT];
    prog_group_desc->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    prog_group_desc->hitgroup.moduleIS = ptxModules[MODULE_ID_IS_VOLUMEBOX];
    prog_group_desc->hitgroup.entryFunctionNameIS = "__intersection__hitVolumebox";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    prog_group_desc->hitgroup.entryFunctionNameAH = nullptr;
    prog_group_desc->hitgroup.moduleAH = nullptr;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_IS_VOLUMESPHERE];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    prog_group_desc->hitgroup.moduleCH = ptxModules[MODULE_ID_CLOSESTHIT];
    prog_group_desc->hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    prog_group_desc->hitgroup.moduleIS = ptxModules[MODULE_ID_IS_VOLUMESPHERE];
    prog_group_desc->hitgroup.entryFunctionNameIS = "__intersection__hitVolumesphere";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    prog_group_desc->hitgroup.entryFunctionNameAH = nullptr;
    prog_group_desc->hitgroup.moduleAH = nullptr;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_LENS_PINHOLE];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_LENS_SHADER];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__perspectiveCamera";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    
    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_LIGHT_DIFFUSE];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_LIGHT_SAMPLE];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__diffuselight";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_DIELECTRIC];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_DIELECTRIC_REFLECTION];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__dielectricMat";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_ISOTROPIC];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_ISOTRopic_REFLECTION];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__ISOTROPIC";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_LAMBERTIAN];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_LAMBERTIAN_MATERIAL];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__lambertian";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

   

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_METAL];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_METAL_MATERIAL];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__metal";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_NORMAL];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_NORMAL_MATERIAL];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__normal";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_CHECKEDTEX];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_CHECKEDTEX];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__checkerTexture";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_CONSTANTTEX];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_CONSTANTTEX];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__constantTexture";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_IMAGETEX];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_IMAGETEX];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__imagetexture";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_NOISETEX];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_NOISETEX];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__noisetexture";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_NULLTEX];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_NULLTEX];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__nullTexture";
       
    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_COSINEPDF];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_COSINEPDF];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__cosineGenerate";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_MIXTUREBIASPDF];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_MIXTUREBIAS];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__mixtureBIAS_generate";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_MIXTUREPDF];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_MIXTURE];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__mixture_generate";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_RECTPDF_X];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_RECTPDF];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__rect_x_generate";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_RECTPDF_Y];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_RECTPDF];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__rect_y_generate";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_RECTPDF_Z];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_RECTPDF];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__rect_z_generate";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_COSINEVAL];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_COSINEPDF];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__cosineValue";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_MIXTUREBIASVAL];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_MIXTUREBIAS];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__mixtureBIAS_value";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_MIXTUREVAL];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_MIXTURE];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__mixture_value";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_RECTVAL_X];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_RECTPDF];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__rect_x_value";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_RECTVAL_Y];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_RECTPDF];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__rect_y_value";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_RECTVAL_Z];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_RECTPDF];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__rect_z_value";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_LIGHT_REFLECT_PDF];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_LAMBERTIAN_MATERIAL];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__eval_bsdf_diffuse_reflection";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_LIGHT_REF_DIELECT_PDF];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_DIELECTRIC_REFLECTION];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__eval_bsdf_dielect_reflection";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    prog_group_desc = &programGroupDescriptions[PROGRAM_ID_LIGHT_REF_METAL_PDF];
    prog_group_desc->kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    prog_group_desc->callables.moduleDC = ptxModules[MODULE_ID_METAL_MATERIAL];
    prog_group_desc->callables.entryFunctionNameDC = "__direct_callable__eval_bsdf_metal_reflection";
    prog_group_desc->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    programGroups.resize(NUM_PROGRAM_IDENTIFIERS);
    OPTIX_CHECK_LOG(
        optixProgramGroupCreate(
            m_context,
            programGroupDescriptions.data(),
            NUM_PROGRAM_IDENTIFIERS,                             // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            programGroups.data()
        )
    );
    
    /*m_rayGenProgram = m_context->createProgramFromPTXString(
        raygen_ptx_c, "rayGenProgram");
    m_context->setRayGenerationProgram(0, m_rayGenProgram);
    m_context["numSamples"]->setInt(m_Ns);
    m_context["maxRayDepth"]->setInt(m_maxRayDepth);*/
}

void Director::createLights()
{
    LightDefinition light;
    light.emission = make_float3(15.f);
    light.vecU = make_float3(343.f-213.f,0, 0.f);
    light.vecV = make_float3(0, 0, 332.f - 227.f);
    light.position = make_float3(213.f, 554.f, 227.f);
    light.area = length(cross(light.vecU, light.vecV));
    light.normal = normalize(cross(light.vecU, light.vecV));

    //m_lightDefinitions.push_back(light);
}

void Director::initLaunchParams()
{
 /*   CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&Params.accum_buffer),
        state.params.width * state.params.height * sizeof(float4)
    ));*/

    Params.frame_buffer = nullptr; // Will be set when output buffer is mapped

  //  state.params.subframe_index = 0u;
    Params.handle = m_scene.m_root;
    dynamic_cast<ioPerspectiveCamera*>(m_scene.camera)->getfrustum(Params.cameraOrigin,
        Params.cameraU, Params.cameraV, Params.cameraW, Params.cameraLowerLeftCorner,
        Params.cameraHorizontal, Params.cameraVertical);
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>( &Params.MatParams),
        sizeof(MaterialParams) * m_scene.materialList.size()));

    std::vector<MaterialParams> matparams(m_scene.materialList.size());
    
    for (int i = 0; i < m_scene.materialList.size(); ++i)
    {
        m_scene.materialList[i]->assignTo(matparams[i]);

        if (!m_scene.materialList[i]->TexArray.empty()) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&matparams[i].texArr),
                sizeof(textureParam) * m_scene.materialList[i]->TexArray.size()));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(matparams[i].texArr),
                m_scene.materialList[i]->TexArray.data(),
                sizeof(textureParam) * m_scene.materialList[i]->TexArray.size(),
                ::cudaMemcpyHostToDevice));
        }
    }

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(Params.MatParams),
        matparams.data(), sizeof(MaterialParams) * matparams.size(),
        ::cudaMemcpyHostToDevice));

    Params.cameraTime0 = 0.f;
    Params.cameraTime1 = 1.f;
    Params.skyLight = m_scene.m_lightDefinitions.empty();;
    Params.numLights = m_scene.m_lightDefinitions.size();
    if (Params.numLights > 0) {

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&Params.lights),
            sizeof(LightDefinition) * m_scene.m_lightDefinitions.size()));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(Params.lights), m_scene.m_lightDefinitions.data(),
            sizeof(LightDefinition) * m_scene.m_lightDefinitions.size(), ::cudaMemcpyHostToDevice));

        

        if (m_scene.MCpdf.p0 != nullptr) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&Params.pdf.p0), sizeof(pdfCallfun)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(Params.pdf.p0),
                m_scene.MCpdf.p0, m_scene.MCpdf.p0 ? sizeof(pdfCallfun) : 0, ::cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&Params.pdf.p1), sizeof(pdfCallfun)));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(Params.pdf.p1),
                m_scene.MCpdf.p1, m_scene.MCpdf.p1 ? sizeof(pdfCallfun) : 0, ::cudaMemcpyHostToDevice));

        }
        memcpy(&Params.pdf.pdfrect, &m_scene.MCpdf.pdfrect, sizeof(hitRectData));
        //CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&Params.pdf), sizeof(pdfCallfun)));
        //CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(Params.pdf), &m_scene.MCpdf, sizeof(pdfCallfun),
            //::cudaMemcpyHostToDevice));
        Params.pdf.pdfGenIdx = m_scene.MCpdf.pdfGenIdx;
        Params.pdf.pdfValIdx = m_scene.MCpdf.pdfValIdx;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(SysParamter)));
    
    //state.params.handle = state.ias_handle;
}

void Director::initMissProgram()
{
   /* m_missProgram = m_context->createProgramFromPTXString(
        miss_ptx_c, "missProgram");
    m_context->setMissProgram(0, m_missProgram);*/
}

// void Director::initExceptionProgram()
// {
//   m_missProgram = m_context->createProgramFromPTXString(
//     exception_ptx_c, "exceptionProgram");
//   m_context->setMissProgram(0, m_missProgram);
// }

void Director::createPipeline()
{
  

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;

    OPTIX_CHECK_LOG(optixPipelineCreate(
        m_context,
        &pipeline_compile_options,
        &pipeline_link_options,
        programGroups.data(),
        programGroups.size(),
        LOG, &LOG_SIZE,
        &pipeline
    ));

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stackSizes = {};
    for (int i = 0; i < programGroups.size(); ++i)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroups[i], &stackSizes, 
                    pipeline));

    }
 /*   OPTIX_CHECK(optixUtilAccumulateStackSizes(state.raygen_prog_group, &stackSizes, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.miss_group, &stackSizes, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.sphere_hit_group, &stackSizes, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.tri_hit_group, &stackSizes, state.pipeline));*/

    unsigned int maxTraceDepth = 1;
    unsigned int maxCCDepth = 0;
    unsigned int maxDCDepth = 0;
    unsigned int directCallableStackSizeFromTraversal;
    unsigned int directCallableStackSizeFromState;
    unsigned int continuationStackSize;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stackSizes,
        maxTraceDepth,
        maxCCDepth,
        maxDCDepth,
        &directCallableStackSizeFromTraversal,
        &directCallableStackSizeFromState,
        &continuationStackSize
    ));

    // This is 3 since the largest depth is IAS->MT->GAS
    unsigned int maxTraversalDepth = 3;

    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline,
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState,
        continuationStackSize,
        maxTraversalDepth
    ));
}

void Director::createSBT()
{
    CUdeviceptr   d_raygen_record;
    const size_t  raygen_record_size = sizeof(RecordHeader);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

    RecordHeader rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[PROGRAM_ID_RAYGENERATION], &rg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr   d_miss_records;
    const size_t  miss_record_size = sizeof(RecordHeader);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_records), miss_record_size));

    RecordHeader ms_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[PROGRAM_ID_MISS_RADIANCE], &ms_sbt));
    

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_miss_records),
        &ms_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice
    ));


    std::vector<HitGroupRecord> hitgroup_records;
    size_t hitrecCnt = 0;
      
    
    HitGroupRecord Hit_record;

    //
    // Hit groups
    //
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[PROGRAM_ID_IS_PHERE],
        &Hit_record));
    for (int i = 0; i < m_scene.geometryList.size(); ++i)
    {
        if (nullptr != dynamic_cast<ioSphere*>(m_scene.geometryList[i])) {
            HitGroupRecord hr = {};
            hitgroup_records.push_back(hr);
            memcpy(hitgroup_records[hitrecCnt].header, Hit_record.header, OPTIX_SBT_RECORD_HEADER_SIZE);
            hitgroup_records[hitrecCnt].data.color = make_float3(0.9f, 0.1f, 0.1f);
            hitgroup_records[hitrecCnt].data.center = m_scene.geometryList[i]->getCenter(); //make_float3(-1.0f, -0.5f, 0.0f);
            hitgroup_records[hitrecCnt].data.radius = m_scene.geometryList[i]->getR();//0.5f;
            hitrecCnt++;
        }
        //   (m_scene.geoInstList.size());
    }

    CUdeviceptr   d_hitgroup_records;
    size_t  hitgroup_record_size = sizeof(HitGroupRecord) * m_scene.geometryList.size();
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_hitgroup_records),
        sizeof(HitGroupRecord) * m_scene.geometryList.size()
    ));
    
 /*   CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_hitgroup_records),
        hitgroup_records.data(),
        hitgroup_record_size,
        cudaMemcpyHostToDevice
    ));*/

 
    //hitrecCnt = m_scene.geoInstList.size() - hitrecCnt;
    //std::vector<HitGroupRecord> hitPH_records;
    //const int hitPh_record_size = sizeof(HitGroupRecord) * hitrecCnt; //hitgroup_records.size();
  
    //hitPH_records.resize(hitrecCnt);
    HitGroupRecord Hit_SP_record;
    //
    // Hit groups
    //
    OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[PROGRAM_ID_IS_MOVING_PHERE],
        &Hit_SP_record));
    
    for (int i = 0; i < m_scene.geometryList.size(); ++i)
    {
        if (nullptr != dynamic_cast<ioMovingSphere*>(m_scene.geometryList[i]))
        {
            HitGroupRecord hr = {};
            hitgroup_records.push_back(hr);
            memcpy(hitgroup_records[hitrecCnt].header, Hit_SP_record.header, OPTIX_SBT_RECORD_HEADER_SIZE);
            hitgroup_records[hitrecCnt].data.color = make_float3(0.9f, 0.1f, 0.1f);
            hitgroup_records[hitrecCnt].data.center = m_scene.geometryList[i]->getCenter(); //make_float3(-1.0f, -0.5f, 0.0f);
            hitgroup_records[hitrecCnt].data.radius = m_scene.geometryList[i]->getR();//0.5f;
            hitgroup_records[hitrecCnt].data.center1 = dynamic_cast<ioMovingSphere*>(m_scene.geometryList[i])->getCenter1();
            dynamic_cast<ioMovingSphere*>(m_scene.geometryList[i])->getTimeVal(
                hitgroup_records[hitrecCnt].data.time0, hitgroup_records[hitrecCnt].data.time1);
            hitgroup_records[hitrecCnt].data.bMotion = true;
            hitrecCnt++;
            //memcpy(hitPH_records[ii].header, Hit_SP_record.header, OPTIX_SBT_RECORD_HEADER_SIZE);
            //hitPH_records[ii].data.color = make_float3(0.9f, 0.1f, 0.1f);
            //hitPH_records[ii].data.center = m_scene.geometryList[i]->getCenter(); //make_float3(-1.0f, -0.5f, 0.0f);
            //hitPH_records[ii].data.radius = m_scene.geometryList[i]->getR();//0.5f;
            //
            //hitPH_records[ii].data.center1 = dynamic_cast<ioMovingSphere*>(m_scene.geometryList[i])->getCenter1();
            //dynamic_cast<ioMovingSphere*>(m_scene.geometryList[i])->getTimeVal(
            //        hitPH_records[ii].data.time0, hitPH_records[ii].data.time1);
            //hitPH_records[ii].data.bMotion = true;
            //ii++;
            
        }
    }


 /*   OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[PROGRAM_ID_IS_AARECT_Z],
        &Hit_SP_record));*/
   /* int ii = 0;*/
    for (int i = 0; i < m_scene.geometryList.size(); ++i)
    {
        if (nullptr != dynamic_cast<ioAARect*>(m_scene.geometryList[i]))
        {
            ioAARect* rectp = dynamic_cast<ioAARect*>(m_scene.geometryList[i]);
            switch (rectp->kind) {
            case ::X_AXIS:
                OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[PROGRAM_ID_IS_AARECT_X],
                    &Hit_SP_record));
                break;
            case ::Y_AXIS:
                OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[PROGRAM_ID_IS_AARECT_Y],
                    &Hit_SP_record));
                break;
            case ::Z_AXIS:
                OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[PROGRAM_ID_IS_AARECT_Z],
                    &Hit_SP_record));
                break;
            }
            HitGroupRecord hr = {};
            hitgroup_records.push_back(hr);
            memcpy(hitgroup_records[hitrecCnt].header, Hit_SP_record.header, OPTIX_SBT_RECORD_HEADER_SIZE);
            ioAARect* rect = dynamic_cast<ioAARect*>(m_scene.geometryList[i]);
            //hitgroup_records[hitrecCnt].data.color = make_float3(0.9f, 0.1f, 0.1f);
            //hitgroup_records[hitrecCnt].data.center = m_scene.geometryList[i]->getCenter(); //make_float3(-1.0f, -0.5f, 0.0f);
            //hitgroup_records[hitrecCnt].data.radius = m_scene.geometryList[i]->getR();//0.5f;
            //hitgroup_records[hitrecCnt].data.center1 = dynamic_cast<ioMovingSphere*>(m_scene.geometryList[i])->getCenter1();
            //dynamic_cast<ioMovingSphere*>(m_scene.geometryList[i])->getTimeVal(
            //    hitgroup_records[hitrecCnt].data.time0, hitgroup_records[hitrecCnt].data.time1);
            //hitgroup_records[hitrecCnt].data.bMotion = true;
            hitgroup_records[hitrecCnt].data.rectData.a0 = rect->m_a0;
            hitgroup_records[hitrecCnt].data.rectData.a1 = rect->m_a1;
            hitgroup_records[hitrecCnt].data.rectData.b0 = rect->m_b0;
            hitgroup_records[hitrecCnt].data.rectData.b1 = rect->m_b1;
            hitgroup_records[hitrecCnt].data.rectData.k = rect->m_k;
            hitgroup_records[hitrecCnt].data.rectData.flip = rect->m_flip;
            hitrecCnt++;
            //memcpy(hitPH_records[ii].header, Hit_SP_record.header, OPTIX_SBT_RECORD_HEADER_SIZE);
            //hitPH_records[ii].data.color = make_float3(0.9f, 0.1f, 0.1f);
            //hitPH_records[ii].data.center = m_scene.geometryList[i]->getCenter(); //make_float3(-1.0f, -0.5f, 0.0f);
            //hitPH_records[ii].data.radius = m_scene.geometryList[i]->getR();//0.5f;
            //
            //hitPH_records[ii].data.center1 = dynamic_cast<ioMovingSphere*>(m_scene.geometryList[i])->getCenter1();
            //dynamic_cast<ioMovingSphere*>(m_scene.geometryList[i])->getTimeVal(
            //        hitPH_records[ii].data.time0, hitPH_records[ii].data.time1);
            //hitPH_records[ii].data.bMotion = true;
            //ii++;

        }
    }

    for (int i = 0; i < m_scene.geometryList.size(); ++i)
    {
        if (nullptr != dynamic_cast<ioVolumeBox*>(m_scene.geometryList[i]))
        {
            ioVolumeBox* rectp = dynamic_cast<ioVolumeBox*>(m_scene.geometryList[i]);
            
            OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[PROGRAM_ID_IS_VOLUMEBOX],
                    &Hit_SP_record));
            
            HitGroupRecord hr = {};
            hitgroup_records.push_back(hr);
            memcpy(hitgroup_records[hitrecCnt].header, Hit_SP_record.header, OPTIX_SBT_RECORD_HEADER_SIZE);
            
            
            hitgroup_records[hitrecCnt].data.VolBoxData.boxMax = rectp->m_boxMax;
            hitgroup_records[hitrecCnt].data.VolBoxData.boxMin = rectp->m_boxMin;
            hitgroup_records[hitrecCnt].data.VolBoxData.density = rectp->m_density;
            
            hitrecCnt++;
            

        }
    }

    for (int i = 0; i < m_scene.geometryList.size(); ++i)
    {
        if (nullptr != dynamic_cast<ioVolumeSphere*>(m_scene.geometryList[i]))
        {
            ioVolumeSphere* rectp = dynamic_cast<ioVolumeSphere*>(m_scene.geometryList[i]);

            OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[PROGRAM_ID_IS_VOLUMESPHERE],
                &Hit_SP_record));

            HitGroupRecord hr = {};
            hitgroup_records.push_back(hr);
            memcpy(hitgroup_records[hitrecCnt].header, Hit_SP_record.header, OPTIX_SBT_RECORD_HEADER_SIZE);


            hitgroup_records[hitrecCnt].data.center = make_float3( rectp->m_cx0,rectp->m_cy0,rectp->m_cz0);
            hitgroup_records[hitrecCnt].data.radius = rectp->m_r;
            hitgroup_records[hitrecCnt].data.density = rectp->m_density;

            hitrecCnt++;


        }
    }

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>( d_hitgroup_records),
        hitgroup_records.data(),
        hitgroup_record_size,
        cudaMemcpyHostToDevice
    ));

    std::array<RecordHeader, NUM_CALLABE_TEX_IDS + NUM_CALLABLE_CAMERA +
        NUM_CALLABLE_MAT_IDS + NUM_CALLABLE_PDF_IDS> texRecords;
    
    CUdeviceptr d_texRec;
    
    const size_t DC_record_size = sizeof(RecordHeader) * (NUM_CALLABE_TEX_IDS + NUM_CALLABLE_CAMERA +
        NUM_CALLABLE_MAT_IDS + NUM_CALLABLE_PDF_IDS);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_texRec),
        DC_record_size
    ));
    
    for(int i =0; i < texRecords.size(); ++i)
      OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[PROGRAM_ID_LENS_PINHOLE+i],
        &texRecords[i]));
     
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_texRec),
        texRecords.data(),
        DC_record_size,
        cudaMemcpyHostToDevice
    ));

    m_sbt.raygenRecord = d_raygen_record;
    m_sbt.missRecordBase = d_miss_records;
    m_sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    m_sbt.missRecordCount = 1;
    m_sbt.hitgroupRecordBase = d_hitgroup_records;
    m_sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof(HitGroupRecord));
    m_sbt.hitgroupRecordCount = m_scene.geometryList.size();
    m_sbt.callablesRecordBase = d_texRec;
    m_sbt.callablesRecordCount = NUM_CALLABE_TEX_IDS + NUM_CALLABLE_CAMERA +
        NUM_CALLABLE_MAT_IDS + NUM_CALLABLE_PDF_IDS;
    m_sbt.callablesRecordStrideInBytes = sizeof(RecordHeader);

}

void Director::initDenoiser()
{
    OptixDenoiserOptions optionsDenoiser = {};

    OPTIX_CHECK(optixDenoiserCreate(m_context, OPTIX_DENOISER_MODEL_KIND_LDR, &optionsDenoiser,
        &m_denoiser));
    
    memset(&m_sizesDenoiser, 0, sizeof(OptixDenoiserSizes));
    
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, m_Nx, m_Ny, &m_sizesDenoiser));
    
    m_scratchSizeInBytes = m_sizesDenoiser.withoutOverlapScratchSizeInBytes;

    //CUDA_CHECK(cudaFree((void*)m_d_stateDenoiser));
    CUDA_CHECK(cudaMalloc( (void**) &m_d_stateDenoiser, m_sizesDenoiser.stateSizeInBytes));

    //CUDA_CHECK(cudaFree((void*)m_d_scratchDenoiser));
    CUDA_CHECK(cudaMalloc((void**) & m_d_scratchDenoiser, m_scratchSizeInBytes));

    OPTIX_CHECK(optixDenoiserSetup(m_denoiser, m_custream,
        m_Nx, m_Ny,
        m_d_stateDenoiser, m_sizesDenoiser.stateSizeInBytes,
        m_d_scratchDenoiser, m_scratchSizeInBytes));
    
    m_paramsDenoiser = {};

    CUDA_CHECK(cudaMalloc((void**)&m_paramsDenoiser.hdrIntensity, sizeof(float)));
    
    const size_t sizeBuffers = sizeof(float4) * m_Nx * m_Ny;
    CUDA_CHECK(cudaMalloc((void**)&m_d_denoisedBuffer, sizeBuffers));
    setDenoiserImages();
}

void Director::setDenoiserImages()
{

    m_layer = {};
    m_guideLayer = {};

    // Noisy beauty buffer.
    m_layer.input.data = (CUdeviceptr) m_outputBuffer->map(); // This gets set by the render() function.
    m_layer.input.width = m_Nx;
    m_layer.input.height = m_Ny;

    m_layer.input.rowStrideInBytes = m_Nx * sizeof(float4);
    m_layer.input.pixelStrideInBytes = sizeof(float4);
    m_layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;


    // OptiX 7.3 changed the image handling into input and guide layers.
    // For the HDR dednoiser, the beauty buffer is the only input layer.
    // Optional albedo and normal layers are inside the guide layers now.
    //m_numInputLayers = 1;

    // Denoised output buffer.
    m_layer.output.data = m_d_denoisedBuffer; // If the denoised buffer is GPU local memory, otherwise set in render() after mapping the PBO.
    m_layer.output.width = m_Nx;
    m_layer.output.height = m_Ny;

    m_layer.output.rowStrideInBytes = m_Nx * sizeof(float4);
    m_layer.output.pixelStrideInBytes = sizeof(float4);
    m_layer.output.format = OPTIX_PIXEL_FORMAT_FLOAT4;
}

void Director::createScene(unsigned int sceneNumber)
{

    int error = m_scene.init(m_context, m_Nx, m_Ny, m_Ns, m_maxRayDepth, sceneNumber);
    if (error) {
        int exit_code = EXIT_FAILURE;
        std::exit( exit_code );
    }
    createPipeline();
    createSBT();
    //createLights();
    initLaunchParams();
  //  initOutputBuffer();
   
    if (_verbose) {
        std::string desc = m_scene.getDescription();
        std::cerr << "INFO: Scene description: " << desc << std::endl;
    }
}

void Director::renderFrame()
{
     float4* result_buffer_data = m_outputBuffer->map();
     Params.frame_buffer = result_buffer_data;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params),
        &Params,
        sizeof(SysParamter),
        cudaMemcpyHostToDevice
        
        ));

    optixLaunch(pipeline, m_custream, d_params,
        sizeof(SysParamter), &m_sbt,
        m_Nx, m_Ny, 1);

    m_outputBuffer->unmap();
    
    m_layer.input.data = (CUdeviceptr)m_outputBuffer->map();
    OPTIX_CHECK(optixDenoiserComputeIntensity(m_denoiser, m_custream,
        &m_layer.input, m_paramsDenoiser.hdrIntensity,
        m_d_scratchDenoiser, m_sizesDenoiser.withoutOverlapScratchSizeInBytes));

    m_layer.output.data = m_d_denoisedBuffer;
    OPTIX_CHECK(optixDenoiserInvoke(m_denoiser, m_custream, &m_paramsDenoiser,
        m_d_stateDenoiser, m_sizesDenoiser.stateSizeInBytes,
        &m_guideLayer, &m_layer, 1, 0, 0, // OptiX 7.3 has m_numInputLayers == 1 here.
        m_d_scratchDenoiser, m_scratchSizeInBytes));

    CUDA_CHECK(cudaMemcpy((void*)hostBuffer, (void*)m_d_denoisedBuffer, 
        sizeof(float4) * m_Nx * m_Ny, cudaMemcpyDeviceToHost));

    
 //   CUDA_SYNC_CHECK();
    //m_context->validate();
    //m_context->launch(0,         // Program ID
    //                  m_Nx, m_Ny // Launch Dimensions
    //    );
}

void Director::printPPM()
{
    float4* bufferData = m_outputBuffer->getHostPointer();
    //   Print ppm header
    std::cout << "P3\n" << m_Nx << " " << m_Ny << "\n255\n";
    //   Parse through bufferdata
    for (int j = m_Ny - 1; j >= 0;  j--)
    {
        for (int i = 0; i < m_Nx; i++)
        {
            float4* floatData = ((float4*)hostBuffer) + ((m_Nx*j + i));
            float r = floatData->x;
            float g = floatData->y;
            float b = floatData->z;
            int ir = int(255.99f * clamp(r,0.f,1.f));
            int ig = int(255.99f * clamp(g,0.f,1.f));
            int ib = int(255.99f * clamp(b,0.f,1.f));
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
  // m_outputBuffer->unmap();
}
