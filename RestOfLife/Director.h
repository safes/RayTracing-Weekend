#ifndef DIRECTOR_H
#define DIRECTOR_H

// Include before the optix includes
#include "scene/ioScene.h"
#include "./lib/raydata.cuh"

#include <optix.h>
#include "shaders/sysparameter.h"
#include "shaders/FunctionIdx.h"

struct RecordHeader
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<textureParam>         TexRecord;
//typedef Record<MissData>           MissRecord;
typedef Record<HitGroupData>       HitGroupRecord;

enum ModuleIdentifier
{
    MODULE_ID_RAYGENERATION,
    /*MODULE_ID_EXCEPTION,*/
    MODULE_ID_MISS,
    MODULE_ID_CLOSESTHIT,
    MODULE_ID_IS_HIT,
    MODULE_ID_IS_MOVE_HIT,
    MODULE_ID_IS_AARECT_X,
    MODULE_ID_IS_AARECT_Y,
    MODULE_ID_IS_AARECT_Z,
    MODULE_ID_IS_VOLUMEBOX,
    MODULE_ID_IS_VOLUMESPHERE,
    MODULE_ID_LENS_SHADER,
    MODULE_ID_LIGHT_SAMPLE,
    MODULE_ID_DIELECTRIC_REFLECTION,
    MODULE_ID_ISOTRopic_REFLECTION,
    MODULE_ID_LAMBERTIAN_MATERIAL,
    MODULE_ID_METAL_MATERIAL,
    MODULE_ID_NORMAL_MATERIAL,

    MODULE_ID_CHECKEDTEX,
    MODULE_ID_CONSTANTTEX,
    MODULE_ID_IMAGETEX,
    MODULE_ID_NOISETEX,
    MODULE_ID_NULLTEX,

    MODULE_ID_COSINEPDF,
    MODULE_ID_MIXTUREBIAS,
    MODULE_ID_MIXTURE,
    MODULE_ID_RECTPDF,
    NUM_MODULE_IDENTIFIERS
};

enum ProgramIdentifier
{
    PROGRAM_ID_RAYGENERATION,
    /*PROGRAM_ID_EXCEPTION,*/
    PROGRAM_ID_MISS_RADIANCE,
    /*PROGRAM_ID_MISS_SHADOW,*/
    /*PROGRAM_ID_HIT_RADIANCE,*/
    PROGRAM_ID_IS_PHERE,
    PROGRAM_ID_IS_MOVING_PHERE,
    PROGRAM_ID_IS_AARECT_X,
    PROGRAM_ID_IS_AARECT_Y,
    PROGRAM_ID_IS_AARECT_Z,
    PROGRAM_ID_IS_VOLUMEBOX,
    PROGRAM_ID_IS_VOLUMESPHERE,
    // Callables
    PROGRAM_ID_LENS_PINHOLE,
    /*   PROGRAM_ID_LENS_FISHEYE,
       PROGRAM_ID_LENS_SPHERE,*/
    PROGRAM_ID_LIGHT_DIFFUSE,
    PROGRAM_ID_DIELECTRIC,
    PROGRAM_ID_ISOTROPIC,
    PROGRAM_ID_LAMBERTIAN,
    PROGRAM_ID_METAL,
    PROGRAM_ID_NORMAL,

    PROGRAM_ID_CHECKEDTEX,
    PROGRAM_ID_CONSTANTTEX,
    PROGRAM_ID_IMAGETEX,
    PROGRAM_ID_NOISETEX,
    PROGRAM_ID_NULLTEX,

    PROGRAM_ID_COSINEPDF,
    PROGRAM_ID_MIXTUREBIASPDF,
    PROGRAM_ID_MIXTUREPDF,
    PROGRAM_ID_RECTPDF_X,
    PROGRAM_ID_RECTPDF_Y,
    PROGRAM_ID_RECTPDF_Z,

    PROGRAM_ID_COSINEVAL,
    PROGRAM_ID_MIXTUREBIASVAL,
    PROGRAM_ID_MIXTUREVAL,
    PROGRAM_ID_RECTVAL_X,
    PROGRAM_ID_RECTVAL_Y,
    PROGRAM_ID_RECTVAL_Z,

    PROGRAM_ID_LIGHT_REFLECT_PDF,
    PROGRAM_ID_LIGHT_REF_DIELECT_PDF,
    PROGRAM_ID_LIGHT_REF_METAL_PDF,
    NUM_PROGRAM_IDENTIFIERS
};



class Director
{
public:
    Director(bool verbose, bool debug) : _verbose(verbose), _debug(debug) {}

    void init(unsigned int width, unsigned int height, unsigned int samples);
    void destroy();

    void createScene(unsigned int sceneNumber);
    void renderFrame();
    void printPPM();
    void createModule();

private:
    void createPipeline();
    void createSBT();
    void initLaunchParams();
private:
    int m_Nx;
    int m_Ny;
    int m_Ns;
    int m_maxRayDepth;

    OptixDenoiser       m_denoiser;
    OptixDenoiserSizes  m_sizesDenoiser;
    size_t              m_scratchSizeInBytes;
    OptixDenoiserParams m_paramsDenoiser;
    OptixDenoiserLayer  m_layer;
    CUdeviceptr         m_d_stateDenoiser;
    CUdeviceptr         m_d_scratchDenoiser;
    CUdeviceptr         m_d_denoisedBuffer;
    OptixDenoiserGuideLayer  m_guideLayer ;
    float4              *hostBuffer;
    OptixDeviceContext m_context;
    sutil::CUDAOutputBuffer<float4>  *m_outputBuffer;
    SysParamter Params;
    CUdeviceptr d_params;
    OptixProgramGroup m_rayGenProgram;
    OptixProgramGroup m_missProgram;
    OptixProgramGroup m_colsestHit;
    OptixProgramGroup m_directcallProg[3];
    // optix::Program m_exceptionProgram;
    OptixPipelineCompileOptions pipeline_compile_options;
    OptixPipeline pipeline;
    OptixModule ptxModules[NUM_MODULE_IDENTIFIERS];
    std::vector<OptixProgramGroup> programGroups;
    OptixShaderBindingTable m_sbt;

   
  /*  OptixModule ptx_spheremodule;
    OptixModule ptx_movespheremodule;
    OptixModule ptx_hitmodule;*/
    // Scene Objects
    ioScene m_scene;
    CUstream m_custream;
    void initContext();
    void initOutputBuffer();
    void initRayGenProgram();
    void initMissProgram();
    void initDenoiser();
    void setDenoiserImages();
    // void initExceptionProgram();
    void createLights();
    bool _verbose = false;
    bool _debug = false;
};

#endif //!DIRECTOR_H
