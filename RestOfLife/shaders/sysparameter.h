#pragma once
#include <vector_types.h>
#include "../lib/raydata.cuh"

struct MaterialParams {
	int matindex;  // for call fun idx
	int lightreflectIdx;
	textureParam* texArr;
	
	union{
	float fuzz;
	float eta;
	};
};

struct pdfCallfun {
	int    pdfGenIdx;
	int    pdfValIdx;
   /* float bias;
	hitRectData pdfrect;*/
	
    union  {
        float bias;
		hitRectData pdfrect;
		
	};
	pdfCallfun* p0;
	pdfCallfun* p1;
		
};

struct SysParamter {

	OptixTraversableHandle handle;
	int m_Nx;
	int m_Ny;
	float4* frame_buffer;
	int maxRayDepth;
	int numSamples;
	LightDefinition* lights;
	int numLights;
	int	skyLight;
	pdfCallfun pdf;
	float3 cameraOrigin;
	float3 cameraU;
	float3 cameraV;
	float3 cameraW;
	float cameraTime0;
	float cameraTime1;
	float cameraLensRadius;

	float3 cameraLowerLeftCorner;
    float3 cameraHorizontal;
	float3 cameraVertical;

	int cameraType;
	
	MaterialParams* MatParams;
   
};