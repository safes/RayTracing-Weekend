
#include "texture.cuh"
#include "sutil/vec_math.h"
#include "../shaders/sysparameter.h"
//rtDeclareVariable(float, scale, , );
//
//// buffer definitions
//rtBuffer<float3, 1> ranvec;
//rtBuffer<int, 1> perm_x;
//rtBuffer<int, 1> perm_y;
//rtBuffer<int, 1> perm_z;
extern "C" __constant__ SysParamter Parameter;
  __device__  float3* ranvec;
  __device__ int* perm_x;
  __device__  int* perm_y;
  __device__  int* perm_z;


inline __device__ float perlin_interp(float3 c[2][2][2], float u, float v, float w)
 {
	float uu = u * u * (3 - 2 * u);
	float vv = v * v * (3 - 2 * v);
	float ww = w * w * (3 - 2 * w);
	float accum = 0;

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			for (int k = 0; k < 2; k++) {
				float3 weight_v = make_float3(u - i, v - j, w - k);
				accum += (i * uu + (1 - i) * (1 - uu)) *
					(j * vv + (1 - j) * (1 - vv)) *
					(k * ww + (1 - k) * (1 - ww)) * dot(c[i][j][k], weight_v);
			}

	return accum;
}

inline __device__ float noise(float3 p) {
    float u = p.x - floor(p.x);
    float v = p.y - floor(p.y);
    float w = p.z - floor(p.z);

    int i = floor(p.x);
    int j = floor(p.y);
    int k = floor(p.z);
    float3 c[2][2][2];

    for (int di = 0; di < 2; di++)
        for (int dj = 0; dj < 2; dj++)
            for (int dk = 0; dk < 2; dk++)
                c[di][dj][dk] = ranvec[perm_x[(i + di) & 255] ^ perm_y[(j + dj) & 255] ^ perm_z[(k + dk) & 255]];

    return perlin_interp(c, u, v, w);
}

inline __device__ float turb(float3 p)
{
    float accum = 0;
    float3 temp_p = p;
    float weight = 1.0;

    for (int i = 0; i < 7; i++) {
        accum += weight * noise(temp_p);
        weight *= 0.5f;
        temp_p *= 2.f;
    }

    return fabs(accum);
}


extern "C" __device__ float3 __direct_callable__noisetexture(MaterialParams const& matPar, float u, float v, float3 p) {
   // MaterialParams const& matPar = Parameter.MatParams[matIdx];
    textureParam* texPar = (textureParam*)matPar.texArr;
    ranvec = (float3*) texPar->randvec;
    perm_x = (int*) texPar->permX;
    perm_y = (int*) texPar->permY;
    perm_z = (int*) texPar->permZ;

    return make_float3(1.f) * 0.5f * (1.0f + sinf(texPar->scale*p.z + 5.f*turb(texPar->scale*p)));
    //return make_float3(1.f) * 0.5f * (1.0f + sinf(scale*p.x + 5.f*turb(scale*p)));
    //return make_float3(1.f) * 0.5f * (1.0f + sinf(scale*p.x + 10.f*turb(scale*p)));
}
