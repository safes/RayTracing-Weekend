#ifndef IO_TEXTURE_H
#define IO_TEXTURE_H

#include "../../external/rtw_stb_image.h"

#include <random>
#include <iostream>

#include <optix.h>
#include "../shaders/FunctionIdx.h"
#include "vector_types.h"
#include "sutil/vec_math.h"
#include "sutil/CUDAOutputBuffer.h"

extern "C" const char null_texture_ptx_c[];
extern "C" const char constant_texture_ptx_c[];
extern "C" const char checkered_texture_ptx_c[];
extern "C" const char noise_texture_ptx_c[];
extern "C" const char image_texture_ptx_c[];

static __inline__ float localRnd() {
 // static std::random_device rd;  //Will be used to obtain a seed for the rand
  static std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd(
  static std::uniform_real_distribution<float> dis(0.f, 1.f);
  return dis(gen);
}

struct ioTexture {
 //   virtual optix::Program assignTo(optix::GeometryInstance gi, optix::Context &g_context) const = 0;
    virtual textureParam getTexRec() const = 0;
    int texIdx; // directable call function index
};

struct ioNullTexture : public ioTexture {

ioNullTexture() : blank(make_float3(0.f, 0.f, 0.f)) { texIdx = CALLABLE_NULLTEX; }

 /*   virtual optix::Program assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
        optix::Program textProg =  getProgram(g_context);
        textProg["color"]->setFloat(blank);
        gi["sampleTexture"]->setProgramId(textProg);
        return textProg;
    }

    virtual optix::Program  getProgram(optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(null_texture_ptx_c, "sampleTexture");
        return textProg;
    }*/
virtual textureParam getTexRec() const
{
    textureParam par = {};
    par.color = blank;
    par.texCallidx = texIdx;
    return par;
}
    const float3 blank;
};


struct ioConstantTexture : public ioTexture {
    ioConstantTexture(const float3& c) : color(c) {
        texIdx = CALLABLE_CONSTANTTEX;
    }

 /*   virtual optix::Program  assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
        optix::Program textProg =  getProgram(g_context);
        textProg["color"]->setFloat(color);
        gi["sampleTexture"]->setProgramId(textProg);
        return textProg;
    }

    virtual optix::Program  getProgram(optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(constant_texture_ptx_c, "sampleTexture");
        return textProg;
    }*/

    virtual textureParam getTexRec() const
    {
        textureParam par = {};
        par.color = color;
        par.texCallidx = texIdx;
        return par;
    }

    const float3 color;
};


struct ioCheckerTexture : public ioTexture {
ioCheckerTexture(const ioTexture *o, const ioTexture *e) : odd(o), even(e) {
    texIdx = CALLABLE_CHECKEDTEX;
}

  /*  virtual optix::Program  assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
        optix::Program textProg = getProgram(g_context);
        textProg["odd"]->setProgramId(odd->assignTo(gi, g_context));
        textProg["even"]->setProgramId(even->assignTo(gi, g_context));
        gi["sampleTexture"]->setProgramId(textProg);
        return textProg;
    }

    virtual optix::Program getProgram(optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(checkered_texture_ptx_c, "sampleTexture");
        return textProg;
    }*/
virtual textureParam getTexRec() const
{
    textureParam par = {};
    par.odd = odd->texIdx;
    par.even = even->texIdx;
    par.texCallidx = texIdx;
    return par;
}
    const ioTexture* odd;
    const ioTexture* even;
};

struct ioNoiseTexture : public ioTexture {
    ioNoiseTexture(const float s) : scale(s) {
        texIdx = CALLABLE_NOISETEX;
    }

    virtual float3 unit_float3(float x, float y, float z) const {
        float l = sqrtf(x*x + y*y + z*z);
        return make_float3(x/l, y/l, z/l);
    }

    void permute(int *p) const {
        for (int i = 256 - 1; i > 0; i--) {
		    int target = int(localRnd() * (i + 1));
		    int tmp = p[i];

		    p[i] = p[target];
		    p[target] = tmp;
	    }
    }

    void perlin_generate_perm(std::array<int,256>& perm_map) const {
       
      //  std::array<int, 256> perm_map;
       // perm_buffer = new sutil::CUDAOutputBuffer<int>(sutil::CUDAOutputBufferType::ZERO_COPY,
       //      16,16); //g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, 256);
       // int *perm_map = static_cast<int*>(perm_buffer->map());

        for (int i = 0; i < 256; i++)
		    perm_map[i] = i;
        permute(&perm_map[0]);
    //    perm_buffer->unmap();
    }

   /* virtual optix::Program assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
        optix::Buffer ranvec = g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 256);
        float3 *ranvec_map = static_cast<float3*>(ranvec->map());

        for (int i = 0; i < 256; ++i)
            ranvec_map[i] = unit_float3(-1 + 2 * localRnd(), -1 + 2 * localRnd(), -1 + 2 * localRnd());
        ranvec->unmap();

        optix::Buffer perm_x, perm_y, perm_z;
        perlin_generate_perm(perm_x, g_context);
        perlin_generate_perm(perm_y, g_context);
        perlin_generate_perm(perm_z, g_context);

        optix::Program textProg =  getProgram(g_context);
        textProg["ranvec"]->set(ranvec);
        textProg["perm_x"]->set(perm_x);
        textProg["perm_y"]->set(perm_y);
        textProg["perm_z"]->set(perm_z);
        textProg["scale"]->setFloat(scale);

        gi["sampleTexture"]->setProgramId(textProg);

        return textProg;
    }*/

  /*  virtual optix::Program getProgram(optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(noise_texture_ptx_c, "sampleTexture");
        return textProg;
    };*/

    virtual textureParam getTexRec() const
    {
        textureParam par = {};
        par.scale = scale;
        par.texCallidx = texIdx;
        
        //sutil::CUDAOutputBuffer<uchar4> ranvec(sutil //g_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 256);
        std::array< float3, 256> ranvec_map;// = static_cast<float3*>(ranvec->map());

        for (int i = 0; i < 256; ++i)
            ranvec_map[i] = unit_float3(-1 + 2 * localRnd(), -1 + 2 * localRnd(), -1 + 2 * localRnd());
        //ranvec->unmap();

        std::array<int,256> perm_x, perm_y, perm_z;
        perlin_generate_perm(perm_x);
        perlin_generate_perm(perm_y);
        perlin_generate_perm(perm_z);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&par.permX), sizeof(int) * 256));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(par.permX), perm_x.data(), sizeof(int) * 256,
            ::cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&par.permY), sizeof(int) * 256));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(par.permY), perm_y.data(), sizeof(int) * 256,
            ::cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&par.permZ), sizeof(int) * 256));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(par.permZ), perm_z.data(), sizeof(int) * 256,
            ::cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&par.randvec), sizeof(float3) * 256));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(par.randvec), ranvec_map.data(), sizeof(float3) * 256,
            ::cudaMemcpyHostToDevice));


        return par;

    }

    const float scale;
};


struct ioImageTexture : public ioTexture{
    ioImageTexture(const std::string f) : fileName(f) {
        texIdx = CALLABLE_IMAGETEX;
        loadTexture(fileName);
    }

    void loadTexture( const std::string fileName)  {
        int nx, ny, nn;
        unsigned char *tex_data = stbi_load((char*)fileName.c_str(), &nx, &ny, &nn, 0);
        std::cerr << "INFO: image " << fileName << " loaded: (" << nx << 'x' << ny << ") depth: " << nn << std::endl;
        
       /* optix::TextureSampler sampler = context->createTextureSampler();
        sampler->setWrapMode(0, RT_WRAP_REPEAT);
        sampler->setWrapMode(1, RT_WRAP_REPEAT);
        sampler->setWrapMode(2, RT_WRAP_REPEAT);
        sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        sampler->setMaxAnisotropy(1.0f);
        sampler->setMipLevelCount(1u);
        sampler->setArraySize(1u);*/

      //  sutil::CUDAOutputBuffer<unsigned char> buffer(sutil::CUDAOutputBufferType::ZERO_COPY, nx, ny);
      //  unsigned char * data = static_cast<unsigned char *>(buffer.map());
        std::vector<uchar4> h_texture_data(nx * ny);
      //  createTextureImageOnHost(h_texture_data.data(), tex_width, tex_height, mat_index);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                int bindex = (j * nx + i);
                int iindex = ((ny - j - 1) * nx + i) * nn;
                if (false) { // (i==0) {
                    std::cerr << static_cast<unsigned int>(tex_data[iindex + 0]) << ' '
                              << static_cast<unsigned int>(tex_data[iindex + 1]) << ' '
                              << static_cast<unsigned int>(tex_data[iindex + 2]) << '\t' ;
                }

                h_texture_data[bindex].x = tex_data[iindex + 0];
                h_texture_data[bindex].y = tex_data[iindex + 1];
                h_texture_data[bindex].z = tex_data[iindex + 2];

                if(nn == 4)
                    h_texture_data[bindex].w = tex_data[iindex + 3];
                else//3-channel images
                    h_texture_data[bindex].w = (unsigned char)255;
            }
        }

        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        const int tex_width = nx;
        int       tex_height = ny;
        cudaArray_t texture_data;
        int32_t               pitch = nx * 4 * sizeof(unsigned char);
        CUDA_CHECK(cudaMallocArray(&texture_data, &channel_desc, tex_width, tex_height));
        CUDA_CHECK(cudaMemcpy2DToArray(texture_data, 0, 0, h_texture_data.data(),
            pitch, pitch, ny, cudaMemcpyHostToDevice));

        TexObj =  defineTextureOnDevice(0, texture_data, tex_width, tex_height);


     
         
    }

    cudaTextureObject_t  defineTextureOnDevice(int device_idx, cudaArray_t tex_array, int tex_width, int tex_height)
    {
        //CUDA_CHECK(cudaSetDevice(device_idx));

        cudaResourceDesc res_desc;
        std::memset(&res_desc, 0, sizeof(cudaResourceDesc));
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = tex_array;

        cudaTextureDesc tex_desc;
        std::memset(&tex_desc, 0, sizeof(cudaTextureDesc));
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode = cudaFilterModePoint;
        tex_desc.borderColor[0] = 1.0f;
        

        cudaResourceViewDesc* res_view_desc = nullptr;

        cudaTextureObject_t tex;
        CUDA_CHECK(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, res_view_desc));

        return tex;
    }


    /*virtual optix::Program assignTo(optix::GeometryInstance gi, optix::Context &g_context) const override {
        optix::Program textProg =  getProgram(g_context);
        textProg["data"]->setTextureSampler(loadTexture(g_context, fileName));
        gi["sampleTexture"]->setProgramId(textProg);
        return textProg;
    }

    virtual optix::Program getProgram(optix::Context &g_context) const override {
        optix::Program textProg = g_context->createProgramFromPTXString(image_texture_ptx_c, "sampleTexture");
        return textProg;
    }*/

    virtual textureParam getTexRec() const
    {
        textureParam par = {};
        par.texCallidx = texIdx;
        par.imagetex = TexObj;
        return par;
    }

    cudaTextureObject_t TexObj;
    const std::string fileName;
};

#endif
