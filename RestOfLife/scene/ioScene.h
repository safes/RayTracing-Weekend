#ifndef IO_SCENE_H
#define IO_SCENE_H

#include <vector>
#include <iostream>
#include <optix.h>

//#include "geometry/ioGroup.h"
#include "../geometry/ioTransform.h"
#include "../geometry/ioGeometryGroup.h"
#include "../geometry/ioGeometryInstance.h"
#include "../geometry/ioGeometry.h"

#include "../geometry/ioSphere.h"
#include "../geometry/ioMovingSphere.h"
#include "../geometry/ioAARect.h"
//#include "geometry/ioVolumeBox.h"
//#include "geometry/ioVolumeSphere.h"

//#include "../pdf/ioPdf.h"

#include "../texture/ioTexture.h"
#include "../material/ioNormalMaterial.h"
#include "../material/ioLambertianMaterial.h"
#include "../material/ioMetalMaterial.h"
#include "../material/ioDielectricMaterial.h"
#include "../material/ioDiffuseLightMaterial.h"
#include "../material/ioIsotropicMaterial.h"

#include "scene/ioCamera.h"

// needed for randf()
#include "../lib/random.cuh"
struct pdfCallfun_host {
    int    pdfGenIdx;
    int    pdfValIdx;
    
    union {
        float bias;
        hitRectData pdfrect;
        
    };
    pdfCallfun_host *p0;
    pdfCallfun_host *p1;
};

class ioScene
{
public:
    ioScene() { }

    int init(OptixDeviceContext& context, int Nx, int Ny, int Ns, int maxRayDepth, int Nscene) {

        m_Nx = Nx;
        m_Ny = Ny;
        m_numSamples = Ns;
        m_maxRayDepth = maxRayDepth;


        // PDFs now set per scene
        getScenePdf(Nscene);
        /*if (thePdf) {
            initRayGenProgram(context, thePdf);
        } else {
            initRayGenProgram(context);
        }*/
        //initMissProgram(context);

        //topGroup.init(context);
        //topGroup.get()->setAcceleration(context->createAcceleration("Trbvh"));

        //optix::Group world;

        switch(Nscene){
        case 0:
            CornellBox(context, Nx, Ny);
            break;
        case 1:
            MovingSpheres(context, Nx, Ny);
            break;
        case 2:
            InOneWeekendLight(context, Nx, Ny);
            break;
        case 3:
             VolumesCornellBox(context, Nx, Ny);
            break;
        case 4:
            TheNextWeekFinal(context, Nx, Ny);
            break;
        default:
            std::cerr << "ERROR: Scene " << Nscene << " unknown." << std::endl;
            return 1;
        }

        // created in each scene
        camera->init(context);

        // Setting World Variable
       // context["sysWorld"]->set(world);
        return 0;
    }

    void getScenePdf(int Nscene) {
        //pdfCallfun_host PDF = {};

        switch(Nscene){
        case 0:
         //   PDF = new ioMixturePDF (new ioCosinePDF(), new ioRectY_PDF(213.f, 343.f, 227.f, 332.f, 554.f));
            MCpdf = { CALLABLE_ID_mixture_generate,CALLABLE_ID_mixture_value };
            MCpdf.p0 = new pdfCallfun_host{ CALLABLE_ID_cosineGenerate,CALLABLE_ID_cosineValue };
            MCpdf.p1 = new pdfCallfun_host{ CALLABLE_ID_rect_Y_generate,CALLABLE_ID_rect_y_value };
            MCpdf.p1->pdfrect = {213.f, 343.f, 227.f, 332.f, 554.9f} ;
            
            break;
        case 1:
            MCpdf = { CALLABLE_ID_cosineGenerate,CALLABLE_ID_cosineValue };// new ioCosinePDF();
            break;
        case 2:
            //PDF = new ioMixtureBiasPDF (new ioCosinePDF(), new ioRectZ_PDF(3.f, 5.f, 2.3f, 3.f+3.f, -2.0f), 0.15f);
            //PDF = new ioMixtureBiasPDF (new ioCosinePDF(), new ioRectZ_PDF(3.f, 5.f, 2.3f, 3.f+3.f, -2.0f), 0.95f);
         //   PDF = new ioMixturePDF (new ioCosinePDF(), new ioRectZ_PDF(3.f, 5.f, 2.3f, 3.f+3.f, -2.0f));
            MCpdf = { CALLABLE_ID_mixture_generate,CALLABLE_ID_mixture_value };
            MCpdf.p0 = new pdfCallfun_host{ CALLABLE_ID_cosineGenerate,CALLABLE_ID_cosineValue };
            MCpdf.p1 = new pdfCallfun_host{ CALLABLE_ID_rect_Z_generate,CALLABLE_ID_rect_z_value };
            MCpdf.p1->pdfrect = { 3.f, 5.f, 2.3f, 3.f + 3.f, -2.0f };
            
            break;
        case 3:
            //PDF = new ioMixturePDF (new ioCosinePDF(), new ioRectY_PDF(213.f, 343.f, 227.f, 332.f, 554.f));
            MCpdf = { CALLABLE_ID_mixture_generate,CALLABLE_ID_mixture_value };
            MCpdf.p0 = new pdfCallfun_host{ CALLABLE_ID_cosineGenerate,CALLABLE_ID_cosineValue };
            MCpdf.p1 = new pdfCallfun_host{ CALLABLE_ID_rect_Y_generate,CALLABLE_ID_rect_y_value };
            MCpdf.p1->pdfrect = { 213.f, 343.f, 227.f, 332.f, 554.f };

            break;
        case 4:
          //  PDF = new ioMixturePDF (new ioCosinePDF(), new ioRectY_PDF(123.f, 423.f, 147.f, 412.f, 554.f));
            MCpdf = { CALLABLE_ID_mixture_generate,CALLABLE_ID_mixture_value };
            MCpdf.p0 = new pdfCallfun_host{ CALLABLE_ID_cosineGenerate,CALLABLE_ID_cosineValue };
            MCpdf.p1 = new pdfCallfun_host{ CALLABLE_ID_rect_Y_generate,CALLABLE_ID_rect_y_value };
            MCpdf.p1->pdfrect = { 123.f, 423.f, 147.f, 412.f, 554.f };

            break;
        default:
            std::cerr << "ERROR: Scene " << Nscene << " unknown." << std::endl;
        }
        //return PDF;
    }


 /*   void initRayGenProgram(optix::Context& context) {
        m_rayGenProgram = context->createProgramFromPTXString(
            raygen_ptx_c, "rayGenProgram");
        context->setRayGenerationProgram(0, m_rayGenProgram);
        context->setEntryPointCount(1);
        context["numSamples"]->setInt(m_numSamples);
        context["maxRayDepth"]->setInt(m_maxRayDepth);
    }

    void initRayGenProgram(optix::Context& context, ioPdf* pdf) {
        m_rayGenProgram = context->createProgramFromPTXString(
            raygen_ptx_c, "rayGenProgram");
        context->setEntryPointCount(1);
        m_rayGenProgram["generate"]->setProgramId(pdf->assignGenerate(context));
        m_rayGenProgram["value"]->setProgramId(pdf->assignValue(context));

        context->setRayGenerationProgram(0, m_rayGenProgram);
        context["numSamples"]->setInt(m_numSamples);
        context["maxRayDepth"]->setInt(m_maxRayDepth);
    }*/


  /*  void initMissProgram(optix::Context& context) {
        m_missProgram = context->createProgramFromPTXString(
            miss_ptx_c, "missProgram");
        context->setMissProgram(0, m_missProgram);
    }*/


    void MovingSpheres(OptixDeviceContext& context, int Nx, int Ny) {
        sceneDescription = "InOneWeekend final scene with moving spheres";

        ioTexture *fiftyPercentGrey = new ioConstantTexture(make_float3(0.5f, 0.5f, 0.5f));
        ioTexture *fiftyPercentReddishGrey = new ioConstantTexture(make_float3(0.7f, 0.6f, 0.5f));
        ioTexture *reddish = new ioConstantTexture(make_float3(0.4f, 0.2f, 0.1f));

        // Big Sphere
        geometryList.push_back(new ioSphere(0.0f, -1000.0f, 0.0, 1000.0f));
        materialList.push_back(new ioLambertianMaterial(fiftyPercentGrey));

        // Medium Spheres
        geometryList.push_back(new ioSphere(0.0f, 1.0f, 0.0, 1.0f));
        geometryList.push_back(new ioSphere(-4.0f, 1.0f, 0.0, 1.0f));
        geometryList.push_back(new ioSphere(4.0f, 1.0f, 0.0, 1.0f));

        materialList.push_back(new ioDielectricMaterial(1.5f));
        materialList.push_back(new ioLambertianMaterial(reddish));
        materialList.push_back(new ioMetalMaterial(fiftyPercentReddishGrey, 0.1f));

        // Small Spheres
        uint32_t seed = 0x314759;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float chooseMat = randf(seed);
                float x = a + 0.8f*randf(seed);
                float y = 0.2f;
                float z = b + 0.9f*randf(seed);
                float z_squared = (z)*(z);
                float dist = sqrtf(
                    (x-4.0f)*(x-4.0f) +
                    //(y-0.2f)*(y-0.2f) +
                    z_squared
                    );

                // keep out area near medium spheres
                if ((dist > 0.9f) ||
                    ((z_squared > 0.7f) && ((x*x - 16.0f) > -2.f)))
                {
                    if (chooseMat < 0.70f)
                    {
                        geometryList.push_back(new ioMovingSphere(x,y,z,
                                                                  x, y + 0.18f ,z,
                                                                  0.2f,
                                                                  0.f, 1.f));
                        materialList.push_back(new ioLambertianMaterial(
                                                   new ioConstantTexture(make_float3(randf(seed), randf(seed), randf(seed))
                                                       )));
                    }
                    else if (chooseMat < 0.85f)
                    {
                        geometryList.push_back(new ioSphere(x,y,z, 0.2f));
                        materialList.push_back(new ioMetalMaterial(
                                                   new ioConstantTexture(make_float3(0.5f*(1.0f-randf(seed)),
                                                                                     0.5f*(1.0f-randf(seed)),
                                                                                     0.5f*(1.0f-randf(seed)))),
                                                   0.5f*randf(seed))
                            );
                    }
                    else if (chooseMat < 0.93f)
                    {
                        geometryList.push_back(new ioSphere(x,y,z, 0.2f));
                        materialList.push_back(new ioDielectricMaterial(1.5f));
                    }
                    else
                    {
                        geometryList.push_back(new ioSphere(x,y,z, 0.2f));
                        materialList.push_back(new ioDielectricMaterial(1.5f));
                        geometryList.push_back(new ioSphere(x,y,z, (0.2f-0.007f)));
                        materialList.push_back(new ioDielectricMaterial(1.5f));
                    }
                }
            }
        }

        // init all geometry
        for(int i = 0; i < geometryList.size(); i++)
            geometryList[i]->init(context);

        // GeometryInstance
        geoInstList.resize(geometryList.size());
        int idxsp = 0, idxmsp = 0;
        // Taking advantage of geometryList.size == materialList.size
        for (int i = 0; i < geoInstList.size(); i++)
        {
            geoInstList[i] = ioGeometryInstance();
            if (dynamic_cast<ioMovingSphere*>(geometryList[i])) {
                
                geoInstList[i].init(context,i, idxmsp);
                idxmsp++;
            }
            else {
                
                geoInstList[i].init(context,i, idxsp);
                idxsp++;
            }
            
            geoInstList[i].setGeometry(*geometryList[i]);
          //  materialList[i]->assignTo(geoInstList[i].get(), context);
        }

        for (int i = 0; i < geoInstList.size(); i++) {
            if (dynamic_cast<ioMovingSphere*>(geometryList[i])) {
                geoInstList[i].get().sbtOffset += idxsp;
            }
        }
        // World & Acceleration
        geometryGroup.init(context);
        for (int i = 0; i < geoInstList.size(); i++)
            geometryGroup.addChild(geoInstList[i]);

        
       m_root = geometryGroup.buildInstanceAccel();
       // topGroup.addChild(geometryGroup.get(), context);

        // Create and Init our scene camera
        camera = new ioPerspectiveCamera(
            13.0f, 2.0f, 3.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            20.0f, float(Nx) / float(Ny),
            /*aperture*/0.1f,
            /*focus_distance*/10.f,
            /* t0 and t1 */0.f, 1.f
            );

        // Do not have a light in scene
      //  context["skyLight"]->setInt(true);

    }



    void  InOneWeekendLight(OptixDeviceContext& context, int Nx, int Ny)  {

        sceneDescription = "IOW Scene with a light box";

        //ioTexture *fiftyPercentGrey = new ioConstantTexture(make_float3(0.5f, 0.5f, 0.5f));
        ioTexture *constantGrey = new ioConstantTexture(make_float3(0.7f, 0.7f, 0.7f));
        // ioTexture *constantGreen = new ioConstantTexture(make_float3(0.2f, 0.3f, 0.1f));
        // ioTexture *constantPurple = new ioConstantTexture(make_float3(0.4f, 0.2f, 0.9f));
        ioTexture *saddleBrown = new ioConstantTexture(make_float3(139/255.f,  69/255.f,  19/255.f));
        ioTexture *reallyDarkGrey = new ioConstantTexture(make_float3( 9/255.f,  9/255.f,  9/255.f));
        ioTexture *black = new ioConstantTexture(make_float3( 0/255.f,  0/255.f,  0/255.f));

        ioTexture *checkered = new ioCheckerTexture(reallyDarkGrey, saddleBrown);
        ioTexture *noise4 = new ioNoiseTexture(4.f);
        ioTexture *noise1 = new ioNoiseTexture(1.f);

        ioTexture *earthGlobeImage = new ioImageTexture("assets/earthmap.jpg");

        ioTexture* light4 =  new ioConstantTexture(make_float3(4.f, 4.f, 4.f));
        ioTexture* light8 =  new ioConstantTexture(make_float3(8.f, 8.f, 8.f));
        ioTexture* light25 =  new ioConstantTexture(make_float3(25.f, 25.f, 25.f));
        ioTexture* light16 =  new ioConstantTexture(make_float3(16.f, 16.f, 16.f));

        // Big Sphere
        geometryList.push_back(new ioSphere(0.0f, -1000.0f, 0.0, 1000.0f));
        //materialList.push_back(new ioLambertianMaterial(noise4));
        materialList.push_back(new ioLambertianMaterial(noise1));
        //materialList.push_back(new ioLambertianMaterial(checkered));

        // Medium Spheres
        geometryList.push_back(new ioSphere(-4.0f, 1.0f, 0.0, 1.0f));
        geometryList.push_back(new ioSphere(0.0f, 1.0f, 0.0, 1.0f));
        geometryList.push_back(new ioSphere(4.0f, 1.0f, 0.0, 1.0f));

        materialList.push_back(new ioMetalMaterial(constantGrey, 0.4f));
        materialList.push_back(new ioLambertianMaterial(earthGlobeImage));
        materialList.push_back(new ioDielectricMaterial(1.5f));

        geometryList.push_back(new ioAARect(3.f, 5.f, 1.f, 3.f, -2.0f,false,  Z_AXIS));
        materialList.push_back(new ioDiffuseLightMaterial(light16));

        LightDefinition light;
        light.emission = make_float3(16.f, 16.f, 16.f);
        light.vecU = make_float3(5.f - 3.f, 0, 0.f);
        light.vecV = make_float3(0, 3.f-1.f, 0.f);
        light.position = make_float3(3.f, 1.f, -2.f);
        light.area = length(cross(light.vecU, light.vecV));
        light.normal = normalize(cross(light.vecU, light.vecV));
        m_lightDefinitions.push_back(light);

        // geometryList.push_back(new ioAARect(-6.f, -1.f, -2.f, 2.f, 10.f,  Y_AXIS));
        // materialList.push_back(new ioDiffuseLightMaterial(light4));

        // Small Spheres
        uint32_t seed = 0x6314759;
        for (int a = -11; a < 11; a++)
        {
            for (int b = -11; b < 11; b++)
            {
                float chooseMat = randf(seed);
                float x = a + 0.8f*randf(seed);
                float y = 0.2f;
                float z = b + 0.9f*randf(seed);
                float z_squared = (z)*(z);
                float dist = sqrtf(
                    (x-4.0f)*(x-4.0f) +
                    //(y-0.2f)*(y-0.2f) +
                    z_squared
                    );

                // keep out area near medium spheres
                if ((dist > 0.9f) ||
                    ((z_squared > 0.7f) && ((x*x - 16.0f) > -2.f)))
                {
                    if (chooseMat < 0.70f)
                    {
                        geometryList.push_back(new ioSphere(x,y,z, 0.2f));
                        materialList.push_back(new ioLambertianMaterial(
                                                   new ioConstantTexture(make_float3(randf(seed), randf(seed), randf(seed))
                                                       )));
                    }
                    else if (chooseMat < 0.85f)
                    {
                        geometryList.push_back(new ioSphere(x,y,z, 0.2f));
                        materialList.push_back(new ioMetalMaterial(
                                                   new ioConstantTexture(make_float3(0.5f*(1.0f-randf(seed)),
                                                                                     0.5f*(1.0f-randf(seed)),
                                                                                     0.5f*(1.0f-randf(seed)))),
                                                   0.5f*randf(seed)));
                    }
                    else if (chooseMat < 0.93f)
                    {
                        geometryList.push_back(new ioSphere(x,y,z, 0.2f));
                        materialList.push_back(new ioDielectricMaterial(1.5f));
                    }
                    else
                    {
                        geometryList.push_back(new ioSphere(x,y,z, 0.2f));
                        materialList.push_back(new ioDielectricMaterial(1.5f));
                        geometryList.push_back(new ioSphere(x,y,z, (0.2f-0.007f)));
                        materialList.push_back(new ioDielectricMaterial(1.5f));
                    }
                }
            }
        }

        // init all geometry
        for(int i = 0; i < geometryList.size(); i++) {
            geometryList[i]->init(context);
        }
        // GeometryInstance
        geoInstList.resize(geometryList.size());
        int idxsp = 0, idxmsp = 0, idxrect =0;
        // Taking advantage of geometryList.size == materialList.size
        for (int i = 0; i < geoInstList.size(); i++)
        {
            geoInstList[i] = ioGeometryInstance();
            if (dynamic_cast<ioMovingSphere*>(geometryList[i])) {

                geoInstList[i].init(context, i, idxmsp);
                idxmsp++;
            }
            else if (dynamic_cast<ioAARect*>(geometryList[i])) {
                geoInstList[i].init(context, i, idxrect);
                idxrect++;
            }
            else{

                geoInstList[i].init(context, i, idxsp);
                idxsp++;
            }

            geoInstList[i].setGeometry(*geometryList[i]);
            //  materialList[i]->assignTo(geoInstList[i].get(), context);
        }

        for (int i = 0; i < geoInstList.size(); i++) {
            if (dynamic_cast<ioMovingSphere*>(geometryList[i])) {
                geoInstList[i].get().sbtOffset += idxsp;
            }
            else if (dynamic_cast<ioAARect*>(geometryList[i])) {
                geoInstList[i].get().sbtOffset += (idxsp + idxmsp);
            }
        }
        // Taking advantage of geometryList.size == materialList.size
        //for (int i = 0; i < geoInstList.size(); i++)
        //{
        //    //std::cerr << i << std::endl;
        //    geoInstList[i] = ioGeometryInstance();
        //    geoInstList[i].init(context,i,);
        //    geoInstList[i].setGeometry(*geometryList[i]);
        //    materialList[i]->assignTo(geoInstList[i].get(), context);
        //}

        // World & Acceleration
        geometryGroup.init(context);  // init() sets acceleration
        for (int i = 0; i < geoInstList.size(); i++)
            geometryGroup.addChild(geoInstList[i]);

        m_root = geometryGroup.buildInstanceAccel();

        // Create and Init our scene camera
        camera = new ioPerspectiveCamera(
            13.0f, 2.0f, 3.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            20.0f, float(Nx) / float(Ny),
            /*aperture*/0.08f,
            /*focus_distance*/10.f
            );

        // Have a light in scene
      /*  context["skyLight"]->setInt(false);

        return topGroup.get();*/

    }

    void CornellBox(OptixDeviceContext& context, int Nx, int Ny) {

        sceneDescription = "Cornell box";

        ioMaterial *wallRed = new ioLambertianMaterial(new ioConstantTexture(make_float3(0.65f, 0.05f, 0.05f)));
        ioMaterial *wallGreen = new ioLambertianMaterial(new ioConstantTexture(make_float3(0.12f, 0.45f, 0.15f)));
        ioMaterial *wallWhite = new ioLambertianMaterial(new ioConstantTexture(make_float3(0.73f, 0.73f, 0.73f)));
        ioMaterial *aluminum = new ioMetalMaterial(new ioConstantTexture(make_float3(0.91, 0.92, 0.92)), 0.018);
        ioTexture  *light15 =  new ioConstantTexture(make_float3(15.f, 15.f, 15.f));
        // medium glass sphere
        geometryList.push_back(new ioSphere(190.f, 90.f, 190.f, 90.f));
        materialList.push_back(new ioDielectricMaterial(1.5f));
        
        geometryList.push_back(new ioAARect(0.f, 555.f, 0.f, 555.f, 555.f,  true, X_AXIS)); // left wall
        geometryList.push_back(new ioAARect(0.f, 555.f, 0.f, 555.f, 0.f,   false, X_AXIS)); // right wall
        geometryList.push_back(new ioAARect(0.f, 555.f, 0.f, 555.f, 555.f,  true, Y_AXIS)); // roof
        geometryList.push_back(new ioAARect(0.f, 555.f, 0.f, 555.f, 0.f,   false, Y_AXIS)); // floor
        geometryList.push_back(new ioAARect(0.f, 555.f, 0.f, 555.f, 555.f,  true, Z_AXIS)); // back wall

        geometryList.push_back(new ioAARect(213.f, 343.f, 227.f, 332.f, 554.9f, true, Y_AXIS)); // light

        materialList.push_back(wallGreen);
        materialList.push_back(wallRed );
        materialList.push_back(wallWhite);
        materialList.push_back(wallWhite);
        materialList.push_back(wallWhite);
        materialList.push_back(new ioDiffuseLightMaterial(light15));

        // place some objects in the box

        // // bigger Sphere
        // geometryList.push_back(new ioSphere(365.f, 165.f, 295.f, 165.f));
        // materialList.push_back(wallWhite);

        // // small sphere
        // geometryList.push_back(new ioSphere(185.f, 75.f, 155.f, 75.f));
        // materialList.push_back(wallWhite);
        // materialList.push_back(new ioDielectricMaterial(1.5f));

        
 // init all geometry
       
        // boxes
        const float z1Theta = -15.5f * (CUDART_PI_F/180.f);
        float3 b1size = make_float3(165.f, 330.f,   165.f);
       // float3 b1tr = make_float3(265.f - (0.40f)*fabs(cosf(z1Theta))*b1size.x, fabs(sinf(z1Theta))*b1size.x, 255.f + (0.5f)*fabs(cosf(z1Theta))*b1size.x);
        float3 b1tr = make_float3(265.f, 0.f, 295.f);
		 OptixTraversableHandle box1 = ioGeometryGroup::createBox(7, make_float3(0.f), b1size, 
            aluminum, context,geometryList);
        materialList.push_back(aluminum);
        materialList.push_back(aluminum);
        materialList.push_back(aluminum);
        materialList.push_back(aluminum);
        materialList.push_back(aluminum);
        materialList.push_back(aluminum);
        sutil::Matrix4x4 transf = ioTransform::translate(b1tr); 
        //transf = transf * ioTransform::rotateZ(z1Theta * (180.f / CUDART_PI_F));
        transf *= ioTransform::rotateY(15.f);
        for (int i = 0; i < geometryList.size(); i++) {
            geometryList[i]->init(context);
        }
    /*    optix::GeometryGroup box1 = ioGeometryGroup::createBox(make_float3(0.f), b1size, aluminum, context);
        topGroup.addChild(ioTransform::translate(b1tr,
                                                 ioTransform::rotateY(15.f+8.f,
                                                                      ioTransform::rotateZ(z1Theta*(180.f/CUDART_PI_F),
                                                                                           box1, context),
                                                                      context),
                                                 context),
                          context);*/

        // float3 b2size = make_float3(165.f, 165.f,   165.f);
        // float3 b2tr = make_float3(130.f, 0.f, 65.f);
        // optix::GeometryGroup box2 = ioGeometryGroup::createBox(make_float3(0.f), b2size, wallWhite, context);
        // topGroup.addChild(ioTransform::translate(b2tr,
        //                                          ioTransform::rotateY(-18.f,
        //                                                               box2, context),
        //                                          context),
        //                   context);

        // // add a "fog"
        // ioMaterial *ambientFog = new ioIsotropicMaterial(new ioConstantTexture(make_float3(0.73f)));
        // topGroup.addChild(ioGeometryInstance::createVolumeBox(make_float3(0.f), make_float3(555.f), 3e-5f, ambientFog,
        //                                                       context), context);


        uint32_t seed = 0x6314759;
               
        // GeometryInstance
        geoInstList.resize(13);

        // Taking advantage of geometryList.size == materialList.size
        int i = 0;
        for ( i = 0; i < 7; i++)
        {
            //std::cerr << i << std::endl;
          //  geoInstList[i] = ioGeometryInstance();
            geoInstList[i].init(context,i ,i);
            geoInstList[i].setGeometry(*geometryList[i]);
            //materialList[i]->assignTo(geoInstList[i].get(), context);
        }

        for (; i < 13; i++) {
            geoInstList[i].init(context, i, i);
            geoInstList[i].setGeometry(*geometryList[i]);
          //  geoInstList[i].get().traversableHandle = box1;
            std::copy  (transf.getData(),transf.getData()+12, 
                & geoInstList[i].get().transform[0]         );
        }
        // World & Acceleration
        geometryGroup.init(context);  // init() sets acceleration
        for (int i = 0; i < geoInstList.size(); i++)
            geometryGroup.addChild(geoInstList[i]);
        m_root = geometryGroup.buildInstanceAccel();
      
        LightDefinition light;
        light.emission = make_float3(15.f);
        light.vecU = make_float3(343.f - 213.f, 0, 0.f);
        light.vecV = make_float3(0, 0, 332.f - 227.f);
        light.position = make_float3(213.f, 554.f, 227.f);
        light.area = length(cross(light.vecU, light.vecV));
        light.normal = normalize(cross(light.vecU, light.vecV));
        m_lightDefinitions.push_back(light);
        // Create and Init our scene camera
        camera = new ioPerspectiveCamera(
            278.f, 278.f, -800.f,
            278.f, 278.f, 0.f,
            0.0f, 1.0f, 0.0f,
            40.0f, float(m_Nx) / float(m_Ny),
            /*aperture*/1.f,
            /*focus_distance*/10.f
            ,0.f,1.f);

        //context["skyLight"]->setInt(true);
   //     context["skyLight"]->setInt(false);

     //   return topGroup.get();
    }


    void VolumesCornellBox(OptixDeviceContext& context, int Nx, int Ny) {

        sceneDescription = "Cornell box with volumes (participating media)";

        ioMaterial *wallRed = new ioLambertianMaterial(new ioConstantTexture(make_float3(0.65f, 0.05f, 0.05f)));
        ioMaterial *wallGreen = new ioLambertianMaterial(new ioConstantTexture(make_float3(0.12f, 0.45f, 0.15f)));
        ioMaterial *wallWhite = new ioLambertianMaterial(new ioConstantTexture(make_float3(0.73f, 0.73f, 0.73f)));
        ioTexture* light15 =  new ioConstantTexture(make_float3(15.f, 15.f, 15.f));

        ioMaterial *blackFog = new ioIsotropicMaterial(new ioConstantTexture(make_float3(0.f)));
        ioMaterial *whiteFog = new ioIsotropicMaterial(new ioConstantTexture(make_float3(1.f)));

        geometryList.push_back(new ioAARect(0.f, 555.f, 0.f, 555.f, 555.f,  true, X_AXIS)); // left wall
        geometryList.push_back(new ioAARect(0.f, 555.f, 0.f, 555.f, 0.f,   false, X_AXIS)); // right wall
        geometryList.push_back(new ioAARect(0.f, 555.f, 0.f, 555.f, 555.f,  true, Y_AXIS)); // roof
        geometryList.push_back(new ioAARect(0.f, 555.f, 0.f, 555.f, 0.f,   false, Y_AXIS)); // floor
        geometryList.push_back(new ioAARect(0.f, 555.f, 0.f, 555.f, 555.f,  true, Z_AXIS)); // back wall

        geometryList.push_back(new ioAARect(213.f, 343.f, 227.f, 332.f, 554.f, true, Y_AXIS)); // light

        materialList.push_back(wallGreen);
        materialList.push_back(wallRed);
        materialList.push_back(wallWhite);
        materialList.push_back(wallWhite);
        materialList.push_back(wallWhite);

        materialList.push_back(new ioDiffuseLightMaterial(light15));
  // init all geometry
        for (int i = 0; i < geometryList.size(); i++) {
            geometryList[i]->init(context);
        }
        // GeometryInstance
        geoInstList.resize(geometryList.size());

        // Taking advantage of geometryList.size == materialList.size
        for (int i = 0; i < geoInstList.size(); i++)
        {
            //std::cerr << i << std::endl;
            geoInstList[i] = ioGeometryInstance();
            geoInstList[i].init(context, i, i);
            geoInstList[i].setGeometry(*geometryList[i]);
            //materialList[i]->assignTo(geoInstList[i].get(), context);
        }

        // World & Acceleration
        geometryGroup.init(context);  // init() sets acceleration
        for (int i = 0; i < geoInstList.size(); i++)
            geometryGroup.addChild(geoInstList[i]);
        // place some objects in the box

        // // bigger Sphere
        // geometryList.push_back(new ioSphere(365.f, 165.f, 295.f, 165.f));
        // materialList.push_back(wallWhite);

        // // small sphere
        // geometryList.push_back(new ioSphere(185.f, 75.f, 155.f, 75.f));
        // materialList.push_back(wallWhite);

        // // medium glass sphere
        // geometryList.push_back(new ioSphere(185.f, 105.f, 155.f, 105.f));
        // materialList.push_back(new ioDielectricMaterial(1.5f));

        // boxes
        const float z1Theta = -12.5f * (CUDART_PI_F/180.f);
        float3 b1size = make_float3(165.f,  330.f,   165.f);
        float3 b1tr = make_float3(265.f, fabs(sinf(z1Theta))*b1size.x + 0.f, 255.f);
        // //  Rear Box, white wall - transform
        // optix::GeometryGroup box1 = ioGeometryGroup::createBox(make_float3(0.f), b1size, wallWhite, context);
        // topGroup.addChild(ioTransform::translate(b1tr,
        //                                          ioTransform::rotateZ(z1Theta*(180.f/CUDART_PI_F),
        //                                                               ioTransform::rotateY(15.f,
        //                                                                                    box1, context),
        //                                                               context),
        //                                          context),
        //                   context);

        // //  Rear Box, Volume, black fog - transform
        materialList.push_back(blackFog);
        OptixInstance box1 = ioGeometryInstance::createVolumeBox(6,6,make_float3(0.f), b1size, 0.006f, blackFog, context,geometryList);
   /*     topGroup.addChild(ioTransform::translate(b1tr,
                                                 ioTransform::rotateZ(z1Theta*(180.f/CUDART_PI_F),
                                                                      ioTransform::rotateY(15.f,
                                                                                           box1, context),
                                                                      context),
                                                 context),
                          context);*/


        sutil::Matrix4x4 trans = ioTransform::translate(b1tr); 
        trans *= ioTransform::rotateZ(z1Theta * (180.f / CUDART_PI_F));
        trans *= ioTransform::rotateY(15.f);
        std::swap_ranges(&box1.transform[0],
            box1.transform + 12, trans.getData());

        float3 b2origin = make_float3(0.f);
        float3 b2size = make_float3(165.f, 165.f,   165.f);
        float3 b2tr = make_float3(130.f, 0.f, 65.f);
        // //  Box, white wall - transform
        // optix::GeometryGroup box2 = ioGeometryGroup::createBox(make_float3(0.f), b2size, wallWhite, context);
        // topGroup.addChild(ioTransform::translate(b2tr,
        //                                          ioTransform::rotateY(-18.f,
        //                                                               box2, context),
        //                                          context),
        //                   context);

        // //  Box, Volume Box - transform
        // optix::GeometryInstance box2 = ioGeometryInstance::createVolumeBox(make_float3(0.f), b2size,
        //                                                                 0.005f, whiteFog, context);
        // topGroup.addChild(ioTransform::translate(b2tr,
        //                                          ioTransform::rotateY(-18.f,
        //                                                               box2, context),
        //                                          context),
        //                   context);

        // // white fog volume box
        // b2origin += b2tr;
        // b2size += b2tr;
        // geometryList.push_back(new ioVolumeBox(b2origin, b2size, 0.01f));
        // materialList.push_back(whiteFog);

        // white fog volume sphere - transform
        b2origin += make_float3(165.f/2.f, 75.f, 165.f/2.f);
        materialList.push_back(whiteFog);
        OptixInstance sphere2 = ioGeometryInstance::createVolumeSphere(7,7,b2origin, 75.f, 0.005f, whiteFog, context,geometryList);
      
        sutil::Matrix4x4 trans2 = ioTransform::translate(b2tr);
     //   trans2.translate(b2tr);
        std::swap_ranges(&sphere2.transform[0],
            sphere2.transform + 12, trans2.getData());
        // // white fog volume sphere
        // b2origin += b2tr;
        // b2origin += make_float3(165.f/2.f, 75.f, 165.f/2.f);
        // geometryList.push_back(new ioVolumeSphere (b2origin.x , b2origin.y, b2origin.z,
        //                                           75.f, 0.01f));
        // materialList.push_back(whiteFog);


        uint32_t seed = 0x6314759;

        
        geometryGroup.get().push_back(box1);
        geometryGroup.get().push_back(sphere2);
       // topGroup.addChild(geometryGroup.get(), context);
        m_root = geometryGroup.buildInstanceAccel();
        // Create and Init our scene camera
        camera = new ioPerspectiveCamera(
            278.f, 278.f, -800.f,
            278.f, 278.f, 0.f,
            0.0f, 1.0f, 0.0f,
            40.0f, float(Nx) / float(Ny),
            /*aperture*/0.1f,
            /*focus_distance*/10.f
            );

        //context["skyLight"]->setInt(true);
      /*  context["skyLight"]->setInt(false);

        return topGroup.get();*/
    }


    void TheNextWeekFinal(OptixDeviceContext& context, int Nx, int Ny) {
        sceneDescription = "The Next Week final scene";

        ioTexture *brown = new ioConstantTexture(make_float3(0.7f, 0.3f, 0.1f));
        ioTexture *groundGreenish = new ioConstantTexture(make_float3(0.48f, 0.83f, 0.53f));
        ioTexture *metal1 = new ioConstantTexture(make_float3(0.8f, 0.8f, 0.9f));
        ioTexture *noisep1 = new ioNoiseTexture(0.1f);
        ioTexture *noise4 = new ioNoiseTexture(4.f);

        ioTexture *earthGlobeImage = new ioImageTexture("assets/earthmap.jpg");
        ioTexture *light7 =  new ioConstantTexture(make_float3(7.f, 7.f, 7.f));

        uint32_t seed = 0x6314759;

        //
        ioMaterial *glassyBlueFog = new ioIsotropicMaterial(new ioConstantTexture(make_float3(0.2f, 0.4f, 0.9f)));
        ioMaterial *ambientFog = new ioIsotropicMaterial(new ioConstantTexture(make_float3(0.95f)));
        ioMaterial *ground = new ioLambertianMaterial(groundGreenish);
// light
        geometryList.push_back(new ioAARect(123.f, 423.f, 147.f, 412.f, 554.f, true, Y_AXIS)); // light
        materialList.push_back(new ioDiffuseLightMaterial(light7));
        LightDefinition light;
        light.emission = make_float3(7.f, 7.f, 7.f);
        light.vecU = make_float3(423.f - 123.f, 0, 0.f);
        light.vecV = make_float3(0, 0.f, 412.f-147.f);
        light.position = make_float3(123.f, 554.f, 147.f);
        light.area = length(cross(light.vecU, light.vecV));
        light.normal = normalize(cross(light.vecU, light.vecV));
        m_lightDefinitions.push_back(light);
        // brown moving sphere
        float3 center = make_float3(400.f, 400.f, 200.f);
        float3 center1tr = center + make_float3(30.f, 0.f, 0.f);
        // glass sphere
        geometryList.push_back(new ioSphere(260.f, 150.f, 45.f, 50.f) );
        materialList.push_back(new ioDielectricMaterial(1.5f));

        // metal sphere
        geometryList.push_back(new ioSphere(0.f, 150.f, 145.f, 50.f) );
        //materialList.push_back(new ioMetalMaterial(metal1, 10.f));
        materialList.push_back(new ioMetalMaterial(metal1, 0.2f));

        // blue glassy sphere with a volume
        float3 centerGlassy = make_float3(360.f, 150.f, 45.f);
        geometryList.push_back(new ioSphere(centerGlassy.x, centerGlassy.y, centerGlassy.z, 70.f));
        materialList.push_back(new ioDielectricMaterial(1.5f));
        
        // room ambient / boundary
        geometryList.push_back(new ioSphere(0.f, 0.f, 0.f, 5000.f) );
        materialList.push_back(new ioDielectricMaterial(1.5f));
        
        // earth globe
        geometryList.push_back(new ioSphere(400.f, 200.f, 400.f, 100.0f));
        materialList.push_back(new ioLambertianMaterial(earthGlobeImage));

        // marble (perlin noise) sphere
        geometryList.push_back(new ioSphere(220.f, 280.f, 300.f, 80.f));
        materialList.push_back(new ioLambertianMaterial(noisep1));
        geometryList.push_back(new ioMovingSphere(center.x, center.y, center.z, center1tr.x, center1tr.y, center1tr.z,
            50.f, 0.f, 1.f));
        materialList.push_back(new ioLambertianMaterial(brown));

        // init all geometry
        for (int i = 0; i < geometryList.size(); i++) {
            geometryList[i]->init(context);
        }
        // GeometryInstance
        geoInstList.resize(geometryList.size());
        int sphereidx = 0, movspidx = 1, rctidx = 1, volrc = 0, volsp = 0;
        // Taking advantage of geometryList.size == materialList.size
        for (int i = 0; i < geoInstList.size(); i++)
        {
            //std::cerr << i << std::endl;
            geoInstList[i] = ioGeometryInstance();
            if (dynamic_cast<ioMovingSphere*>(geometryList[i]) )
            {
                geoInstList[i].init(context, i, 1006);
                
            }
            else if (dynamic_cast<ioAARect*>(geometryList[i])) {
                geoInstList[i].init(context, i, 1007);
            }
            else {
                geoInstList[i].init(context, i, i-1);
                sphereidx++;
            }
            
            geoInstList[i].setGeometry(*geometryList[i]);
            //materialList[i]->assignTo(geoInstList[i].get(), context);
        }

        // World & Acceleration
        //geometryGroup.init(context);  // init() sets acceleration
        //for (int i = 0; i < geoInstList.size(); i++)
        //    geometryGroup.addChild(geoInstList[i]);

        // ground
        for(int i = 0; i < 20; i++){
            for(int j = 0; j < 20; j++){
                float w = 100.f;
                float x0 = -1000 + i * w;
                float z0 = -1000 + j * w;
                float y0 = 0.f;
                float x1 = x0 + w;
                float y1 = 100 * (randf(seed) + 0.01f);
                float z1 = z0 + w;
                OptixTraversableHandle box = ioGeometryGroup::createBox(0, make_float3(x0, y0, z0), make_float3(x1, y1, z1),
                    ground, context, geometryList);
                materialList.push_back(ground);
                materialList.push_back(ground);
                materialList.push_back(ground);
                materialList.push_back(ground);
                materialList.push_back(ground);
                materialList.push_back(ground);

                /*for (int ii = geometryList.size()-6; ii < geometryList.size(); ii++) {
                    geometryList[ii]->init(context);
                }*/

                for (int jj = 0,  ii = geometryList.size() - 6; jj < 6;++ii, ++jj) {
                    geometryList[ii]->init(context);
                    ioGeometryInstance Instance;
                    Instance.init(context, i * 120 + j*6 + 8+jj, j * 6 + 1008 + i * 120 +jj);
                    Instance.setGeometry(*geometryList[ii]);
                    /*float trans[12] = { 1.f,0.f,0.f,0.f,
                                       0.f,1.f,0.f,0.f,
                                       0.f,0.f,1.f,0.f };
                    std::swap_ranges(&Instance.get().transform[0],
                        Instance.get().transform + 12, trans);*/
                    geoInstList.push_back(Instance);
                }
                
            }
        }
        materialList.push_back(glassyBlueFog);
        OptixInstance volSphere0 = ioGeometryInstance::createVolumeSphere(geoInstList.size(),
            geometryList.size() + 1000, centerGlassy, 70.f, 0.2f, glassyBlueFog,
            context, geometryList);

        materialList.push_back(ambientFog);
        OptixInstance volSphere1 = ioGeometryInstance::createVolumeSphere(geoInstList.size()+1, 
            geometryList.size() + 1000, make_float3(0.f), 500.f, 8e-5f, ambientFog,
            context, geometryList);

        ioMaterial *white = new ioLambertianMaterial(new ioConstantTexture(make_float3(0.93f)));
        std::vector<OptixInstance> d_list;
        unsigned int insIdx = geoInstList.size() + 2;
        for(int j = 0; j < 1000; j++) {
            d_list.push_back(ioGeometryInstance::createSphere(insIdx++,
               6+j,  make_float3(165 * randf(seed), 165 * randf(seed), 165 * randf(seed)),
                                                              10.f, white, context, geometryList));
            materialList.push_back(white);
            sutil::Matrix4x4 transmat = ioTransform::translate(make_float3(-100.f, 270.f, 395.f)); 
            transmat *= ioTransform::rotateY(20.f);
            std::swap_ranges(&d_list[j].transform[0],
                d_list[j].transform + 12, transmat.getData());
        }
              

      
        // World & Acceleration
        geometryGroup.init(context);  // init() sets acceleration
        for (int i = 0; i < geoInstList.size(); i++)
            geometryGroup.addChild(geoInstList[i]);

        geometryGroup.get().push_back(volSphere0);
        geometryGroup.get().push_back(volSphere1);

        geometryGroup.get().insert(geometryGroup.get().end(), d_list.begin(), d_list.end());
        m_root = geometryGroup.buildInstanceAccel();
        

        // Create and Init our scene camera
        const float3 lookfrom = make_float3(478.f, 278.f, -600.f);
        const float3 lookat = make_float3(278.f, 278.f, 0.f);
        const float3 up = make_float3(0.f, 1.f, 0.f);
        const float fovy = 40.0f;
        const float aspect = (float(Nx) / float(Ny));
        const float aperture = (0.1f);
        const float focus_distance = (10.f);

        camera = new ioPerspectiveCamera(
            lookfrom.x, lookfrom.y, lookfrom.z,
            lookat.x, lookat.y, lookat.z,
            up.x, up.y, up.z,
            fovy, aspect,
            aperture,
            focus_distance,
            /* t0 and t1 */0.f, 1.f
            );

     
    }

    void destroy() {
        for(int i = 0; i < materialList.size(); i++)
            materialList[i]->destroy();

        for(int i = 0; i < geometryList.size(); i++)
            geometryList[i]->destroy();

        for(int i = 0; i < geoInstList.size(); i++)
            geoInstList[i].destroy();

        geometryGroup.destroy();

        camera->destroy();
        delete camera;
    }

    std::string getDescription() {
        return sceneDescription;
    }

public:
    std::string sceneDescription;

    int m_Nx;
    int m_Ny;
    int m_maxRayDepth;
    int m_numSamples;

    std::vector<ioMaterial*> materialList;
    std::vector<ioGeometry*> geometryList;
    std::vector<ioGeometryInstance> geoInstList;
    ioGeometryGroup geometryGroup;
    //ioGroup topGroup;
    ioCamera* camera;
    OptixTraversableHandle m_root;
    //optix::Group world;
    pdfCallfun_host MCpdf;
    std::vector<LightDefinition> m_lightDefinitions;
};

#endif //!IO_SCENE_H
