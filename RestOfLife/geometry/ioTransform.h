#ifndef IO_TRANSFORM_H
#define IO_TRANSFORM_H

#include <optix.h>
#include "sutil/vec_math.h"
#include <vector_types.h>
#include <math_constants.h>
#include "sutil/Matrix.h"

// translate utility functions
class ioTransform
{
public:

    static sutil::Matrix4x4 translate(float3& offset){
        sutil::Matrix4x4 matrix = translateMatrix(offset);
         
        return matrix;
    }
    
    static sutil::Matrix4x4 rotateAboutPoint(float angleDegrees, float3& point){
        sutil::Matrix4x4 matrix = rotateAboutPointMatrix(angleDegrees * CUDART_PI_F / 180.f, point);

        
        return matrix;
    }

    static sutil::Matrix4x4 rotateX(float angleDegrees){
        sutil::Matrix4x4 matrix = rotateMatrixX(angleDegrees * CUDART_PI_F / 180.f);

        return matrix;
    }
      

    static sutil::Matrix4x4 rotateY(float angleDegrees){
        sutil::Matrix4x4 matrix = rotateMatrixY(angleDegrees * CUDART_PI_F / 180.f);

        return matrix;
    }

   

    


    static sutil::Matrix4x4 rotateZ(float angleDegrees){
        sutil::Matrix4x4 matrix = rotateMatrixZ(angleDegrees * CUDART_PI_F / 180.f);

       
        return matrix ;
    }
       

    

    

    static sutil::Matrix4x4 scale(float3& scale){
        sutil::Matrix4x4 matrix (  sutil::Matrix4x4::scale(scale).getData());

      
        return matrix;
    }


private:

    static sutil::Matrix4x4 translateMatrix(float3 offset){
        float floatM[16] = {
            1.0f, 0.0f, 0.0f, offset.x,
            0.0f, 1.0f, 0.0f, offset.y,
            0.0f, 0.0f, 1.0f, offset.z,
            0.f,  0.f, 0.f,   1.f
        };
        sutil::Matrix4x4 mm(floatM);

        return mm;
    }

    // rotateAboutPoint
    static sutil::Matrix4x4 rotateAboutPointMatrix(float angle, float3 offset){
        float floatM[16] = {
            cosf(angle), 0.0f, -sinf(angle), offset.x - cosf(angle) * offset.x + sinf(angle) * offset.z,
            0.0f,        1.0f,         0.0f,                                                        0.f,
            sinf(angle), 0.0f,  cosf(angle), offset.z - sinf(angle) * offset.x - cosf(angle) * offset.z,
            0.f,  0.f, 0.f,   1.f
        };
        sutil::Matrix4x4 mm(floatM);

        return mm;
    }

    // rotateX functions
    static sutil::Matrix4x4 rotateMatrixX(float angle){
        float floatM[16] = {
            1.0f,         0.0f,         0.0f, 0.0f,
            0.0f,  cosf(angle), -sinf(angle), 0.0f,
            0.0f,  sinf(angle),  cosf(angle), 0.0f,
            0.f,  0.f, 0.f,   1.f
        };
        sutil::Matrix4x4 mm(floatM);

        return mm;
    }

    // rotate  Y functions
    static sutil::Matrix4x4 rotateMatrixY(float angle){
        float floatM[16] = {
            cosf(angle),  0.0f,  sinf(angle), 0.0f,
            0.0f,         1.0f,         0.0f, 0.0f,
            -sinf(angle), 0.0f,  cosf(angle), 0.0f,
            0.0f,         0.0f,         0.0f, 1.0f
        };
        sutil::Matrix4x4 mm(floatM);

        return mm;
    }


    // rotateZ functions
    static sutil::Matrix4x4 rotateMatrixZ(float angle){
        float floatM[16] = {
            cosf(angle), -sinf(angle), 0.0f, 0.0f,
            sinf(angle),  cosf(angle), 0.0f, 0.0f,
            0.0f,         0.0f, 1.0f, 0.0f,
            0.f,  0.f, 0.f,   1.f
        };
        sutil::Matrix4x4 mm(floatM);

        return mm;
    }


};

#endif
