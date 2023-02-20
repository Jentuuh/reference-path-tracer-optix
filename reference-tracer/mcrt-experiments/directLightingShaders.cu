#pragma once

#include <optix_device.h>
#include "random.hpp"
#include "vec_math.hpp"

#include "LaunchParams.hpp"
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#define NUM_SAMPLES_PER_STRATIFY_CELL 20000
#define NUM_DIRECTIONS 500
#define PI 3.14159265358979323846f

using namespace mcrt;

namespace mcrt {

    extern "C" __constant__ LaunchParamsDirectLighting optixLaunchParams;

    static __forceinline__ __device__
        void* unpackPointer(uint32_t i0, uint32_t i1)
    {
        const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
        void* ptr = reinterpret_cast<void*>(uptr);
        return ptr;
    }

    static __forceinline__ __device__
        void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    static __forceinline__ __device__ T* getPRD()
    {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T*>(unpackPointer(u0, u1));
    }

    extern "C" __global__ void __closesthit__radiance__direct__lighting()
    {
        const MeshSBTDataDirectLighting& sbtData
            = *(const MeshSBTDataDirectLighting*)optixGetSbtDataPointer();

        // ------------------------------------------------------------------
        // gather some basic hit information
        // ------------------------------------------------------------------
        const int primID = optixGetPrimitiveIndex();
        const glm::ivec3 index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        // Barycentric tex coords
        const glm::vec2 tc =
             (1.f - u - v) * sbtData.texcoord[index.x]
            + u * sbtData.texcoord[index.y]
            + v * sbtData.texcoord[index.z];


        //printf("TEXCOORDS %f %f", tc.x, tc.y);
        

        // Pass the texture coordinates of the ray-geometry intersection
        // to the PRD, so we can read them out in our launch shader.
        glm::vec2& prd = *(glm::vec2*)getPRD<glm::vec2>();
        prd = tc;
    }

    extern "C" __global__ void __anyhit__radiance__direct__lighting(){
        // Do nothing
    }

    extern "C" __global__ void __miss__radiance__direct__lighting()
    {
        // If the light ray did not hit any geometry, we'll write (-1;-1) to the 
        // texture coordinate so we can verify this later
        glm::vec2& prd = *(glm::vec2*)getPRD<glm::vec2>();
        prd = glm::vec2{ -1.0f, -1.0f};
    }

    extern "C" __global__ void __raygen__renderFrame__direct__lighting()
    {
        const auto& lights = optixLaunchParams.lights;

        // Get thread indices
        const int lightIndex = optixGetLaunchIndex().x;
        const int stratifyIndexX = optixGetLaunchIndex().y;
        const int stratifyIndexY = optixGetLaunchIndex().z;

        unsigned int seed = tea<4>(stratifyIndexY * 5 + stratifyIndexX, lightIndex);

        // Look up the light properties for the light in question
        LightData lightProperties = optixLaunchParams.lights[lightIndex];
        float stratifyCellWidth = lightProperties.width / optixLaunchParams.stratifyResX;
        float stratifyCellHeigth = lightProperties.height / optixLaunchParams.stratifyResY;
        
        // We start from the light origin, and calculate the origin of the current stratification cell based on the stratifyIndex of this thread
        glm::vec3 cellOrigin = lightProperties.origin + (stratifyIndexX * stratifyCellWidth * lightProperties.du) + (stratifyIndexY * stratifyCellHeigth * lightProperties.dv);
        glm::vec3 cellMax = cellOrigin + (stratifyCellWidth * lightProperties.du) + (stratifyCellHeigth * lightProperties.dv);

        // Send out a ray for each sample
        for (int i = 0; i < NUM_SAMPLES_PER_STRATIFY_CELL; i++)
        {
            // The per-ray data in which we'll store the texture coordinate of the vertex that the ray hits
            glm::vec2 rayTexCoordPRD = glm::vec2(0.0f);
            uint32_t u0, u1;
            packPointer(&rayTexCoordPRD, u0, u1);

            // Randomize ray origins within a cell
            float2 cellOffset = float2{rnd(seed), rnd(seed)};

            glm::vec3 rayOrigin = cellOrigin + (cellOffset.x * stratifyCellWidth * lightProperties.du) + (cellOffset.y * stratifyCellHeigth * lightProperties.dv);


            for (int j = 0; j < NUM_DIRECTIONS; j++)
            {
                // Generate random direction on hemisphere along normal (now rather inefficient, TODO: optimize)
                // e.g.: align normal with (0,1,0), calculate from there on.
                // https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

                // [theta, phi]
                float3 normal3f = float3{ lightProperties.normal.x, lightProperties.normal.y, lightProperties.normal.z };
                float3 randomDir = float3{ 0,0,0 };

                // Generate new directions until we have found one in the same hemisphere as the normal centered hemisphere
                while (true)
                {

                    float2 randomThetaPhi = float2{ rnd(seed) * PI, rnd(seed) * 2.0f * PI };
                    float x = sin(randomThetaPhi.x) * cos(randomThetaPhi.y);
                    float y = sin(randomThetaPhi.x) * sin(randomThetaPhi.y);
                    float z = cos(randomThetaPhi.x);
                    randomDir = float3{ x,y,z };

                    float dotproduct = dot(randomDir, normal3f);

                    if (dotproduct > 0) {
                        break;
                    }
                }

                float3 rayOrigin3f = float3{ rayOrigin.x, rayOrigin.y, rayOrigin.z };

                optixTrace(optixLaunchParams.traversable,
                    rayOrigin3f,
                    randomDir,
                    0.f,    // tmin
                    1e20f,  // tmax
                    0.0f,   // rayTime
                    OptixVisibilityMask(255),
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
                    0,  // SBT offset
                    1,  // SBT stride
                    0,  // missSBTIndex 
                    u0, u1);

                if (rayTexCoordPRD.x != -1.0f && rayTexCoordPRD.y != -1.0f)
                {
                    // TODO: WEIGH THE LIGHT CONTRIBUTION BY COSINE OF THE INCIDENT RAY ANGLE!
                    // Calculate light contribution
                    const int r = int(255.99f * lightProperties.power.x);
                    const int g = int(255.99f * lightProperties.power.y);
                    const int b = int(255.99f * lightProperties.power.z); 


                    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
                    // to make stb_image_write happy ...
                    const uint32_t rgba = 0xff000000
                        | (r << 0) | (g << 8) | (b << 16);

                    // Convert UV coordinates to buffer index and write to color buffer
                    const int xCoord = int(rayTexCoordPRD.x * optixLaunchParams.directLightingTexture.size);
                    const int yCoord = int(rayTexCoordPRD.y * optixLaunchParams.directLightingTexture.size);

                    const uint32_t uvIndex = (yCoord * optixLaunchParams.directLightingTexture.size) + xCoord;
                    // The write operation needs to be atomic because we're using a scattering approach, that is, multiple lights (threads)
                    // might be accessing the UV map at the same time
                    atomicAdd(&optixLaunchParams.directLightingTexture.colorBuffer[uvIndex], rgba);
                }
            }
        }
    }
}