#pragma once

#include <optix_device.h>
#include "random.hpp"
#include "vec_math.hpp"

#include "LaunchParams.hpp"
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#define NUM_SAMPLES_PER_STRATIFY_CELL 150
#define PI 3.14159265358979323846f
#define EPSILON 0.0000000000002f

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

    static __forceinline__ __device__ DirectLightingPRD loadDirectLightingPRD()
    {
        DirectLightingPRD prd = {};

        prd.rayOrigin.x = __uint_as_float(optixGetPayload_0());
        prd.rayOrigin.y = __uint_as_float(optixGetPayload_1());
        prd.rayOrigin.z = __uint_as_float(optixGetPayload_2());

        prd.lightSamplePos.x = __uint_as_float(optixGetPayload_3());
        prd.lightSamplePos.y = __uint_as_float(optixGetPayload_4());
        prd.lightSamplePos.z = __uint_as_float(optixGetPayload_5());

        prd.resultColor.x = __uint_as_float(optixGetPayload_6());
        prd.resultColor.y = __uint_as_float(optixGetPayload_7());
        prd.resultColor.z = __uint_as_float(optixGetPayload_8());

        return prd;
    }

    static __forceinline__ __device__ void storeDirectLightingPRD(DirectLightingPRD prd)
    {
        optixSetPayload_0(__float_as_uint(prd.rayOrigin.x));
        optixSetPayload_1(__float_as_uint(prd.rayOrigin.y));
        optixSetPayload_2(__float_as_uint(prd.rayOrigin.z));

        optixSetPayload_3(__float_as_uint(prd.lightSamplePos.x));
        optixSetPayload_4(__float_as_uint(prd.lightSamplePos.y));
        optixSetPayload_5(__float_as_uint(prd.lightSamplePos.z));

        optixSetPayload_6(__float_as_uint(prd.resultColor.x));
        optixSetPayload_7(__float_as_uint(prd.resultColor.y));
        optixSetPayload_8(__float_as_uint(prd.resultColor.z));
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

        // Barycentric world coords
        const glm::vec3 worldPos =
            (1.f - u - v) * sbtData.vertex[index.x]
            + u * sbtData.vertex[index.y]
            + v * sbtData.vertex[index.z];

        DirectLightingPRD prd = loadDirectLightingPRD();
        float squaredDistOriginLight = (((prd.lightSamplePos.x - prd.rayOrigin.x) * (prd.lightSamplePos.x - prd.rayOrigin.x)) + ((prd.lightSamplePos.y - prd.rayOrigin.y) * (prd.lightSamplePos.y - prd.rayOrigin.y)) + ((prd.lightSamplePos.z - prd.rayOrigin.z) * (prd.lightSamplePos.z - prd.rayOrigin.z)));
        float squaredDistOriginIntersection = (((worldPos.x - prd.rayOrigin.x) * (worldPos.x - prd.rayOrigin.x)) + ((worldPos.y - prd.rayOrigin.y) * (worldPos.y - prd.rayOrigin.y)) + ((worldPos.z - prd.rayOrigin.z) * (worldPos.z - prd.rayOrigin.z)));

        if (squaredDistOriginLight < squaredDistOriginIntersection || squaredDistOriginIntersection < EPSILON)
        {
            //printf("Contribute!");
            prd.resultColor = { 1.0f, 1.0f, 1.0f };
        }
        else {
            prd.resultColor = { 0.0f, 0.0f, 0.0f };
        }

        storeDirectLightingPRD(prd);
    }

    extern "C" __global__ void __anyhit__radiance__direct__lighting() {
        // Do nothing
    }

    extern "C" __global__ void __miss__radiance__direct__lighting()
    {
        // If the ray doesn't hit anything we can conclude it contributes
        DirectLightingPRD prd = loadDirectLightingPRD();
        prd.resultColor = { 1.0f, 1.0f, 1.0f };
        storeDirectLightingPRD(prd);
    }

    extern "C" __global__ void __raygen__renderFrame__direct__lighting()
    {
        const auto& lights = optixLaunchParams.lights;

        // Get thread indices
        const int uIndex = optixGetLaunchIndex().x;
        const int vIndex = optixGetLaunchIndex().y;

        glm::vec3 UVWorldPos = optixLaunchParams.uvWorldPositions.UVDataBuffer[vIndex * optixLaunchParams.directLightingTexture.size + uIndex].worldPosition;
        const glm::vec3 UVNormal = optixLaunchParams.uvWorldPositions.UVDataBuffer[vIndex * optixLaunchParams.directLightingTexture.size + uIndex].worldNormal;
        const glm::vec3 diffuseColor = optixLaunchParams.uvWorldPositions.UVDataBuffer[vIndex * optixLaunchParams.directLightingTexture.size + uIndex].diffuseColor;

        // We apply a small offset of 0.00001f in the direction of the normal to the UV world pos, to 'mitigate' floating point rounding errors causing false occlusions/illuminations
        UVWorldPos = glm::vec3{ UVWorldPos.x + UVNormal.x * 0.00001f, UVWorldPos.y + UVNormal.y * 0.00001f, UVWorldPos.z + UVNormal.z * 0.00001f };

        // Iterate over all lights
        for (int i = 0; i < optixLaunchParams.amountLights; i++)
        {
            unsigned int seed = tea<4>(vIndex * optixLaunchParams.directLightingTexture.size + uIndex, i);

            // Look up the light properties for the light in question
            LightData lightProperties = optixLaunchParams.lights[i];
            float stratifyCellWidth = lightProperties.width / optixLaunchParams.stratifyResX;
            float stratifyCellHeight = lightProperties.height / optixLaunchParams.stratifyResY;


            glm::vec3 totalLightContribution = { 0.0f, 0.0f, 0.0f };
            for (int stratifyIndexX = 0; stratifyIndexX < optixLaunchParams.stratifyResX; stratifyIndexX++)
            {
                for (int stratifyIndexY = 0; stratifyIndexY < optixLaunchParams.stratifyResY; stratifyIndexY++)
                {
                    // We start from the light origin, and calculate the origin of the current stratification cell based on the stratifyIndex of this thread
                    glm::vec3 cellOrigin = lightProperties.origin + (stratifyIndexX * stratifyCellWidth * lightProperties.du) + (stratifyIndexY * stratifyCellHeight * lightProperties.dv);

                    // Send out a ray for each sample
                    for (int l = 0; l < NUM_SAMPLES_PER_STRATIFY_CELL; l++)
                    {
                        // Randomize ray origins within a cell
                        float2 cellOffset = float2{ rnd(seed), rnd(seed) };
                        glm::vec3 rayDestination = cellOrigin + (cellOffset.x * stratifyCellWidth * lightProperties.du) + (cellOffset.y * stratifyCellHeight * lightProperties.dv);

                        // Our ray direction is defined by the line between the texel and the light sample
                        glm::vec3 rayDir = rayDestination - UVWorldPos;

                        float3 rayOrigin3f = float3{ UVWorldPos.x, UVWorldPos.y, UVWorldPos.z };
                        float3 rayDir3f = float3{ rayDir.x, rayDir.y, rayDir.z };

                        DirectLightingPRD prd{};
                        prd.rayOrigin = UVWorldPos;
                        prd.lightSamplePos = rayDestination;

                        unsigned int u0, u1, u2, u3, u4, u5, u6, u7, u8;

                        u0 = __float_as_uint(prd.rayOrigin.x);
                        u1 = __float_as_uint(prd.rayOrigin.y);
                        u2 = __float_as_uint(prd.rayOrigin.z);
                        u3 = __float_as_uint(prd.lightSamplePos.x);
                        u4 = __float_as_uint(prd.lightSamplePos.y);
                        u5 = __float_as_uint(prd.lightSamplePos.z);

                        optixTrace(optixLaunchParams.traversable,
                            rayOrigin3f,
                            rayDir3f,
                            0.f,    // tmin
                            1e20f,  // tmax
                            0.0f,   // rayTime
                            OptixVisibilityMask(255),
                            OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
                            0,  // SBT offset
                            1,  // SBT stride
                            0,  // missSBTIndex 
                            u0, u1, u2, u3, u4, u5, u6, u7, u8);
                        prd.resultColor = { __uint_as_float(u6), __uint_as_float(u7), __uint_as_float(u8) };


                        // Cosine weighted contribution (perpendicular incident directions have a higher contribution)
                        float3 uvNormal3f = float3{ UVNormal.x, UVNormal.y, UVNormal.z };
                        float cosContribution = dot(normalize(rayDir3f), uvNormal3f);

                        if (cosContribution > 0.0f)
                        {
                            // TODO: (Note that BRDF is currently omitted here)
                            //float intensity = 255.99f * cosContribution * prd.resultColor.x * lightProperties.power.x;
                            float intensity = cosContribution * prd.resultColor.x * lightProperties.power.x;
                            totalLightContribution += intensity * diffuseColor;
                        }
                    }
                }
            }

            // Average out the samples contributions
            totalLightContribution /= NUM_SAMPLES_PER_STRATIFY_CELL * optixLaunchParams.stratifyResX * optixLaunchParams.stratifyResY;
            // totalLightContribution /= PI;

            //const int r = int(totalLightContribution.x);
            //const int g = int(totalLightContribution.y);
            //const int b = int(totalLightContribution.z);

            // convert to 32-bit rgba value (we explicitly set alpha to 0xff
            // to make stb_image_write happy ...
            //const uint32_t rgba = 0xff000000
            //    | (r << 0) | (g << 8) | (b << 16);

            int uvIndex = (vIndex * optixLaunchParams.directLightingTexture.size * 3) + (uIndex * 3);
            optixLaunchParams.directLightingTexture.colorBuffer[uvIndex + 0] = totalLightContribution.x;
            optixLaunchParams.directLightingTexture.colorBuffer[uvIndex + 1] = totalLightContribution.y;
            optixLaunchParams.directLightingTexture.colorBuffer[uvIndex + 2] = totalLightContribution.z;
            //optixLaunchParams.directLightingTexture.colorBuffer[uvIndex] += rgba;
        }
    }
}