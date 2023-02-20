#pragma once

#include <optix_device.h>
#include "random.hpp"
#include "vec_math.hpp"

#include "LaunchParams.hpp"
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#define INDIRECT_BRDF_SAMPLES 2
#define PI 3.14159265358979323846f

using namespace mcrt;

namespace mcrt {
    // Launch parameters in constant memory, filled in by optix upon
    // optixLaunch (this gets filled in from the buffer we pass to
    // optixLaunch)
    extern "C" __constant__ LaunchParamsTutorial optixLaunchParams;


    static __forceinline__ __device__ referenceTracerPRD loadRayPRD()
    {
        referenceTracerPRD prd = {};

        prd.resultColor.x = __uint_as_float(optixGetPayload_0());
        prd.resultColor.y = __uint_as_float(optixGetPayload_1());
        prd.resultColor.z = __uint_as_float(optixGetPayload_2());

        return prd;
    }

    static __forceinline__ __device__ void storeRayPRD(referenceTracerPRD prd)
    {
        optixSetPayload_0(__float_as_uint(prd.resultColor.x));
        optixSetPayload_1(__float_as_uint(prd.resultColor.y));
        optixSetPayload_2(__float_as_uint(prd.resultColor.z));
    }

    //------------------------------------------------------------------------------
    // closest hit and anyhit programs for radiance-type rays.
    //
    // Note eventually we will have to create one pair of those for each
    // ray type and each geometry type we want to render; but this
    // simple example doesn't use any actual geometries yet, so we only
    // create a single, dummy, set of them (we do have to have at least
    // one group of them to set up the SBT)
    //------------------------------------------------------------------------------

    extern "C" __global__ void __closesthit__shadow()
    {
        /* not going to be used ... */
    }

    extern "C" __global__ void __closesthit__radiance()
    { 
        const MeshSBTData& sbtData
            = *(const MeshSBTData*)optixGetSbtDataPointer(); 

        // ------------------------------------------------------------------
        // gather some basic hit information
        // ------------------------------------------------------------------
        const int   primID = optixGetPrimitiveIndex();
        const glm::ivec3 index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        // ------------------------------------------------------------------
        // compute normal, using either shading normal (if avail), or
        // geometry normal (fallback)
        // ------------------------------------------------------------------
        glm::vec3 Ng;   
        const glm::vec3& A = sbtData.vertex[index.x];
        const glm::vec3& B = sbtData.vertex[index.y];
        const glm::vec3& C = sbtData.vertex[index.z];
        Ng = cross(B - A, C - A);

        glm::vec3 Ns;
        if (sbtData.normal) {
            Ns = (1.f - u - v) * sbtData.normal[index.x]
                + u * sbtData.normal[index.y]
                + v * sbtData.normal[index.z];
        }
        else {
            Ns = Ng;
        }

        // ------------------------------------------------------------------
        // face-forward and normalize normals
        // ------------------------------------------------------------------
        float3 rayDirf3 = optixGetWorldRayDirection();
        const glm::vec3 rayDir = { rayDirf3.x, rayDirf3.y, rayDirf3.z };

        if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
        Ng = normalize(Ng);

        if (dot(Ng, Ns) < 0.f)
            Ns -= 2.f * dot(Ng, Ns) * Ng;
        Ns = normalize(Ns);

        // ------------------------------------------------------------------
        // compute diffuse material color, including diffuse texture, if
        // available
        // ------------------------------------------------------------------
        glm::vec3 diffuseColor = sbtData.color;
        if (sbtData.hasTexture && sbtData.texcoord) {
            // Barycentric tex coords
            const glm::vec2 tc
                = (1.f - u - v) * sbtData.texcoord[index.x]
                + u * sbtData.texcoord[index.y]
                + v * sbtData.texcoord[index.z];

            float4 fromTexf4 = tex2D<float4>(sbtData.texture, tc.x, tc.y);
            glm::vec4 fromTexture = glm::vec4{ fromTexf4.x,fromTexf4.y,fromTexf4.z,fromTexf4.w };

            diffuseColor = (glm::vec3)fromTexture;
        }

        // ------------------------------------------------------------------
        // compute direct lighting
        // ------------------------------------------------------------------
        const glm::vec3 surfPos
            = (1.f - u - v) * sbtData.vertex[index.x]
            + u * sbtData.vertex[index.y]
            + v * sbtData.vertex[index.z];

        const auto& lights = optixLaunchParams.lights;
        unsigned int seed = tea<4>(u, v);

        glm::vec3 totalLightContribution = { 0.0f, 0.0f, 0.0f };
        for (int l = 0; l < optixLaunchParams.amountLights; l++)
        {
            // Look up the light properties for the current light
            LightData lightProperties = optixLaunchParams.lights[l];
            float stratifyCellWidth = lightProperties.width / optixLaunchParams.stratifyResX;
            float stratifyCellHeight = lightProperties.height / optixLaunchParams.stratifyResY;

            for (int stratifyIndexX = 0; stratifyIndexX < optixLaunchParams.stratifyResX; stratifyIndexX++)
            {
                for (int stratifyIndexY = 0; stratifyIndexY < optixLaunchParams.stratifyResY; stratifyIndexY++)
                {
                    // We start from the light origin, and calculate the origin of the current stratification cell based on the stratifyIndex of this thread
                    glm::vec3 cellOrigin = lightProperties.origin + (stratifyIndexX * stratifyCellWidth * lightProperties.du) + (stratifyIndexY * stratifyCellHeight * lightProperties.dv);
                    
                    // Randomize ray origins within a cell
                    float2 cellOffset = float2{ rnd(seed), rnd(seed) };
                    glm::vec3 lightSamplePos = cellOrigin + (cellOffset.x * stratifyCellWidth * lightProperties.du) + (cellOffset.y * stratifyCellHeight * lightProperties.dv);

                    glm::vec3 rayDir = lightSamplePos - surfPos;

                    glm::vec3 rayOrigin = surfPos + 1e-3f * Ng; // 'Mitigate' rounding errors
                    float3 rayOrigin3f = { rayOrigin.x, rayOrigin.y, rayOrigin.z };
                    float3 lightDir3f = { rayDir.x, rayDir.y, rayDir.z };

                    referenceTracerPRD lightVisibility;

                    //glm::vec3 lightVisibility = glm::vec3{ 0.0f, 0.0f, 0.0f };
                    // the values we store the PRD pointer in:
                    uint32_t u0 = 0, u1 = 0, u2 = 0;
                    //packPointer(&lightVisibility, u0, u1);

                    // Shadow ray
                    optixTrace(optixLaunchParams.traversable,
                        rayOrigin3f,
                        lightDir3f,
                        1e-3f,      // tmin
                        1.f - 1e-3f,  // tmax
                        0.0f,       // rayTime
                        OptixVisibilityMask(255),
                        // For shadow rays: skip any/closest hit shaders and terminate on first
                        // intersection with anything. The miss shader is used to mark if the
                        // light was visible.
                        OPTIX_RAY_FLAG_DISABLE_ANYHIT
                        | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                        | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                        SHADOW_RAY_TYPE,            // SBT offset
                        RAY_TYPE_COUNT,             // SBT stride
                        SHADOW_RAY_TYPE,            // missSBTIndex 
                        u0, u1, u2);

                    lightVisibility.resultColor = glm::vec3{ __uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2) };
                    float cosWeight = fabsf(dot(rayDir, Ns));
                    totalLightContribution += cosWeight * lightVisibility.resultColor.x * lightProperties.power;
                }
            }
        }
        // Average out the samples contributions
        totalLightContribution /= optixLaunchParams.amountLights * optixLaunchParams.stratifyResX * optixLaunchParams.stratifyResY;


        // ------------------------------------------------------------------
        // final shading: a bit of ambient, a bit of directional ambient,
        // and directional component based on shadowing
        // ------------------------------------------------------------------
        referenceTracerPRD prd = loadRayPRD();
        prd.resultColor = diffuseColor * totalLightContribution;
        storeRayPRD(prd);
    }

    extern "C" __global__ void __anyhit__radiance()
    { /*! for this simple example, this will remain empty */
    }
    extern "C" __global__ void __anyhit__shadow()
    { /*! not going to be used */
    }


    //------------------------------------------------------------------------------
    // miss program that gets called for any ray that did not have a
    // valid intersection
    //
    // as with the anyhit/closest hit programs, in this example we only
    // need to have _some_ dummy function to set up a valid SBT
    // ------------------------------------------------------------------------------

    extern "C" __global__ void __miss__radiance()
    {
        referenceTracerPRD prd = loadRayPRD();
        prd.resultColor = { 1.0f, 1.0f, 1.0f };
        storeRayPRD(prd);
    }

    extern "C" __global__ void __miss__shadow()
    {
        referenceTracerPRD prd = loadRayPRD();
        prd.resultColor = { 1.0f, 1.0f, 1.0f };
        storeRayPRD(prd);
    }

    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    extern "C" __global__ void __raygen__renderFrame()
    {
        // compute a test pattern based on pixel ID
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;

        const auto& camera = optixLaunchParams.camera;

        // our per-ray data for this example. what we initialize it to
        // won't matter, since this value will be overwritten by either
        // the miss or hit program, anyway
        referenceTracerPRD pixelColorPRD;

        // the values we store the PRD pointer in:
        uint32_t u0 = 0, u1 = 0, u2 = 0;

        // normalized screen plane position, in [0,1]^2
        const glm::vec2 screen(glm::vec2{ float(ix) + .5f, float(iy) + .5f }
            / glm::vec2{ optixLaunchParams.frame.size });

        // generate ray direction
        glm::vec3 rayDir = glm::normalize(camera.direction
            + (screen.x - 0.5f) * camera.horizontal
            + (screen.y - 0.5f) * camera.vertical);

        float3 camPos = float3{ camera.position.x, camera.position.y, camera.position.z };
        float3 rayDirection = float3{ rayDir.x, rayDir.y, rayDir.z };

        optixTrace(optixLaunchParams.traversable,
            camPos,
            rayDirection,
            0.f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
            RADIANCE_RAY_TYPE,            // SBT offset
            RAY_TYPE_COUNT,               // SBT stride
            RADIANCE_RAY_TYPE,            // missSBTIndex 
            u0, u1, u2);

        pixelColorPRD.resultColor = glm::vec3{ __uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2) };

        const int r = int(255.99f * pixelColorPRD.resultColor.x);
        const int g = int(255.99f * pixelColorPRD.resultColor.y);
        const int b = int(255.99f * pixelColorPRD.resultColor.z);

        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        const uint32_t rgba = 0xff000000
            | (r << 0) | (g << 8) | (b << 16);


        // and write to frame buffer ...
        const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
        optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
    }


}