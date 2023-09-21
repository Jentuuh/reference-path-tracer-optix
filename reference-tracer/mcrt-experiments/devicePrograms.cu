#pragma once

#include <optix_device.h>
#include "random.hpp"
#include "vec_math.hpp"

#include "LaunchParams.hpp"
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#define INDIRECT_BRDF_SAMPLES 5
#define PI 3.14159265358979323846f

using namespace mcrt;

namespace mcrt {
    // Launch parameters in constant memory, filled in by optix upon
    // optixLaunch (this gets filled in from the buffer we pass to
    // optixLaunch)
    extern "C" __constant__ LaunchParamsTutorial optixLaunchParams;


    //static __forceinline__ __device__ referenceTracerPRD loadRayPRD()
    //{
    //    referenceTracerPRD prd = {};

    //    prd.resultColor.x = __uint_as_float(optixGetPayload_0());
    //    prd.resultColor.y = __uint_as_float(optixGetPayload_1());
    //    prd.resultColor.z = __uint_as_float(optixGetPayload_2());

    //    return prd;
    //}

    //static __forceinline__ __device__ void storeRayPRD(referenceTracerPRD prd)
    //{
    //    optixSetPayload_0(__float_as_uint(prd.resultColor.x));
    //    optixSetPayload_1(__float_as_uint(prd.resultColor.y));
    //    optixSetPayload_2(__float_as_uint(prd.resultColor.z));
    //}


    static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
    {
        const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
        void* ptr = reinterpret_cast<void*>(uptr);
        return ptr;
    }


    static __forceinline__ __device__ void  packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
    {
        const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }


    static __forceinline__ __device__ referenceTracerPRD* getPRD()
    {
        const unsigned int u0 = optixGetPayload_0();
        const unsigned int u1 = optixGetPayload_1();
        return reinterpret_cast<referenceTracerPRD*>(unpackPointer(u0, u1));
    }

    static __forceinline__ __device__ void setPayloadOcclusion(bool occluded)
    {
        optixSetPayload_0(static_cast<unsigned int>(occluded));
    }

    static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        referenceTracerPRD* prd
    )
    {
        // TODO: deduce stride from num ray-types passed in params

        unsigned int u0, u1;
        packPointer(prd, u0, u1);
        optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            RADIANCE_RAY_TYPE,        // SBT offset
            RAY_TYPE_COUNT,           // SBT stride
            RADIANCE_RAY_TYPE,        // missSBTIndex
            u0, u1);
    }


    static __forceinline__ __device__ bool traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
    )
    {
        unsigned int occluded = 0u;
        optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            SHADOW_RAY_TYPE,      // SBT offset
            RAY_TYPE_COUNT,          // SBT stride
            SHADOW_RAY_TYPE,      // missSBTIndex
            occluded);
        return occluded;
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
        setPayloadOcclusion(true);
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

        const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDirf3;



        referenceTracerPRD* prd = getPRD();

        // No emission from other geometry
        prd->emitted = glm::vec3{0.0f, 0.0f, 0.0f};


        // Generate random direction on hemisphere
        unsigned int seed = prd->seed;
        {
            float2 uniformRandoms = float2{ rnd(seed), rnd(seed) };
            float randomTheta = uniformRandoms.x * 2 * PI;
            float randomZ = (uniformRandoms.y * 2.0f) - 1.0f;
            float3 randomDir = float3{ sqrtf(1 - (randomZ * randomZ)) * cos(randomTheta), sqrtf(1 - (randomZ * randomZ)) * sin(randomTheta), randomZ };

            // If the generated random direction is not in the oriented hemisphere, invert it
            if (dot(randomDir, float3{Ng.x, Ng.y, Ng.z }) < 0)
            {
                randomDir = float3{ -randomDir.x, -randomDir.y, -randomDir.z };
            }

            prd->direction = glm::vec3{ randomDir.x, randomDir.y, randomDir.z } ;
            prd->origin = glm::vec3{P.x, P.y, P.z};

            prd->attenuation *= diffuseColor;
            prd->countEmitted = false;
        }
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);
        prd->seed = seed;

        LightData light = optixLaunchParams.lights[0];
        const glm::vec3 lightSamplePos = light.origin + (z1 * light.width * light.du) + (z2 * light.height * light.dv);

        const float distanceToLightSample = length(lightSamplePos - glm::vec3{ P.x, P.y, P.z });
        const glm::vec3 dirToLightSample = normalize(lightSamplePos - glm::vec3{P.x, P.y, P.z});
        float angleRayGeometry = dot(Ng, dirToLightSample);
        float angleLightRay = -dot(light.normal, dirToLightSample);
        float weight = 0.0f;

        if (angleRayGeometry > 0.0f && angleLightRay > 0.0f)
        {
            const bool occluded = traceOcclusion(
                optixLaunchParams.traversable,
                P,
                float3{dirToLightSample.x, dirToLightSample.y, dirToLightSample.z },
                0.01f,         // tmin
                distanceToLightSample - 0.01f  // tmax
            );

            if (!occluded)
            {
                const float A = length(cross(light.du * light.width, light.dv * light.height));
                weight = angleRayGeometry * angleLightRay * A / (PI * distanceToLightSample * distanceToLightSample);
            }
        }
        prd->radiance += light.power * weight;
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
        referenceTracerPRD* prd = getPRD();

        prd->radiance = glm::vec3{0.0f, 0.0f, 0.0f};
        prd->emitted = glm::vec3{0.0f, 0.0f, 0.0f };
        prd->done = true;
    }

    extern "C" __global__ void __miss__shadow()
    {
        // Not used
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
        unsigned int seed = tea<4>(ix + iy * optixLaunchParams.frame.size.x, iy);


        glm::vec3 result = {0.0f, 0.0f, 0.0f};
        int i = optixLaunchParams.samplesPerPixel;
        do
        {
            const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));
            const glm::vec2 d = 2.0f * glm::vec2{
                (static_cast<float>(ix) + subpixel_jitter.x) / static_cast<float>(optixLaunchParams.frame.size.x),
                (static_cast<float>(iy) + subpixel_jitter.y) / static_cast<float>(optixLaunchParams.frame.size.y)
            } - 1.0f;
            glm::vec3 ray_direction = normalize(d.x * camera.horizontal + d.y * camera.vertical + camera.direction);
            float3 rayDirection3f = float3{ ray_direction.x, ray_direction.y, ray_direction.z };
            float3 camPos = float3{ camera.position.x, camera.position.y, camera.position.z };

            referenceTracerPRD prd;
            prd.emitted = glm::vec3(0.0f, 0.0f, 0.0f);
            prd.radiance = glm::vec3(0.0f, 0.0f, 0.0f);
            prd.attenuation = glm::vec3(1.0f, 1.0f, 1.0f);
            prd.countEmitted = true;
            prd.done = false;
            prd.seed = seed;

            // ===============
            // Trace the path
            // ===============
            int depth = 0;
            for (;;)
            {
                traceRadiance(
                    optixLaunchParams.traversable,
                    camPos,
                    rayDirection3f,
                    0.01f,  // tmin       // TODO: smarter offset
                    1e16f,  // tmax
                    &prd);

                result += prd.emitted;
                result += prd.radiance * prd.attenuation;

                if (prd.done || depth >= 2) // TODO RR, variable for depth
                    break;

                camPos = float3{prd.origin.x, prd.origin.y, prd.origin.z };
                rayDirection3f = float3{ prd.direction.x, prd.direction.y, prd.direction.z };

                ++depth;
            }
        } while (--i);

        const uint3    launch_index = optixGetLaunchIndex();
        const unsigned int fbIndex = iy * optixLaunchParams.frame.size.x + ix;
        glm::vec3      accum_color = result / static_cast<float>(optixLaunchParams.samplesPerPixel);

        // ==========================
        // HDR Exposure Tone Mapping
        // ==========================
        const float exposure = 0.8f;
        const float gamma = 2.2f;

        glm::vec3 mapped = glm::vec3{ 1.0f, 1.0f, 1.0f } - glm::vec3{ exp(-accum_color.x * exposure), exp(-accum_color.y * exposure), exp(-accum_color.z * exposure) };

        // Gamma correction 
        mapped = glm::vec3{ pow(mapped.x, 1.0f / gamma), pow(mapped.y, 1.0f / gamma), pow(mapped.z, 1.0f / gamma) };


        const int r = int(255.99f * mapped.x);
        const int g = int(255.99f * mapped.y);
        const int b = int(255.99f * mapped.z);

        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        const uint32_t rgba = 0xff000000
            | (r << 0) | (g << 8) | (b << 16);

        optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
    }


}