#pragma once

#include <optix_device.h>
#include "random.hpp"
#include "vec_math.hpp"

#include "LaunchParams.hpp"
#include "glm/glm.hpp"

#include "cube_mapping.cuh"
#include "utils.cuh"

#define PI 3.14159265358979323846f
#define EPSILON 0.0000000000002f
#define NUM_DIRECTION_SAMPLES 400
#define PI_OVER_4 0.785398163397425f
#define PI_OVER_2 1.5707963267945f

using namespace mcrt;


namespace mcrt {
    extern "C" __constant__ LaunchParamsRadianceCellScatterUnbiased optixLaunchParams;

    static __forceinline__ __device__ RadianceCellScatterUnbiasedPRD loadRadianceCellScatterUnbiasedPRD()
    {
        RadianceCellScatterUnbiasedPRD prd = {};

        prd.resultColor.x = __uint_as_float(optixGetPayload_0());
        prd.resultColor.y = __uint_as_float(optixGetPayload_1());
        prd.resultColor.z = __uint_as_float(optixGetPayload_2());

        return prd;
    }

    static __forceinline__ __device__ void storeRadianceCellScatterUnbiasedPRD(RadianceCellScatterUnbiasedPRD prd)
    {
        optixSetPayload_0(__float_as_uint(prd.resultColor.x));
        optixSetPayload_1(__float_as_uint(prd.resultColor.y));
        optixSetPayload_2(__float_as_uint(prd.resultColor.z));
    }


    extern "C" __global__ void __closesthit__radiance__cell__scattering__scene__unbiased()
    {
        const MeshSBTDataRadianceCellScatter& sbtData
            = *(const MeshSBTDataRadianceCellScatter*)optixGetSbtDataPointer();

        const int primID = optixGetPrimitiveIndex();
        const glm::ivec3 index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        // Barycentric tex coords
        const glm::vec2 tc
            = (1.f - u - v) * sbtData.texcoord[index.x]
            + u * sbtData.texcoord[index.y]
            + v * sbtData.texcoord[index.z];

        const int uTexelCoord = tc.x * optixLaunchParams.prevBounceTexture.size;
        const int vTexelCoord = tc.y * optixLaunchParams.prevBounceTexture.size;

        // Read color (outgoing radiance) at intersection (NOTE THAT WE ASSUME LAMBERTIAN SURFACE HERE)
        // --> Otherwise BRDF needs to be evaluated for the incoming direction at this point
        float r = optixLaunchParams.prevBounceTexture.colorBuffer[(vTexelCoord * optixLaunchParams.prevBounceTexture.size * 3) + (uTexelCoord * 3) + 0];
        float g = optixLaunchParams.prevBounceTexture.colorBuffer[(vTexelCoord * optixLaunchParams.prevBounceTexture.size * 3) + (uTexelCoord * 3) + 1];
        float b = optixLaunchParams.prevBounceTexture.colorBuffer[(vTexelCoord * optixLaunchParams.prevBounceTexture.size * 3) + (uTexelCoord * 3) + 2];

        RadianceCellScatterUnbiasedPRD prd = loadRadianceCellScatterUnbiasedPRD();
        prd.resultColor = glm::vec3{r,g,b};

        storeRadianceCellScatterUnbiasedPRD(prd);
    }

    extern "C" __global__ void __anyhit__radiance__cell__scattering__scene__unbiased() {
        // Do nothing
    }

    extern "C" __global__ void __miss__radiance__cell__scattering__unbiased()
    {
        RadianceCellScatterUnbiasedPRD prd = loadRadianceCellScatterUnbiasedPRD();
        prd.resultColor = { 0.0f, 0.0f, 0.0f };
        storeRadianceCellScatterUnbiasedPRD(prd);
    }

    extern "C" __global__ void __raygen__renderFrame__cell__scattering__unbiased()
    {
        const int uvIndex = optixGetLaunchIndex().x;
        const int nonEmptyCellIndex = optixLaunchParams.nonEmptyCellIndex;

        // Take different seed for each radiance cell face
        unsigned int seed = tea<4>(uvIndex, nonEmptyCellIndex);

        // Get UV world position for this shader pass
        const int uvInsideOffset = optixLaunchParams.uvsInsideOffsets[nonEmptyCellIndex];
        glm::vec2 uv = optixLaunchParams.uvsInside[uvInsideOffset + uvIndex];
        const int u = int(uv.x * optixLaunchParams.uvWorldPositions.size);
        const int v = int(uv.y * optixLaunchParams.uvWorldPositions.size);


        glm::vec3 UVWorldPos = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.uvWorldPositions.size + u].worldPosition;
        const glm::vec3 UVNormal = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.uvWorldPositions.size + u].worldNormal;
        const glm::vec3 diffuseColor = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.uvWorldPositions.size + u].diffuseColor;

        // Small offset to world position to 'mitigate' floating point errors
        UVWorldPos = glm::vec3{ UVWorldPos.x + UVNormal.x * 0.00001f, UVWorldPos.y + UVNormal.y * 0.00001f, UVWorldPos.z + UVNormal.z * 0.00001f };

        float3 rayOrigin3f = float3{ UVWorldPos.x, UVWorldPos.y, UVWorldPos.z };
        float3 uvNormal3f = float3{ UVNormal.x, UVNormal.y, UVNormal.z };

        //// ==============================================================================
        //// Calculate rotation matrix to align generated directions with normal hemisphere
        //// ==============================================================================
        //float3 up = float3{ 0.0f, 1.0f, 0.0f };
        //glm::mat3x3 rotationMatrix;
        //if (uvNormal3f.x == 0.0f && uvNormal3f.y == -1.0f && uvNormal3f.z == 0.0f)
        //{
        //    float3 rotAxis = cross(up, uvNormal3f);
        //    float sine = length(rotAxis);
        //    float cosine = dot(up, uvNormal3f);
        //    glm::mat3x3 v_x = glm::mat3x3{ {0.0f, rotAxis.z, -rotAxis.y}, {-rotAxis.z, 0.0f, rotAxis.x}, {rotAxis.y, -rotAxis.x, 0.0f} };
        //    glm::mat3x3 v_xSquared = v_x * v_x;
        //    float endFactor = 1.0f / (1 + cosine);
        //    rotationMatrix = glm::mat3x3(1.0f) + v_x + (v_xSquared * endFactor);
        //}
        //else {
        //    // 180 degrees rotation around the X-axis (flips along Y-axis)
        //    rotationMatrix = glm::mat3x3{ {1.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, -1.0f} };
        //}

        // ======================================
        // Radiance + num of samples accumulators
        // ======================================
        glm::vec3 totalRadiance = glm::vec3{ 0.0f, 0.0f, 0.0f };
        int numSamples = 0;

        for (int i = 0; i < NUM_DIRECTION_SAMPLES; i++)
        {
            // =============================================================================================================================================================================
            // Random direction generation (equal-area projection of sphere onto rectangle)  : https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d
            // =============================================================================================================================================================================
            float2 uniformRandoms = float2{ rnd(seed), rnd(seed) };
            float randomTheta = uniformRandoms.x * 2 * PI;
            float randomZ = (uniformRandoms.y * 2.0f) - 1.0f;
            float3 randomDir = float3{ sqrtf(1 - (randomZ * randomZ)) * cos(randomTheta), sqrtf(1 - (randomZ * randomZ)) * sin(randomTheta), randomZ };

            // If the generated random direction is not in the oriented hemisphere, invert it
            if (dot(randomDir, uvNormal3f) < 0)
            {
                randomDir = float3{-randomDir.x, -randomDir.y, -randomDir.z};
            }

            //// =================================================================================================================================================================================
            //// Random direction generation (uniform direction generation with spherical coords)  : https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d
            //// =================================================================================================================================================================================
            //float2 uniformRandoms = float2{ rnd(seed), rnd(seed) };
            //float theta = acos(1.0f - (2.0f * uniformRandoms.x));
            //float phi = 2 * PI * uniformRandoms.y;
            //float3 randomDir = float3{ sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta) };
            //
            //// If the generated random direction is not in the oriented hemisphere, invert it
            //if (dot(normalize(randomDir), normalize(uvNormal3f)) < 0)
            //{
            //    randomDir = float3{-randomDir.x, -randomDir.y, -randomDir.z};
            //}



            //// ===================================================
            //// Concentric sampling on unit disk (see pbrt, p. 778)
            //// ===================================================
            //glm::vec2 randomPoint;
            //glm::vec2 uniformRandoms = glm::vec2{ rnd(seed), rnd(seed) };
            //glm::vec2 remappedRandoms = 2.0f * uniformRandoms - glm::vec2{1.0f, 1.0f};
            //if (remappedRandoms.x == 0.0f && remappedRandoms.y == 0.0f)
            //{
            //    randomPoint = glm::vec2{0.0f, 0.0f};
            //}
            //else {
            //    float theta, r;
            //    if (abs(remappedRandoms.x) > abs(remappedRandoms.y))
            //    {
            //        r = remappedRandoms.x;
            //        theta = PI_OVER_4 * (remappedRandoms.y / remappedRandoms.x);
            //    }
            //    else {
            //        r = remappedRandoms.y;
            //        theta = PI_OVER_2 - (PI_OVER_4 * (remappedRandoms.x / remappedRandoms.y));
            //    }
            //    randomPoint = r * glm::vec2{cos(theta), sin(theta)};
            //}
            //
            //// Calculate z so vector is normalized (project the random point on the unit disk upwards)
            //float z = sqrtf(fmaxf(0.0f, 1.0f - (randomPoint.x * randomPoint.x) - (randomPoint.y * randomPoint.y)));
            //glm::vec3 randomDirInSampleSystem = glm::vec3{randomPoint.x, randomPoint.y, z};

            //// Rotate random direction to the normal-oriented hemisphere
            //glm::vec3 randomDirInHemisphere = rotationMatrix * randomDirInSampleSystem;
            //float3 randomDir = float3{ randomDirInHemisphere.x, randomDirInHemisphere.y, randomDirInHemisphere.z };

            RadianceCellScatterUnbiasedPRD prd;
            unsigned int u0, u1, u2;
            // Trace ray against scene geometry to see if ray is occluded
            optixTrace(optixLaunchParams.sceneTraversable,
                rayOrigin3f,
                randomDir,
                0.f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,      // We only need closest-hit for scene geometry
                0,  // SBT offset
                1,  // SBT stride
                0,  // missSBTIndex
                u0, u1, u2
            );

            prd.resultColor = glm::vec3{__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2)};

            // Cosine weighted contribution
            float cosContribution = dot(normalize(randomDir), normalize(uvNormal3f));
            totalRadiance += glm::vec3{ cosContribution * prd.resultColor.x, cosContribution * prd.resultColor.y, cosContribution * prd.resultColor.z };
            ++numSamples;
        }

        // Monte-Carlo weighted estimation
        const float r_result = totalRadiance.x / (float(numSamples) * PI);
        const float g_result = totalRadiance.y / (float(numSamples) * PI);
        const float b_result = totalRadiance.z / (float(numSamples) * PI);
        
        optixLaunchParams.currentBounceTexture.colorBuffer[(v * optixLaunchParams.currentBounceTexture.size * 3) + (u * 3) + 0] = r_result;
        optixLaunchParams.currentBounceTexture.colorBuffer[(v * optixLaunchParams.currentBounceTexture.size * 3) + (u * 3) + 1] = g_result;
        optixLaunchParams.currentBounceTexture.colorBuffer[(v * optixLaunchParams.currentBounceTexture.size * 3) + (u * 3) + 2] = b_result;
    }
}