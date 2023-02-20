#pragma once

#include <optix_device.h>
#include "random.hpp"
#include "vec_math.hpp"

#include "LaunchParams.hpp"
#include "glm/glm.hpp"

#include "spherical_harmonics.cuh"

#define PI 3.14159265358979323846f
#define EPSILON 0.0000000000002f
#define NUM_SAMPLES_PER_STRATIFY_CELL 5

using namespace mcrt;


namespace mcrt {
    extern "C" __constant__ LaunchParamsRadianceCellScatter optixLaunchParams;

    static __forceinline__ __device__ RadianceCellScatterPRD loadRadianceCellScatterPRD()
    {
        RadianceCellScatterPRD prd = {};

        prd.distanceToClosestIntersectionSquared = __uint_as_float(optixGetPayload_0());
        prd.rayOrigin.x = __uint_as_float(optixGetPayload_1());
        prd.rayOrigin.y = __uint_as_float(optixGetPayload_2());
        prd.rayOrigin.z = __uint_as_float(optixGetPayload_3());

        return prd;
    }

    static __forceinline__ __device__ void storeRadianceCellScatterPRD(RadianceCellScatterPRD prd)
    {
        optixSetPayload_0(__float_as_uint(prd.distanceToClosestIntersectionSquared));
        optixSetPayload_1(__float_as_uint(prd.rayOrigin.x));
        optixSetPayload_2(__float_as_uint(prd.rayOrigin.y));
        optixSetPayload_3(__float_as_uint(prd.rayOrigin.z));
    }


    extern "C" __global__ void __closesthit__radiance__cell__scattering__scene()
    {
        const MeshSBTDataRadianceCellScatter& sbtData
            = *(const MeshSBTDataRadianceCellScatter*)optixGetSbtDataPointer();

        const int primID = optixGetPrimitiveIndex();
        const glm::ivec3 index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        const glm::vec3 intersectionWorldPos =
            (1.f - u - v) * sbtData.vertex[index.x]
            + u * sbtData.vertex[index.y]
            + v * sbtData.vertex[index.z];

        RadianceCellScatterPRD prd = loadRadianceCellScatterPRD();
        float distanceToIntersectionSquared = (((intersectionWorldPos.x - prd.rayOrigin.x) * (intersectionWorldPos.x - prd.rayOrigin.x)) + ((intersectionWorldPos.y - prd.rayOrigin.y) * (intersectionWorldPos.y - prd.rayOrigin.y)) + ((intersectionWorldPos.z - prd.rayOrigin.z) * (intersectionWorldPos.z - prd.rayOrigin.z)));

        prd.distanceToClosestIntersectionSquared = distanceToIntersectionSquared;
        storeRadianceCellScatterPRD(prd);
    }

    extern "C" __global__ void __anyhit__radiance__cell__scattering__scene() {
        // Do nothing
    }

    extern "C" __global__ void __miss__radiance__cell__scattering()
    {
        // Do nothing
    }

    extern "C" __global__ void __raygen__renderFrame__cell__scattering()
    {
        const int uvIndex = optixGetLaunchIndex().x;

        // Take different seed for each radiance cell face
        unsigned int seed = tea<4>(uvIndex, optixLaunchParams.nonEmptyCellIndex);

        // Get UV world position for this shader pass
        const int uvInsideOffset = optixLaunchParams.uvsInsideOffsets[optixLaunchParams.nonEmptyCellIndex];
        glm::vec2 uv = optixLaunchParams.uvsInside[uvInsideOffset + uvIndex];
        const int u = int(uv.x * optixLaunchParams.uvWorldPositions.size);
        const int v = int(uv.y * optixLaunchParams.uvWorldPositions.size);

        glm::vec3 UVWorldPos = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.uvWorldPositions.size + u].worldPosition;
        const glm::vec3 UVNormal = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.uvWorldPositions.size + u].worldNormal;
        float3 uvNormal3f = float3{ UVNormal.x, UVNormal.y, UVNormal.z };

        // We apply a small offset of 0.00001f in the direction of the normal to the UV world pos, to 'mitigate' floating point rounding errors causing false occlusions/illuminations
        UVWorldPos = glm::vec3{ UVWorldPos.x + UVNormal.x * 0.0001f, UVWorldPos.y + UVNormal.y * 0.0001f, UVWorldPos.z + UVNormal.z * 0.0001f };

        // Center of this radiance cell
        glm::vec3 cellCenter = optixLaunchParams.cellCenter;
        float cellSize = optixLaunchParams.cellSize;

        // SH weights for this cell
        float SHweights[8][9];
        int amountBasisFunctions = optixLaunchParams.sphericalHarmonicsWeights.amountBasisFunctions;
        int cellOffset = optixLaunchParams.nonEmptyCellIndex * amountBasisFunctions * 8;

        // Load in SH weights from GPU buffer
        for (int sh_i = 0; sh_i < 8; sh_i++)
        {
            for (int basis_f_i = 0; basis_f_i < 9; basis_f_i++)
            {
                SHweights[sh_i][basis_f_i] = optixLaunchParams.sphericalHarmonicsWeights.weights[cellOffset + sh_i * amountBasisFunctions + basis_f_i];
            }
        }

        glm::vec3 ogSh0 = { cellCenter.x - 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z - 0.5f * cellSize };
        glm::vec3 ogSh1 = { cellCenter.x + 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z - 0.5f * cellSize };
        glm::vec3 ogSh2 = { cellCenter.x - 0.5f * cellSize, cellCenter.y + 0.5f * cellSize, cellCenter.z - 0.5f * cellSize };
        glm::vec3 ogSh3 = { cellCenter.x + 0.5f * cellSize, cellCenter.y + 0.5f * cellSize, cellCenter.z - 0.5f * cellSize };
        glm::vec3 ogSh4 = { cellCenter.x - 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z + 0.5f * cellSize };
        glm::vec3 ogSh5 = { cellCenter.x + 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z + 0.5f * cellSize };
        glm::vec3 ogSh6 = { cellCenter.x - 0.5f * cellSize, cellCenter.y + 0.5f * cellSize, cellCenter.z + 0.5f * cellSize };
        glm::vec3 ogSh7 = { cellCenter.x + 0.5f * cellSize, cellCenter.y + 0.5f * cellSize, cellCenter.z + 0.5f * cellSize };


        // Origins of the SHs located on the corners of each cell
        glm::vec3 cellSHOrigins[8] = { ogSh0, ogSh1, ogSh2, ogSh3, ogSh4, ogSh5, ogSh6, ogSh7 };

        // Irradiance accumulator
        float totalIrradiance = 0.0f;
        // Number of samples accumulator
        int numSamples = 0;

        // Calculate trilinear interpolation weights, see thesis for explanation
        glm::vec3 dirTo0 = UVWorldPos - cellSHOrigins[0];
        glm::vec3 dirTo0Inv = cellSHOrigins[0] - UVWorldPos;
        float3 dirTo03f = float3{ dirTo0.x, dirTo0.y, dirTo0.z };
        float3 dirTo03fInv = float3{ dirTo0Inv.x, dirTo0Inv.y, dirTo0Inv.z };

        glm::vec3 dirTo1 = UVWorldPos - cellSHOrigins[1];
        glm::vec3 dirTo1Inv = cellSHOrigins[1] - UVWorldPos;
        float3 dirTo13f = float3{ dirTo1.x, dirTo1.y, dirTo1.z };
        float3 dirTo13fInv = float3{ dirTo1Inv.x, dirTo1Inv.y, dirTo1Inv.z };

        glm::vec3 dirTo2 = UVWorldPos - cellSHOrigins[2];
        glm::vec3 dirTo2Inv = cellSHOrigins[2] - UVWorldPos;
        float3 dirTo23f = float3{ dirTo2.x, dirTo2.y, dirTo2.z };
        float3 dirTo23fInv = float3{ dirTo2Inv.x, dirTo2Inv.y, dirTo2Inv.z };

        glm::vec3 dirTo3 = UVWorldPos - cellSHOrigins[3];
        glm::vec3 dirTo3Inv = cellSHOrigins[3] - UVWorldPos;
        float3 dirTo33f = float3{ dirTo3.x, dirTo3.y, dirTo3.z };
        float3 dirTo33fInv = float3{ dirTo3Inv.x, dirTo3Inv.y, dirTo3Inv.z };

        glm::vec3 dirTo4 = UVWorldPos - cellSHOrigins[4];
        glm::vec3 dirTo4Inv = cellSHOrigins[4] - UVWorldPos;
        float3 dirTo43f = float3{ dirTo4.x, dirTo4.y, dirTo4.z };
        float3 dirTo43fInv = float3{ dirTo4Inv.x, dirTo4Inv.y, dirTo4Inv.z };

        glm::vec3 dirTo5 = UVWorldPos - cellSHOrigins[5];
        glm::vec3 dirTo5Inv = cellSHOrigins[5] - UVWorldPos;
        float3 dirTo53f = float3{ dirTo5.x, dirTo5.y, dirTo5.z };
        float3 dirTo53fInv = float3{ dirTo5Inv.x, dirTo5Inv.y, dirTo5Inv.z };

        glm::vec3 dirTo6 = UVWorldPos - cellSHOrigins[6];
        glm::vec3 dirTo6Inv = cellSHOrigins[6] - UVWorldPos;
        float3 dirTo63f = float3{ dirTo6.x, dirTo6.y, dirTo6.z };
        float3 dirTo63fInv = float3{ dirTo6Inv.x, dirTo6Inv.y, dirTo6Inv.z };

        glm::vec3 dirTo7 = UVWorldPos - cellSHOrigins[7];
        glm::vec3 dirTo7Inv = cellSHOrigins[7] - UVWorldPos;
        float3 dirTo73f = float3{ dirTo7.x, dirTo7.y, dirTo7.z };
        float3 dirTo73fInv = float3{ dirTo7Inv.x, dirTo7Inv.y, dirTo7Inv.z };

        float3 directions[8] = { dirTo03f, dirTo13f, dirTo23f, dirTo33f, dirTo43f, dirTo53f, dirTo63f, dirTo73f };
        float3 directionsInv[8] = { dirTo03fInv, dirTo13fInv, dirTo23fInv, dirTo33fInv, dirTo43fInv, dirTo53fInv, dirTo63fInv, dirTo73fInv };


        for (int sh = 0; sh < 8; sh++)
        {
            // Check if light probe is facing UV texel
            if (dot(directions[sh], uvNormal3f) > 0) {
                continue;
            }

            glm::vec3 rayOrigin = cellSHOrigins[sh];
            float3 rayOrigin3f = { rayOrigin.x, rayOrigin.y, rayOrigin.z };

            RadianceCellScatterPRD prd{};
            prd.rayOrigin = rayOrigin;

            unsigned int u0, u1, u2, u3;

            u0 = __float_as_uint(1000.0f); // Initialize distanceToProxySquared at 1000.0f, so in case the ray misses all geometry, the distance to the proxy is 'infinite'
            u1 = __float_as_uint(prd.rayOrigin.x);
            u2 = __float_as_uint(prd.rayOrigin.y);
            u3 = __float_as_uint(prd.rayOrigin.z);

            // Trace ray against scene geometry to see if ray is occluded
            optixTrace(optixLaunchParams.sceneTraversable,
                rayOrigin3f,
                directions[sh],
                0.f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,      // We only need closest-hit for scene geometry
                0,  // SBT offset
                1,  // SBT stride
                0,  // missSBTIndex
                u0, u1, u2, u3
            );

            prd.distanceToClosestIntersectionSquared = __uint_as_float(u0);

            // Check if ray is occluded
            float distanceToUVSquared = (((UVWorldPos.x - rayOrigin.x) * (UVWorldPos.x - rayOrigin.x)) + ((UVWorldPos.y - rayOrigin.y) * (UVWorldPos.y - rayOrigin.y)) + ((UVWorldPos.z - rayOrigin.z) * (UVWorldPos.z - rayOrigin.z)));
            if (distanceToUVSquared < (prd.distanceToClosestIntersectionSquared - 0.00001f))
            {
                numSamples++;

                float3 normalizedDirection = normalize(directions[sh]);

                // Basis function evaluations
                float b0 = Y_0_0();
                float b1 = Y_min1_1(normalizedDirection.x, normalizedDirection.y, normalizedDirection.z);
                float b2 = Y_0_1(normalizedDirection.x, normalizedDirection.y, normalizedDirection.z);
                float b3 = Y_1_1(normalizedDirection.x, normalizedDirection.y, normalizedDirection.z);
                float b4 = Y_min2_2(normalizedDirection.x, normalizedDirection.y, normalizedDirection.z);
                float b5 = Y_min1_2(normalizedDirection.x, normalizedDirection.y, normalizedDirection.z);
                float b6 = Y_0_2(normalizedDirection.x, normalizedDirection.y, normalizedDirection.z);
                float b7 = Y_1_2(normalizedDirection.x, normalizedDirection.y, normalizedDirection.z);
                float b8 = Y_2_2(normalizedDirection.x, normalizedDirection.y, normalizedDirection.z);

                // SH reconstruction
                float irradiance = (b0 * SHweights[sh][0]) + (b1 * SHweights[sh][1]) + (b2 * SHweights[sh][2]) + (b3 * SHweights[sh][3]) + (b4 * SHweights[sh][4]) + (b5 * SHweights[sh][5]) + (b6 * SHweights[sh][6]) + (b7 * SHweights[sh][7]) + (b8 * SHweights[sh][8]);
                float cosContribution = dot(normalize(directionsInv[sh]), uvNormal3f);

                totalIrradiance += cosContribution * irradiance;
            }
        }

        // totalIrradiance *= 10.0f;

        const int r = int(255.99 * (totalIrradiance));
        const int g = int(255.99 * (totalIrradiance));
        const int b = int(255.99 * (totalIrradiance));

        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        const uint32_t rgba = 0xff000000
            | (r << 0) | (g << 8) | (b << 16);

        optixLaunchParams.currentBounceTexture.colorBuffer[v * optixLaunchParams.uvWorldPositions.size + u] = rgba;
    }
}


