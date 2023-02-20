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
        glm::vec3 cellOrigin = { cellCenter.x - 0.5f * cellSize,cellCenter.y - 0.5f * cellSize,cellCenter.z - 0.5f * cellSize };    // Origin corner point of the cell

        float invCellVolume = 1.0f / (cellSize * cellSize * cellSize);

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

        // The indices of the SHs that belong to each face, to use while indexing the buffer (L,R,U,D,F,B), (LB, RB, LT, RT)
        int4 cellSHIndices[6] = { int4{4, 0, 6, 2}, int4{1, 5, 3, 7}, int4{2, 3, 6, 7}, int4{4, 5, 0, 1}, int4{0, 1, 2, 3}, int4{5, 4, 7, 6} };
        // Origins of the SHs located on the corners of each cell
        glm::vec3 cellSHOrigins[8] = { ogSh0, ogSh1, ogSh2, ogSh3, ogSh4, ogSh5, ogSh6, ogSh7 };

        // We calculate the UV world position's offset within the cell to do
        // the calculation of the weights for trilinear interpolation
        glm::vec3 diff = UVWorldPos - cellOrigin;

        // Calculate trilinear interpolation weights, see thesis for explanation
        float weightA = (diff.x * diff.y * diff.z) * invCellVolume;
        float weightB = ((1.0f - diff.x) * diff.y * diff.z) * invCellVolume;
        float weightC = (diff.x * (1.0f - diff.y) * diff.z) * invCellVolume;
        float weightD = ((1.0f - diff.x) * (1.0f * diff.y) * diff.z) * invCellVolume;
        float weightE = (diff.x * diff.y * (1.0f - diff.z)) * invCellVolume;
        float weightF = ((1.0f - diff.x) * diff.y * (1.0f - diff.z)) * invCellVolume;
        float weightG = (diff.x * (1.0f - diff.y) * (1.0f - diff.z)) * invCellVolume;
        float weightH = ((1.0f - diff.x) * (1.0f - diff.y) * (1.0f - diff.z)) * invCellVolume;
        float trilinearWeights[8] = { weightA, weightB, weightC, weightD, weightE, weightF, weightG, weightH };

        // Irradiance accumulator
        float totalIrradiance = 0.0f;
        // Number of samples accumulator
        int numSamples = 0;

        for (int sh = 0; sh < 8; sh++)
        {
            glm::vec3 rayDir = cellSHOrigins[sh] - UVWorldPos;
            glm::vec3 rayDirInv = UVWorldPos - cellSHOrigins[sh];
            float3 rayOrigin3f = { cellSHOrigins[sh].x, cellSHOrigins[sh].y, cellSHOrigins[sh].z };

            float3 rayDir3f = { rayDir.x, rayDir.y, rayDir.z };
            float3 rayDir3fInv = { rayDirInv.x, rayDirInv.y, rayDirInv.z };

            RadianceCellScatterPRD prd{};
            prd.rayOrigin = cellSHOrigins[sh];

            unsigned int u0, u1, u2, u3;

            u1 = __float_as_uint(prd.rayOrigin.x);
            u2 = __float_as_uint(prd.rayOrigin.y);
            u3 = __float_as_uint(prd.rayOrigin.z);

            // Trace ray against scene geometry to see if ray is occluded
            optixTrace(optixLaunchParams.sceneTraversable,
                rayOrigin3f,
                rayDir3fInv,
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
            float distanceToUVSquared = (((UVWorldPos.x - cellSHOrigins[sh].x) * (UVWorldPos.x - cellSHOrigins[sh].x)) + ((UVWorldPos.y - cellSHOrigins[sh].y) * (UVWorldPos.y - cellSHOrigins[sh].y)) + ((UVWorldPos.z - cellSHOrigins[sh].z) * (UVWorldPos.z - cellSHOrigins[sh].z)));

            if (distanceToUVSquared < prd.distanceToClosestIntersectionSquared)
            {
                numSamples++;

                // Basis function evaluations
                float b0 = Y_0_0();
                float b1 = Y_min1_1(rayDir.x, rayDir.y, rayDir.z);
                float b2 = Y_0_1(rayDir.x, rayDir.y, rayDir.z);
                float b3 = Y_1_1(rayDir.x, rayDir.y, rayDir.z);
                float b4 = Y_min2_2(rayDir.x, rayDir.y, rayDir.z);
                float b5 = Y_min1_2(rayDir.x, rayDir.y, rayDir.z);
                float b6 = Y_0_2(rayDir.x, rayDir.y, rayDir.z);
                float b7 = Y_1_2(rayDir.x, rayDir.y, rayDir.z);
                float b8 = Y_2_2(rayDir.x, rayDir.y, rayDir.z);

                // Calculate the trilinearly interpolated weights to apply to each basis function
                float w0 = SHweights[sh][0] * trilinearWeights[sh];
                float w1 = SHweights[sh][1] * trilinearWeights[sh];
                float w2 = SHweights[sh][2] * trilinearWeights[sh];
                float w3 = SHweights[sh][3] * trilinearWeights[sh];
                float w4 = SHweights[sh][4] * trilinearWeights[sh];
                float w5 = SHweights[sh][5] * trilinearWeights[sh];
                float w6 = SHweights[sh][6] * trilinearWeights[sh];
                float w7 = SHweights[sh][7] * trilinearWeights[sh];
                float w8 = SHweights[sh][8] * trilinearWeights[sh];

                // SH reconstruction
                float irradiance = (b0 * w0) + (b1 * w1) + (b2 * w2) + (b3 * w3) + (b4 * w4) + (b5 * w5) + (b6 * w6) + (b7 * w7) + (b8 * w8);
      /*          float cosContribution = dot(normalize(rayDir3f), uvNormal3f);

                if (cosContribution > 0.0f) {
                    totalIrradiance += cosContribution * irradiance;
                }*/
                totalIrradiance += irradiance;
            }
        }


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