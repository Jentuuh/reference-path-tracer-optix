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

        float stratifyCellWidth = cellSize / optixLaunchParams.stratifyResX;
        float stratifyCellHeight = cellSize / optixLaunchParams.stratifyResY;
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


        float3 ogLeft{ cellCenter.x - 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z + 0.5f * cellSize };
        float3 ogRight{ cellCenter.x + 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z - 0.5f * cellSize };
        float3 ogUp{ cellCenter.x - 0.5f * cellSize, cellCenter.y + 0.5f * cellSize, cellCenter.z - 0.5f * cellSize };
        float3 ogDown{ cellCenter.x - 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z + 0.5f * cellSize };
        float3 ogFront{ cellCenter.x - 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z - 0.5f * cellSize };
        float3 ogBack{ cellCenter.x + 0.5f * cellSize, cellCenter.y - 0.5f * cellSize, cellCenter.z + 0.5f * cellSize };

        // LEFT, RIGHT, UP, DOWN, FRONT, BACK
        float3 cellNormals[6] = { float3{-1.0f, 0.0f, 0.0f}, float3{1.0f, 0.0f, 0.0f}, float3{0.0f, 1.0f, 0.0f}, float3{0.0f, -1.0f, 0.0f}, float3{0.0f, 0.0f, -1.0f}, float3{0.0f, 0.0f, 1.0f} };
        // Origin, du, dv for each face
        float3 faceOgDuDv[6][3] = { {ogLeft, float3{0.0f, 0.0f, -1.0f}, float3{0.0f, 1.0f, 0.0f} }, {ogRight, float3{0.0f, 0.0f, 1.0f},float3{0.0f, 1.0f, 0.0f} }, {ogUp, float3{1.0f, 0.0f, 0.0f},float3{0.0f, 0.0f, 1.0f} }, {ogDown, float3{1.0f, 0.0f, 0.0f},float3{0.0f, 0.0f, -1.0f}}, {ogFront, float3{1.0f, 0.0f, 0.0f},float3{0.0f, 1.0f, 0.0f} }, {ogBack, float3{-1.0f, 0.0f, 0.0f},float3{0.0f, 1.0f, 0.0f} } };
        // The indices of the SHs that belong to each face, to use while indexing the buffer (L,R,U,D,F,B), (LB, RB, LT, RT)
        int4 cellSHIndices[6] = { int4{4, 0, 6, 2}, int4{1, 5, 3, 7}, int4{2, 3, 6, 7}, int4{4, 5, 0, 1}, int4{0, 1, 2, 3}, int4{5, 4, 7, 6} };
        // Origins of the SHs located on the corners of each cell
        glm::vec3 cellSHOrigins[8] = { ogSh0, ogSh1, ogSh2, ogSh3, ogSh4, ogSh5, ogSh6, ogSh7 };

        // Irradiance accumulator
        float totalIrradiance = 0.0f;
        // Number of samples accumulator
        int numSamples = 0;

        // Loop over cell faces
        for (int face = 0; face < 6; face++)
        {
            // Which SHs of the cell belong to this face
            int4 faceSHIndices = cellSHIndices[face];

            glm::vec3 og = glm::vec3{ faceOgDuDv[face][0].x,faceOgDuDv[face][0].y,faceOgDuDv[face][0].z };
            glm::vec3 du = glm::vec3{ faceOgDuDv[face][1].x,faceOgDuDv[face][1].y,faceOgDuDv[face][1].z };
            glm::vec3 dv = glm::vec3{ faceOgDuDv[face][2].x,faceOgDuDv[face][2].y,faceOgDuDv[face][2].z };

            // Face normal and UV normal need to point in the same direction (hemisphere) for the UV to get contribution from that face
            float cellFaceFacing = dot(uvNormal3f, cellNormals[face]);
            if (cellFaceFacing > 0)
            {
                // For each stratified cell on the face, take samples
                for (int stratifyIndexX = 0; stratifyIndexX < optixLaunchParams.stratifyResX; stratifyIndexX++)
                {
                    for (int stratifyIndexY = 0; stratifyIndexY < optixLaunchParams.stratifyResY; stratifyIndexY++)
                    {
                        glm::vec3 stratifyCellOrigin = og + (stratifyIndexX * stratifyCellWidth * du) + (stratifyIndexY * stratifyCellHeight * dv);

                        // Send out a ray for each sample
                        for (int sample = 0; sample < NUM_SAMPLES_PER_STRATIFY_CELL; sample++)
                        {
                            // Take a random sample on the face's stratified cell, this will be the ray origin
                            float2 randomOffset = float2{ rnd(seed), rnd(seed) };
                            glm::vec3 rayOrigin = stratifyCellOrigin + (randomOffset.x * stratifyCellWidth * du) + (randomOffset.y * stratifyCellHeight * dv);

                            // Ray direction (from the sample to the UV texel)
                            glm::vec3 rayDir = UVWorldPos - rayOrigin;
                            glm::vec3 rayDirInv = rayOrigin - UVWorldPos;

                            // Convert to float3 format
                            float3 rayOrigin3f = float3{ rayOrigin.x, rayOrigin.y, rayOrigin.z };
                            float3 rayDir3f = float3{ rayDir.x, rayDir.y, rayDir.z };
                            float3 rayDir3fInv = float3{ rayDirInv.x, rayDirInv.y, rayDirInv.z };

                            //// Calculate spherical coordinate representation of ray
                            //// (https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates)
                            //float3 normalizedRayDir = normalize(rayDir3f);
                            //double theta = acos(clamp(rayDir3f.z, -1.0, 1.0));
                            //double phi = atan2(rayDir3f.y, rayDir3f.x);

                            RadianceCellScatterPRD prd{};
                            prd.rayOrigin = rayOrigin;

                            unsigned int u0, u1, u2, u3;

                            u1 = __float_as_uint(prd.rayOrigin.x);
                            u2 = __float_as_uint(prd.rayOrigin.y);
                            u3 = __float_as_uint(prd.rayOrigin.z);

                            // Trace ray against scene geometry to see if ray is occluded
                            optixTrace(optixLaunchParams.sceneTraversable,
                                rayOrigin3f,
                                rayDir3f,
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
                            float distanceToUVSquared = (((UVWorldPos.x - rayOrigin.x) * (UVWorldPos.x - rayOrigin.x)) + ((UVWorldPos.y - rayOrigin.y) * (UVWorldPos.y - rayOrigin.y)) + ((UVWorldPos.z - rayOrigin.z) * (UVWorldPos.z - rayOrigin.z)));

                            if (distanceToUVSquared < prd.distanceToClosestIntersectionSquared)
                            {
                                numSamples++;

                                // We calculate the UV world position's offset within the cell to do
                                // the calculation of the weights for trilinear interpolation
                                glm::vec3 diff = UVWorldPos - cellOrigin;

                                // Calculate trilinear interpolation weights, see thesis for explanation
                                glm::vec3 dirTo0 = cellSHOrigins[0] - UVWorldPos;
                                float3 dirTo03f = float3{ dirTo0.x, dirTo0.y, dirTo0.z };
                                float weightA = dot(dirTo03f, uvNormal3f) > 0 ? (diff.x * diff.y * diff.z) * invCellVolume : 0.0f;

                                glm::vec3 dirTo1 = cellSHOrigins[1] - UVWorldPos;
                                float3 dirTo13f = float3{ dirTo1.x, dirTo1.y, dirTo1.z };
                                float weightB = dot(dirTo13f, uvNormal3f) > 0 ? ((1.0f - diff.x) * diff.y * diff.z) * invCellVolume : 0.0f;

                                glm::vec3 dirTo2 = cellSHOrigins[2] - UVWorldPos;
                                float3 dirTo23f = float3{ dirTo2.x, dirTo2.y, dirTo2.z };
                                float weightC = dot(dirTo23f, uvNormal3f) > 0 ? (diff.x * (1.0f - diff.y) * diff.z) * invCellVolume : 0.0f;

                                glm::vec3 dirTo3 = cellSHOrigins[3] - UVWorldPos;
                                float3 dirTo33f = float3{ dirTo3.x, dirTo3.y, dirTo3.z };
                                float weightD = dot(dirTo33f, uvNormal3f) > 0 ? ((1.0f - diff.x) * (1.0f * diff.y) * diff.z) * invCellVolume : 0.0f;

                                glm::vec3 dirTo4 = cellSHOrigins[4] - UVWorldPos;
                                float3 dirTo43f = float3{ dirTo4.x, dirTo4.y, dirTo4.z };
                                float weightE = dot(dirTo43f, uvNormal3f) > 0 ? (diff.x * diff.y * (1.0f - diff.z)) * invCellVolume : 0.0f;

                                glm::vec3 dirTo5 = cellSHOrigins[5] - UVWorldPos;
                                float3 dirTo53f = float3{ dirTo5.x, dirTo5.y, dirTo5.z };
                                float weightF = dot(dirTo53f, uvNormal3f) > 0 ? ((1.0f - diff.x) * diff.y * (1.0f - diff.z)) * invCellVolume : 0.0f;

                                glm::vec3 dirTo6 = cellSHOrigins[6] - UVWorldPos;
                                float3 dirTo63f = float3{ dirTo6.x, dirTo6.y, dirTo6.z };
                                float weightG = dot(dirTo63f, uvNormal3f) > 0 ? (diff.x * (1.0f - diff.y) * (1.0f - diff.z)) * invCellVolume : 0.0f;

                                glm::vec3 dirTo7 = cellSHOrigins[7] - UVWorldPos;
                                float3 dirTo73f = float3{ dirTo7.x, dirTo7.y, dirTo7.z };
                                float weightH = dot(dirTo73f, uvNormal3f) > 0 ? ((1.0f - diff.x) * (1.0f - diff.y) * (1.0f - diff.z)) * invCellVolume : 0.0f;

                                // Basis function evaluations
                                float b0 = Y_0_0();
                                float b1 = Y_min1_1(rayDirInv.x, rayDirInv.y, rayDirInv.z);
                                float b2 = Y_0_1(rayDirInv.x, rayDirInv.y, rayDirInv.z);
                                float b3 = Y_1_1(rayDirInv.x, rayDirInv.y, rayDirInv.z);
                                float b4 = Y_min2_2(rayDirInv.x, rayDirInv.y, rayDirInv.z);
                                float b5 = Y_min1_2(rayDirInv.x, rayDirInv.y, rayDirInv.z);
                                float b6 = Y_0_2(rayDirInv.x, rayDirInv.y, rayDirInv.z);
                                float b7 = Y_1_2(rayDirInv.x, rayDirInv.y, rayDirInv.z);
                                float b8 = Y_2_2(rayDirInv.x, rayDirInv.y, rayDirInv.z);

                                // Calculate the trilinearly interpolated weights to apply to each basis function
                                float w0 = SHweights[0][0] * weightA + SHweights[1][0] * weightB + SHweights[2][0] * weightC + SHweights[3][0] * weightD + SHweights[4][0] * weightE + SHweights[5][0] * weightF + SHweights[6][0] * weightG + SHweights[7][0] * weightH;
                                float w1 = SHweights[0][1] * weightA + SHweights[1][1] * weightB + SHweights[2][1] * weightC + SHweights[3][1] * weightD + SHweights[4][1] * weightE + SHweights[5][1] * weightF + SHweights[6][1] * weightG + SHweights[7][1] * weightH;
                                float w2 = SHweights[0][2] * weightA + SHweights[1][2] * weightB + SHweights[2][2] * weightC + SHweights[3][2] * weightD + SHweights[4][2] * weightE + SHweights[5][2] * weightF + SHweights[6][2] * weightG + SHweights[7][2] * weightH;
                                float w3 = SHweights[0][3] * weightA + SHweights[1][3] * weightB + SHweights[2][3] * weightC + SHweights[3][3] * weightD + SHweights[4][3] * weightE + SHweights[5][3] * weightF + SHweights[6][3] * weightG + SHweights[7][3] * weightH;
                                float w4 = SHweights[0][4] * weightA + SHweights[1][4] * weightB + SHweights[2][4] * weightC + SHweights[3][4] * weightD + SHweights[4][4] * weightE + SHweights[5][4] * weightF + SHweights[6][4] * weightG + SHweights[7][4] * weightH;
                                float w5 = SHweights[0][5] * weightA + SHweights[1][5] * weightB + SHweights[2][5] * weightC + SHweights[3][5] * weightD + SHweights[4][5] * weightE + SHweights[5][5] * weightF + SHweights[6][5] * weightG + SHweights[7][5] * weightH;
                                float w6 = SHweights[0][6] * weightA + SHweights[1][6] * weightB + SHweights[2][6] * weightC + SHweights[3][6] * weightD + SHweights[4][6] * weightE + SHweights[5][6] * weightF + SHweights[6][6] * weightG + SHweights[7][6] * weightH;
                                float w7 = SHweights[0][7] * weightA + SHweights[1][7] * weightB + SHweights[2][7] * weightC + SHweights[3][7] * weightD + SHweights[4][7] * weightE + SHweights[5][7] * weightF + SHweights[6][7] * weightG + SHweights[7][7] * weightH;
                                float w8 = SHweights[0][8] * weightA + SHweights[1][8] * weightB + SHweights[2][8] * weightC + SHweights[3][8] * weightD + SHweights[4][8] * weightE + SHweights[5][8] * weightF + SHweights[6][8] * weightG + SHweights[7][8] * weightH;

                                // SH reconstruction
                                float irradiance = (b0 * w0) + (b1 * w1) + (b2 * w2) + (b3 * w3) + (b4 * w4) + (b5 * w5) + (b6 * w6) + (b7 * w7) + (b8 * w8);
                                float cosContribution = dot(normalize(rayDir3fInv), uvNormal3f);

                                totalIrradiance += cosContribution * irradiance;
                            }
                        }
                    }
                }
            }
        }

        const int r = int(255.99 * (totalIrradiance / float(numSamples / 2.0f)));
        const int g = int(255.99 * (totalIrradiance / float(numSamples / 2.0f)));
        const int b = int(255.99 * (totalIrradiance / float(numSamples / 2.0f)));

        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        const uint32_t rgba = 0xff000000
            | (r << 0) | (g << 8) | (b << 16);

        optixLaunchParams.currentBounceTexture.colorBuffer[v * optixLaunchParams.uvWorldPositions.size + u] = rgba;
    }
}