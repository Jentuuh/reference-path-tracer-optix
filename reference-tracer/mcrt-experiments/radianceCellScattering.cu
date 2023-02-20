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
        float distanceToIntersection = (((intersectionWorldPos.x - prd.rayOrigin.x) * (intersectionWorldPos.x - prd.rayOrigin.x)) + ((intersectionWorldPos.y - prd.rayOrigin.y) * (intersectionWorldPos.y - prd.rayOrigin.y)) + ((intersectionWorldPos.z - prd.rayOrigin.z) * (intersectionWorldPos.z - prd.rayOrigin.z)));

        prd.distanceToClosestIntersectionSquared = distanceToIntersection;
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
        glm::vec2 uv = optixLaunchParams.uvsInside[uvInsideOffset + uvIndex];    // Deze moeten als CUDABuffer worden ingeladen!
        const int u = int(uv.x * optixLaunchParams.uvWorldPositions.size);
        const int v = int(uv.y * optixLaunchParams.uvWorldPositions.size);

        glm::vec3 UVWorldPos = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.uvWorldPositions.size + u].worldPosition;
        const glm::vec3 UVNormal = optixLaunchParams.uvWorldPositions.UVDataBuffer[v * optixLaunchParams.uvWorldPositions.size + u].worldNormal;

        float3 uvNormal3f = float3{ UVNormal.x, UVNormal.y, UVNormal.z };

        // Center of this radiance cell
        glm::vec3 cellCenter = optixLaunchParams.cellCenter;
        float cellSize = optixLaunchParams.cellSize;
        float stratifyCellWidth = cellSize / optixLaunchParams.stratifyResX;
        float stratifyCellHeight = cellSize / optixLaunchParams.stratifyResY;

        float stratifyCellWidthNormalized = 1.0f / optixLaunchParams.stratifyResX;
        float stratifyCellHeightNormalized = 1.0f / optixLaunchParams.stratifyResY;

        // SH weights for this cell
        float SHweights[8][9];
        int amountBasisFunctions = optixLaunchParams.sphericalHarmonicsWeights.amountBasisFunctions;
        int cellOffset = optixLaunchParams.nonEmptyCellIndex * amountBasisFunctions * 8;


        for (int sh_i = 0; sh_i < 8; sh_i++)
        {
            for (int basis_f_i = 0; basis_f_i < 9; basis_f_i++)
            {
                SHweights[sh_i][basis_f_i] = optixLaunchParams.sphericalHarmonicsWeights.weights[cellOffset + sh_i * amountBasisFunctions + basis_f_i];
            }
        }

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

        // Irradiance accumulator
        float totalIrradiance = 0.0f;

        // Loop over cell faces
        for (int face = 0; face < 6; face++)
        {
            // Which SHs of the cell belong to this face
            int4 faceSHIndices = cellSHIndices[face];

            glm::vec3 og = glm::vec3{ faceOgDuDv[face][0].x,faceOgDuDv[face][0].y,faceOgDuDv[face][0].z };
            glm::vec3 du = glm::vec3{ faceOgDuDv[face][1].x,faceOgDuDv[face][1].y,faceOgDuDv[face][1].z };
            glm::vec3 dv = glm::vec3{ faceOgDuDv[face][2].x,faceOgDuDv[face][2].y,faceOgDuDv[face][2].z };

            // Face normal and UV normal need to point in the same direction (hemisphere) for the UV to get contribution from that face
            double cellFaceFacing = dot(uvNormal3f, cellNormals[face]);
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

                            // Convert to float3 format
                            float3 rayOrigin3f = float3{ rayOrigin.x, rayOrigin.y, rayOrigin.z };
                            float3 rayDir3f = float3{ rayDir.x, rayDir.y, rayDir.z };

                            // Calculate spherical coordinate representation of ray
                            // (https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates)
                            float3 normalizedRayDir = normalize(rayDir3f);
                            double theta = acos(normalizedRayDir.z);
                            int signY = signbit(normalizedRayDir.y) == 0 ? 1 : -1;
                            double phi = signY * acos(normalizedRayDir.x / (sqrtf((normalizedRayDir.x * normalizedRayDir.x) + (normalizedRayDir.y * normalizedRayDir.y))));

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

                            prd.distanceToClosestIntersectionSquared = u0;
                            float distanceToUV = (((UVWorldPos.x - prd.rayOrigin.x) * (UVWorldPos.x - prd.rayOrigin.x)) + ((UVWorldPos.y - prd.rayOrigin.y) * (UVWorldPos.y - prd.rayOrigin.y)) + ((UVWorldPos.z - prd.rayOrigin.z) * (UVWorldPos.z - prd.rayOrigin.z)));

                            if (distanceToUV < prd.distanceToClosestIntersectionSquared)
                            {
                                // We calculate the dx and dy offsets to the (x,y) coordinate of the sampled point on a normalized square to use in 
                                // the calculation of the weights for bilinear interpolation
                                float dx = (stratifyIndexX * stratifyCellWidthNormalized + randomOffset.x * stratifyCellWidthNormalized) - 0.5;
                                float dy = (stratifyIndexY * stratifyCellHeightNormalized + randomOffset.y * stratifyCellHeightNormalized) - 0.5;

                                // Calculate bilinear interpolation weights, see thesis for explanation
                                float weightA = (0.5f + dx) * (1.0f - (0.5f + dy));
                                float weightB = (1.0f - (0.5f + dx)) * (1.0f - (0.5f + dy));
                                float weightC = (0.5f + dx) * (0.5f + dy);
                                float weightD = (1.0f - (0.5f + dx)) * (0.5f + dy);

                                // Basis function evaluations
                                float b0 = Y_0_0();
                                float b1 = Y_min1_1(phi, theta);
                                float b2 = Y_0_1(phi, theta);
                                float b3 = Y_1_1(phi, theta);
                                float b4 = Y_min2_2(phi, theta);
                                float b5 = Y_min1_2(phi, theta);
                                float b6 = Y_0_2(phi, theta);
                                float b7 = Y_1_2(phi, theta);
                                float b8 = Y_2_2(phi, theta);

                                // Calculate the outcoming weight to apply to each basis function
                                float w0 = SHweights[faceSHIndices.x][0] * weightC + SHweights[faceSHIndices.y][0] * weightD + SHweights[faceSHIndices.z][0] * weightA + SHweights[faceSHIndices.w][0] * weightB;
                                float w1 = SHweights[faceSHIndices.x][1] * weightC + SHweights[faceSHIndices.y][1] * weightD + SHweights[faceSHIndices.z][1] * weightA + SHweights[faceSHIndices.w][1] * weightB;
                                float w2 = SHweights[faceSHIndices.x][2] * weightC + SHweights[faceSHIndices.y][2] * weightD + SHweights[faceSHIndices.z][2] * weightA + SHweights[faceSHIndices.w][2] * weightB;
                                float w3 = SHweights[faceSHIndices.x][3] * weightC + SHweights[faceSHIndices.y][3] * weightD + SHweights[faceSHIndices.z][3] * weightA + SHweights[faceSHIndices.w][3] * weightB;
                                float w4 = SHweights[faceSHIndices.x][4] * weightC + SHweights[faceSHIndices.y][4] * weightD + SHweights[faceSHIndices.z][4] * weightA + SHweights[faceSHIndices.w][4] * weightB;
                                float w5 = SHweights[faceSHIndices.x][5] * weightC + SHweights[faceSHIndices.y][5] * weightD + SHweights[faceSHIndices.z][5] * weightA + SHweights[faceSHIndices.w][5] * weightB;
                                float w6 = SHweights[faceSHIndices.x][6] * weightC + SHweights[faceSHIndices.y][6] * weightD + SHweights[faceSHIndices.z][6] * weightA + SHweights[faceSHIndices.w][6] * weightB;
                                float w7 = SHweights[faceSHIndices.x][7] * weightC + SHweights[faceSHIndices.y][7] * weightD + SHweights[faceSHIndices.z][7] * weightA + SHweights[faceSHIndices.w][7] * weightB;
                                float w8 = SHweights[faceSHIndices.x][8] * weightC + SHweights[faceSHIndices.y][8] * weightD + SHweights[faceSHIndices.z][8] * weightA + SHweights[faceSHIndices.w][8] * weightB;

                                float irradiance = (b0 * w0) + (b1 * w1) + (b2 * w2) + (b3 * w3) + (b4 * w4) + (b5 * w5) + (b6 * w6) + (b7 * w7) + (b8 * w8);

                                totalIrradiance += irradiance;
                            }
                        }
                    }
                }
            }
        }

        int numSamples = 6 * NUM_SAMPLES_PER_STRATIFY_CELL * optixLaunchParams.stratifyResX * optixLaunchParams.stratifyResY;
        //printf("Total irradiance: %f\n", totalIrradiance);

        const int r = int(255.99 * totalIrradiance);
        const int g = int(255.99 * totalIrradiance);
        const int b = int(255.99 * totalIrradiance);

        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        const uint32_t rgba = 0xff000000
        | (r << 0) | (g << 8) | (b << 16);

        optixLaunchParams.currentBounceTexture.colorBuffer[v * optixLaunchParams.uvWorldPositions.size + u] = rgba;
    }
}