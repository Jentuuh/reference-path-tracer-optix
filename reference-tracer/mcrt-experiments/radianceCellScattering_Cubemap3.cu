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
#define NUM_SAMPLES_PER_STRATIFY_CELL 5
#define STRATIFY_RES_X 5
#define STRATIFY_RES_Y 5

using namespace mcrt;


namespace mcrt {
    extern "C" __constant__ LaunchParamsRadianceCellScatterCubeMap optixLaunchParams;

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
        const int nonEmptyCellIndex = optixLaunchParams.nonEmptyCellIndex;
        const glm::ivec3 cellCoords = optixLaunchParams.cellCoords;
        const int probeResWidth = optixLaunchParams.probeWidthRes;
        const int probeResHeight = optixLaunchParams.probeHeightRes;

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

        float3 uvNormal3f = float3{ UVNormal.x, UVNormal.y, UVNormal.z };

        // We apply a small offset of 0.00001f in the direction of the normal to the UV world pos, to 'mitigate' floating point rounding errors causing false occlusions/illuminations
        UVWorldPos = glm::vec3{ UVWorldPos.x + UVNormal.x * 0.0001f, UVWorldPos.y + UVNormal.y * 0.0001f, UVWorldPos.z + UVNormal.z * 0.0001f };

        // Center of this radiance cell
        glm::vec3 cellCenter = optixLaunchParams.cellCenter;
        float cellSize = optixLaunchParams.cellSize;
        glm::vec3 cellOrigin = { cellCenter.x - 0.5f * cellSize,cellCenter.y - 0.5f * cellSize,cellCenter.z - 0.5f * cellSize };    // Origin corner point of the cell

        float stratifyCellWidth = cellSize / STRATIFY_RES_X;
        float stratifyCellHeight = cellSize / STRATIFY_RES_Y;

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
        // The indices of the probes that belong to each face, to use while indexing the buffer (L,R,U,D,F,B), (LB, RB, LT, RT)
        int cellProbeIndices[6][4] = { {4, 0, 6, 2}, {1, 5, 3, 7}, {2, 3, 6, 7}, {4, 5, 0, 1}, {0, 1, 2, 3}, {5, 4, 7, 6} };

        int probeOffset = ((cellCoords.z * probeResWidth * probeResHeight) + (cellCoords.y * probeResWidth) + cellCoords.x) * 6 * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution);


        // Radiance accumulator
        glm::vec3 totalRadiance = glm::vec3{ 0.0f, 0.0f, 0.0f };
        // Number of samples accumulator
        int numSamples = 0;

        // Loop over cell faces
        for (int face = 0; face < 6; face++)
        {
            // Which probes of the cell belong to this face
            int* faceProbeIndices = cellProbeIndices[face];

            glm::vec3 og = glm::vec3{ faceOgDuDv[face][0].x,faceOgDuDv[face][0].y,faceOgDuDv[face][0].z };
            glm::vec3 du = glm::vec3{ faceOgDuDv[face][1].x,faceOgDuDv[face][1].y,faceOgDuDv[face][1].z };
            glm::vec3 dv = glm::vec3{ faceOgDuDv[face][2].x,faceOgDuDv[face][2].y,faceOgDuDv[face][2].z };


            // For each stratified cell on the face, take samples
            for (int stratifyIndexX = 0; stratifyIndexX < STRATIFY_RES_X; stratifyIndexX++)
            {
                for (int stratifyIndexY = 0; stratifyIndexY < STRATIFY_RES_Y; stratifyIndexY++)
                {
                    glm::vec3 stratifyCellOrigin = og + (stratifyIndexX * stratifyCellWidth * du) + (stratifyIndexY * stratifyCellHeight * dv);

                    // Send out a ray for each sample
                    for (int sample = 0; sample < NUM_SAMPLES_PER_STRATIFY_CELL; sample++)
                    {
                        // Take a random sample on the face's stratified cell, this will be the ray origin
                        float2 randomOffset = float2{ rnd(seed), rnd(seed) };
                        glm::vec3 rayDest = stratifyCellOrigin + (randomOffset.x * stratifyCellWidth * du) + (randomOffset.y * stratifyCellHeight * dv);

                        // Ray direction (from the sample to the UV texel)
                        glm::vec3 rayDir = rayDest - UVWorldPos;
                        glm::vec3 rayDirInv = UVWorldPos - rayDest;

                        // Convert to float3 format
                        float3 rayOrigin3f = float3{ UVWorldPos.x, UVWorldPos.y, UVWorldPos.z };
                        float3 rayDir3f = float3{ rayDir.x, rayDir.y, rayDir.z };
                        float3 rayDir3fInv = float3{ rayDirInv.x, rayDirInv.y, rayDirInv.z };

                        RadianceCellScatterPRD prd{};
                        prd.rayOrigin = UVWorldPos;

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
                        float distanceToUVSquared = (((UVWorldPos.x - rayDest.x) * (UVWorldPos.x - rayDest.x)) + ((UVWorldPos.y - rayDest.y) * (UVWorldPos.y - rayDest.y)) + ((UVWorldPos.z - rayDest.z) * (UVWorldPos.z - rayDest.z)));

                        if (distanceToUVSquared < prd.distanceToClosestIntersectionSquared)
                        {
                            // Find "distant projection" along ray direction of point that we are calculating incoming radiance for, 
                            // this is necessary to sample an approximated correct direction on the radiance probes.
                            glm::vec3 distantProjectedPoint;
                            find_distant_point_along_direction(UVWorldPos, rayDir, &distantProjectedPoint);
                            //printf("UVWorldPos:%f %f %f , projected: %f %f %f, dir: %f %f %f \n", UVWorldPos.x, UVWorldPos.y, UVWorldPos.z, distantProjectedPoint.x, distantProjectedPoint.y, distantProjectedPoint.z, rayDir.x, rayDir.y, rayDir.z);

                            float faceU, faceV;
                            int cubeMapFaceIndex;

                            //// ==================================================================================
                            //// For each probe on the facing face, sample cube map to retrieve incoming radiance
                            //// ==================================================================================
                            //for (int p = 0; p < 4; p++)
                            //{
                            //    glm::vec3 probeSampleDirection = distantProjectedPoint - cellProbeOrigins[faceProbeIndices[p]];
                            //    convert_xyz_to_cube_uv(probeSampleDirection.x, probeSampleDirection.y, probeSampleDirection.z, &cubeMapFaceIndex, &faceU, &faceV);

                            //    int uIndex = optixLaunchParams.cubeMapResolution * faceU;
                            //    int vIndex = optixLaunchParams.cubeMapResolution * faceV;
                            //    int uvOffset = vIndex * optixLaunchParams.cubeMapResolution + uIndex;
                            //    int probeOffset = probeOffsets[faceProbeIndices[p]];

                            //    uint32_t incomingRadiance = optixLaunchParams.cubeMaps[(probeOffset + cubeMapFaceIndex * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution)) + uvOffset];

                            //    // Extract rgb values from light source texture pixel
                            //    uint32_t r = 0x000000ff & (incomingRadiance);
                            //    uint32_t g = (0x0000ff00 & (incomingRadiance)) >> 8;
                            //    uint32_t b = (0x00ff0000 & (incomingRadiance)) >> 16;
                            //    glm::vec3 rgbNormalizedSpectrum = glm::vec3{ r, g, b} / 255.0f;

                            //    // Convert to grayscale (for now we assume 1 color channel)
                            //    const float grayscale = (0.3 * r + 0.59 * g + 0.11 * b) / 255.0f;

                            //    // Cosine weighted contribution
                            //    totalRadiance += rgbNormalizedSpectrum * dot(uvNormal3f, rayDir3f);
                            //    numSamples++;
                            //}

                            // ==================================================================================
                            // Sample the probe in the center of this cell
                            // ==================================================================================
                            glm::vec3 probeSampleDirection = distantProjectedPoint - cellCenter;
                            convert_xyz_to_cube_uv(probeSampleDirection.x, probeSampleDirection.y, probeSampleDirection.z, &cubeMapFaceIndex, &faceU, &faceV);

                            int uIndex = optixLaunchParams.cubeMapResolution * faceU;
                            int vIndex = optixLaunchParams.cubeMapResolution * faceV;
                            int uvOffset = vIndex * optixLaunchParams.cubeMapResolution + uIndex;

                            uint32_t incomingRadiance = optixLaunchParams.cubeMaps[(probeOffset + cubeMapFaceIndex * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution)) + uvOffset];

                            // Extract rgb values from light source texture pixel
                            uint32_t r = 0x000000ff & (incomingRadiance);
                            uint32_t g = (0x0000ff00 & (incomingRadiance)) >> 8;
                            uint32_t b = (0x00ff0000 & (incomingRadiance)) >> 16;
                            glm::vec3 rgbNormalizedSpectrum = glm::vec3{ r, g, b } / 255.0f;

                            // Convert to grayscale (for now we assume 1 color channel)
                            const float intensity = (0.3 * r + 0.59 * g + 0.11 * b) / 255.0f;

                            // Cosine weighted contribution
                            float cosineWeight = dot(uvNormal3f, rayDir3f);
                            if (cosineWeight >= 0)
                            {
                                totalRadiance += rgbNormalizedSpectrum * cosineWeight;
                                numSamples++;
                            }


                            //// ============================================================================================
                            //// Sample most nearby probe only (Problem here is that many probes tend to give black results)
                            //// ============================================================================================
                            //glm::vec3 probeSampleDirection = distantProjectedPoint - cellProbeOrigins[minDistanceIndex];
                            //convert_xyz_to_cube_uv(probeSampleDirection.x, probeSampleDirection.y, probeSampleDirection.z, &cubeMapFaceIndex, &faceU, &faceV);
                            //int uIndex = optixLaunchParams.cubeMapResolution * faceU;
                            //int vIndex = optixLaunchParams.cubeMapResolution * faceV;
                            //int uvOffset = vIndex * optixLaunchParams.cubeMapResolution + uIndex;

                            //uint32_t incomingRadiance = optixLaunchParams.cubeMaps[(probeOffsets[minDistanceIndex] + cubeMapFaceIndex * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution)) + uvOffset];

                            //// Extract rgb values from light source texture pixel
                            //uint32_t r = 0x000000ff & (incomingRadiance);
                            //uint32_t g = (0x0000ff00 & (incomingRadiance)) >> 8;
                            //uint32_t b = (0x00ff0000 & (incomingRadiance)) >> 16;
                            //glm::vec3 rgbNormalizedSpectrum = glm::vec3{ r, g, b } / 255.0f;

                            //// Convert to grayscale (for now we assume 1 color channel)
                            ////const float grayscale = (0.3 * r + 0.59 * g + 0.11 * b) / 255.0f;

                            //// Cosine weighted contribution
                            //totalRadiance += rgbNormalizedSpectrum * dot(uvNormal3f, rayDir3f);
                            //numSamples++;
                        }
                    }
                }
            }
        }

        const int r = int(255.99 * (totalRadiance.x / float(numSamples)));
        const int g = int(255.99 * (totalRadiance.y / float(numSamples)));
        const int b = int(255.99 * (totalRadiance.z / float(numSamples)));

        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        const uint32_t rgba = 0xff000000
            | (r << 0) | (g << 8) | (b << 16);

        optixLaunchParams.currentBounceTexture.colorBuffer[v * optixLaunchParams.currentBounceTexture.size + u] = rgba;
    }
}