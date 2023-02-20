#pragma once

#include <optix_device.h>
#include "random.hpp"
#include "vec_math.hpp"

#include "LaunchParams.hpp"
#include "glm/glm.hpp"

#include "spherical_harmonics.cuh"

#define PI 3.14159265358979323846f
#define EPSILON 0.0000000000002f
#define NUM_SAMPLES_PER_STRATIFY_CELL 10

using namespace mcrt;

namespace mcrt {

    extern "C" __constant__ LaunchParamsRadianceCellGather optixLaunchParams;

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

    static __forceinline__ __device__ RadianceCellGatherPRD loadRadianceCellGatherPRD()
    {
        RadianceCellGatherPRD prd = {};

        prd.distanceToClosestProxyIntersectionSquared = __uint_as_float(optixGetPayload_0());
        prd.rayOrigin.x = __uint_as_float(optixGetPayload_1());
        prd.rayOrigin.y = __uint_as_float(optixGetPayload_2());
        prd.rayOrigin.z = __uint_as_float(optixGetPayload_3());

        return prd;
    }

    static __forceinline__ __device__ void storeRadianceCellGatherPRD(RadianceCellGatherPRD prd)
    {
        optixSetPayload_0(__float_as_uint(prd.distanceToClosestProxyIntersectionSquared));
        optixSetPayload_1(__float_as_uint(prd.rayOrigin.x));
        optixSetPayload_2(__float_as_uint(prd.rayOrigin.y));
        optixSetPayload_3(__float_as_uint(prd.rayOrigin.z));
    }


    extern "C" __global__ void __closesthit__radiance__cell__gathering__scene()
    {
        const MeshSBTDataRadianceCellGather& sbtData
            = *(const MeshSBTDataRadianceCellGather*)optixGetSbtDataPointer();

        const int   primID = optixGetPrimitiveIndex();
        const glm::ivec3 index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        const glm::vec3 intersectionWorldPos =
            (1.f - u - v) * sbtData.vertex[index.x]
            + u * sbtData.vertex[index.y]
            + v * sbtData.vertex[index.z];

        RadianceCellGatherPRD prd = loadRadianceCellGatherPRD();
        float distanceToProxyIntersect = (((intersectionWorldPos.x - prd.rayOrigin.x) * (intersectionWorldPos.x - prd.rayOrigin.x)) + ((intersectionWorldPos.y - prd.rayOrigin.y) * (intersectionWorldPos.y - prd.rayOrigin.y)) + ((intersectionWorldPos.z - prd.rayOrigin.z) * (intersectionWorldPos.z - prd.rayOrigin.z)));

        prd.distanceToClosestProxyIntersectionSquared = distanceToProxyIntersect;
        storeRadianceCellGatherPRD(prd);
    }

    extern "C" __global__ void __anyhit__radiance__cell__gathering__scene() {
        // Do nothing
        printf("Hit scene!");
    }

    extern "C" __global__ void __miss__radiance__cell__gathering()
    {
    }

    extern "C" __global__ void __raygen__renderFrame__cell__gathering()
    {
        // Get thread indices
        const int uIndex = optixGetLaunchIndex().x;
        const int vIndex = optixGetLaunchIndex().y;

        // Amount SH basis functions
        int amountBasisFunctions = optixLaunchParams.sphericalHarmonicsWeights.amountBasisFunctions;

        // Size of a radiance cell + dimensions of each stratified cell
        const float cellSize = optixLaunchParams.cellSize;
        float stratifyCellWidth = cellSize / optixLaunchParams.stratifyResX;
        float stratifyCellHeight = cellSize / optixLaunchParams.stratifyResY;

        float stratifyCellWidthNormalized = 1.0 / optixLaunchParams.stratifyResX;
        float stratifyCellHeightNormalized = 1.0 / optixLaunchParams.stratifyResY;

        // TODO: SKIP PIXELS THAT ARE BLACK!
        uint32_t lightSrcColor = optixLaunchParams.lightSourceTexture.colorBuffer[vIndex * optixLaunchParams.lightSourceTexture.size + uIndex];
        //printf("%d", lightSrcColor);

        glm::vec3 UVWorldPos = optixLaunchParams.uvWorldPositions.UVDataBuffer[vIndex * optixLaunchParams.lightSourceTexture.size + uIndex].worldPosition;
        const glm::vec3 UVNormal = optixLaunchParams.uvWorldPositions.UVDataBuffer[vIndex * optixLaunchParams.lightSourceTexture.size + uIndex].worldNormal;
        float3 uvNormal3f = float3{ UVNormal.x, UVNormal.y, UVNormal.z };

        // We apply a small offset of 0.00001f in the direction of the normal to the UV world pos, to 'mitigate' floating point rounding errors causing false occlusions/illuminations
        UVWorldPos = glm::vec3{ UVWorldPos.x + UVNormal.x * 0.00001f, UVWorldPos.y + UVNormal.y * 0.00001f, UVWorldPos.z + UVNormal.z * 0.00001f };
        
        // Iterate over all non-empty cells
        for (int i = 0; i < optixLaunchParams.nonEmptyCells.size; i++)
        {
            // Take different seed for each radiance cell
            unsigned int seed = tea<4>(vIndex * optixLaunchParams.lightSourceTexture.size + uIndex, i);

            glm::vec3 cellCenter = optixLaunchParams.nonEmptyCells.centers[i];
            glm::vec3 lightToCellDir = { cellCenter.x - UVWorldPos.x, cellCenter.y - UVWorldPos.y, cellCenter.z - UVWorldPos.z };

            float3 rayOrigin3f = float3{ UVWorldPos.x, UVWorldPos.y, UVWorldPos.z };
            float3 rayOgToCellCenter3f = float3{ lightToCellDir.x, lightToCellDir.y, lightToCellDir.z };
            
            // Cosine between vector from ray origin to cell center and texel normal to check if cell is facing
            double radCellFacing = dot(normalize(rayOgToCellCenter3f), uvNormal3f);

            if (radCellFacing > 0)
            {
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
                // The indices of the SHs that belong to each face, to use while indexing the buffer (L,R,U,D,F,B)
                int4 cellSHIndices[6] = { int4{4, 0, 6, 2}, int4{1, 5, 3, 7}, int4{2, 3, 6, 7}, int4{4, 5, 0, 1}, int4{0, 1, 2, 3}, int4{5, 4, 7, 6} };

            
                for (int face = 0; face < 6; face++)
                {
                    // We accumulate the projections of each ray into this buffer, the projection of the lighting function
                    // onto the SH basis functions is estimated by Monte Carlo integration. 
                    double contribution[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                    // Expected value buffer for weightA, weightB, weightC, weightD for bilinear interpolation in the face square/
                    // The expected values of these weights will be used to decide the final contribution to the SH of each corner after projection.
                    double bilinInterpolWeightsExpectedValues[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    // Rays that pass all tests and thus contribute
                    int n_samples = 0;

                    // Check if the current cell face is facing, otherwise skip.
                    double cellFaceFacing = dot(uvNormal3f, cellNormals[face]);
                    if (cellFaceFacing < 0)
                    {
                        // For each stratified cell on the face, take samples
                        for (int stratifyIndexX = 0; stratifyIndexX < optixLaunchParams.stratifyResX; stratifyIndexX++)
                        {
                            for (int stratifyIndexY = 0; stratifyIndexY < optixLaunchParams.stratifyResY; stratifyIndexY++)
                            {
                                glm::vec3 og = glm::vec3{ faceOgDuDv[face][0].x,faceOgDuDv[face][0].y,faceOgDuDv[face][0].z };
                                glm::vec3 du = glm::vec3{ faceOgDuDv[face][1].x,faceOgDuDv[face][1].y,faceOgDuDv[face][1].z };
                                glm::vec3 dv = glm::vec3{ faceOgDuDv[face][2].x,faceOgDuDv[face][2].y,faceOgDuDv[face][2].z };

                                glm::vec3 stratifyCellOrigin = og + (stratifyIndexX * stratifyCellWidth * du) + (stratifyIndexY * stratifyCellHeight * dv);

                                // Send out a ray for each sample
                                for (int sample = 0; sample < NUM_SAMPLES_PER_STRATIFY_CELL; sample++)
                                {
                                    // Take a random sample on the face's stratified cell
                                    float2 randomOffset = float2{ rnd(seed), rnd(seed) };        
                                    glm::vec3 rayDestination = stratifyCellOrigin + (randomOffset.x * stratifyCellWidth * du) + (randomOffset.y * stratifyCellHeight * dv);

                                    // Ray direction
                                    glm::vec3 rayDir = rayDestination - UVWorldPos;

                                    // Convert to float3 format
                                    float3 rayOrigin3f = float3{ UVWorldPos.x, UVWorldPos.y, UVWorldPos.z };
                                    float3 rayDir3f = float3{ rayDir.x, rayDir.y, rayDir.z };

                                    // Calculate spherical coordinate representation of ray
                                    // (https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates)
                                    float3 normalizedRayDir = normalize(rayDir3f);
                                    double theta = acos(normalizedRayDir.z);
                                    int signY = signbit(normalizedRayDir.y) == 0 ? 1 : -1;
                                    double phi = signY * acos(normalizedRayDir.x / (sqrtf((normalizedRayDir.x * normalizedRayDir.x) + (normalizedRayDir.y * normalizedRayDir.y))));

                                    printf("%f %f", theta, phi);

                                    RadianceCellGatherPRD prd{};
                                    prd.rayOrigin = UVWorldPos;

                                    unsigned int u0, u1, u2, u3;

                                    u1 = __float_as_uint(prd.rayOrigin.x);
                                    u2 = __float_as_uint(prd.rayOrigin.y);
                                    u3 = __float_as_uint(prd.rayOrigin.z);

                                    // Call against scene geometry
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

                                    prd.distanceToClosestProxyIntersectionSquared = u0;
                                    float distanceToGridIntersect = (((rayDestination.x - prd.rayOrigin.x) * (rayDestination.x - prd.rayOrigin.x)) + ((rayDestination.y - prd.rayOrigin.y) * (rayDestination.y - prd.rayOrigin.y)) + ((rayDestination.z - prd.rayOrigin.z) * (rayDestination.z - prd.rayOrigin.z)));

                                    if (distanceToGridIntersect < prd.distanceToClosestProxyIntersectionSquared)
                                    {
                                        ++n_samples;

                                        // We calculate the dx and dy offsets to the (x,y) coordinate of the sampled point on a normalized square to use in 
                                        // the calculation of the weights for bilinear extrapolation
                                        float dx = (stratifyIndexX * stratifyCellWidthNormalized + randomOffset.x * stratifyCellWidthNormalized) - 0.5;
                                        float dy = (stratifyIndexY * stratifyCellHeightNormalized + randomOffset.y * stratifyCellHeightNormalized) - 0.5;
                       
                                        // Accumulate bilinear interpolation weights, see thesis for explanation
                                        bilinInterpolWeightsExpectedValues[0] += (0.5f + dx) * (1.0f - (0.5f + dy));
                                        bilinInterpolWeightsExpectedValues[1] += (1.0f - (0.5f + dx)) * (1.0f - (0.5f + dy));
                                        bilinInterpolWeightsExpectedValues[2] += (0.5f + dx) * (0.5f + dy);
                                        bilinInterpolWeightsExpectedValues[3] += (1.0f - (0.5f + dx)) * (0.5f + dy);

                                        // Project lighting function (single ray accumulation) onto SH basis functions
                                        contribution[0] += lightSrcColor * Y_0_0();
                                        contribution[1] += lightSrcColor * Y_min1_1(phi, theta);
                                        contribution[2] += lightSrcColor * Y_0_1(phi, theta);
                                        contribution[3] += lightSrcColor * Y_1_1(phi, theta);
                                        contribution[4] += lightSrcColor * Y_min2_2(phi, theta);
                                        contribution[5] += lightSrcColor * Y_min1_2(phi, theta);
                                        contribution[6] += lightSrcColor * Y_0_2(phi, theta);
                                        contribution[7] += lightSrcColor * Y_1_2(phi, theta);
                                        contribution[8] += lightSrcColor * Y_2_2(phi, theta);
                                    }
                                }
                            }
                        }
                    }

                    if (n_samples > 0)
                    {
                        // Divide by amount of rays that contributed (samples) to get expected weights value
                        double weightsFactor = 1.0 / n_samples;
                        for (int w = 0; w < 4; w++)
                        {
                            bilinInterpolWeightsExpectedValues[w] = bilinInterpolWeightsExpectedValues[w] * weightsFactor;
                        }

                        // Idem for SH basis coefficients, part of the Monte Carlo integration (see paper SH Lighting: 'The gritty details')
                        double weight = 1.0; // TODO: Unsure what this weight needs to be, in the paper they use 4pi because they are uniformly sampling the sphere, but that is not the case here...
                        double contributionFactor = weight / n_samples;
                        for (int w = 0; w < 9; w++)
                        {
                            contribution[w] = contribution[w] * contributionFactor;
                        }

                        // Current non-empty cell * amount of basis functions * 8 SHs per cell 
                        int cellOffset = i * amountBasisFunctions * 8;

                        double weightA = 1.0 / bilinInterpolWeightsExpectedValues[0];
                        double weightB = 1.0 / bilinInterpolWeightsExpectedValues[1];
                        double weightC = 1.0 / bilinInterpolWeightsExpectedValues[2];
                        double weightD = 1.0 / bilinInterpolWeightsExpectedValues[3];

                        for (int w = 0; w < 9; w++)
                        {                                                                                                   // Am i allowed to just add this here, won't this blow up the coefficients?
                            optixLaunchParams.sphericalHarmonicsWeights.weights[cellOffset + cellSHIndices[face].x * amountBasisFunctions + w] += contribution[w] * weightC;
                            optixLaunchParams.sphericalHarmonicsWeights.weights[cellOffset + cellSHIndices[face].y * amountBasisFunctions + w] += contribution[w] * weightD;
                            optixLaunchParams.sphericalHarmonicsWeights.weights[cellOffset + cellSHIndices[face].z * amountBasisFunctions + w] += contribution[w] * weightA;
                            optixLaunchParams.sphericalHarmonicsWeights.weights[cellOffset + cellSHIndices[face].w * amountBasisFunctions + w] += contribution[w] * weightB;
                        }
                    }
                }
            }
        }
    }
}