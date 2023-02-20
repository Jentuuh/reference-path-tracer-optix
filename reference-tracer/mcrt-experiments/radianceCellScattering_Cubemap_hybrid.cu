#pragma once

#include <optix_device.h>
#include "random.hpp"
#include "vec_math.hpp"

#include "LaunchParams.hpp"
#include "glm/glm.hpp"

#include "cube_mapping.cuh"
#include "utils.cuh"

#define PI 3.14159265358979323846f
#define NUM_SAMPLES_HEMISPHERE 400
#define TRACING_RANGE 0.5f

using namespace mcrt;


namespace mcrt {
    extern "C" __constant__ LaunchParamsRadianceCellScatterCubeMap optixLaunchParams;

    static __forceinline__ __device__ RadianceCellScatterPRDHybrid loadRadianceCellScatterPRD()
    {
        RadianceCellScatterPRDHybrid prd = {};

        prd.hitFound = optixGetPayload_0();
        prd.resultColor.x = __uint_as_float(optixGetPayload_1());
        prd.resultColor.y = __uint_as_float(optixGetPayload_2());
        prd.resultColor.z = __uint_as_float(optixGetPayload_3());

        return prd;
    }

    static __forceinline__ __device__ void storeRadianceCellScatterPRD(RadianceCellScatterPRDHybrid prd)
    {
        optixSetPayload_0(prd.hitFound);
        optixSetPayload_1(__float_as_uint(prd.resultColor.x));
        optixSetPayload_2(__float_as_uint(prd.resultColor.y));
        optixSetPayload_3(__float_as_uint(prd.resultColor.z));
    }


    extern "C" __global__ void __closesthit__radiance__cell__scattering__scene()
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

        RadianceCellScatterPRDHybrid prd = loadRadianceCellScatterPRD();
        prd.hitFound = 1;
        prd.resultColor = glm::vec3{ r,g,b };
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

        glm::vec3 cubeMin = { 0.0f, 0.0f ,0.0f };
        glm::vec3 cubeMax = { 1.0f, 1.0f ,1.0f };

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

        // Convert to float3 format
        float3 rayOrigin3f = float3{ UVWorldPos.x, UVWorldPos.y, UVWorldPos.z };
        
        // Probes coordinates
        glm::vec3 probeCoords[7] = {
            optixLaunchParams.cellCenter,
            optixLaunchParams.cellCenter + glm::vec3{optixLaunchParams.cellSize, 0.0f, 0.0f},
            optixLaunchParams.cellCenter - glm::vec3{optixLaunchParams.cellSize, 0.0f, 0.0f},
            optixLaunchParams.cellCenter + glm::vec3{0.0f, optixLaunchParams.cellSize, 0.0f},
            optixLaunchParams.cellCenter - glm::vec3{0.0f, optixLaunchParams.cellSize, 0.0f},
            optixLaunchParams.cellCenter + glm::vec3{0.0f, 0.0f, optixLaunchParams.cellSize},
            optixLaunchParams.cellCenter - glm::vec3{0.0f, 0.0f, optixLaunchParams.cellSize},
        };

        // Probe buffer offsets
        int probeOffsets[7] = { ((cellCoords.z * probeResWidth * probeResHeight) + (cellCoords.y * probeResWidth) + cellCoords.x) * 6 * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution), 
                                ((cellCoords.z * probeResWidth * probeResHeight) + (cellCoords.y * probeResWidth) + cellCoords.x + 1) * 6 * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution),
                                ((cellCoords.z * probeResWidth * probeResHeight) + (cellCoords.y * probeResWidth) + cellCoords.x - 1) * 6 * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution),
                                ((cellCoords.z * probeResWidth * probeResHeight) + ((cellCoords.y + 1) * probeResWidth) + cellCoords.x) * 6 * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution),
                                ((cellCoords.z * probeResWidth * probeResHeight) + ((cellCoords.y - 1) * probeResWidth) + cellCoords.x) * 6 * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution),
                                (((cellCoords.z + 1) * probeResWidth * probeResHeight) + (cellCoords.y * probeResWidth) + cellCoords.x) * 6 * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution),
                                (((cellCoords.z - 1) * probeResWidth * probeResHeight) + (cellCoords.y * probeResWidth) + cellCoords.x) * 6 * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution) };

        // Radiance accumulator
        glm::vec3 totalRadiance = glm::vec3{ 0.0f, 0.0f, 0.0f };

        // Number of samples accumulator
        int numSamples = 0;

        // =============================================
        // Take hemisphere samples of incoming radiance
        // =============================================
        for (int s = 0; s < NUM_SAMPLES_HEMISPHERE; s++)
        {
            // Generate random direction on sphere
            float2 uniformRandoms = float2{ rnd(seed), rnd(seed) };
            float theta = acos(1.0f - (2.0f * uniformRandoms.x));
            float phi = 2 * PI * uniformRandoms.y;
            float3 randomDir = float3{ sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta) };
            
            // If the generated random direction is not in the oriented hemisphere, invert it
            if (dot(normalize(randomDir), normalize(uvNormal3f)) < 0)
            {
                randomDir = float3{-randomDir.x, -randomDir.y, -randomDir.z};
            }

            // Note that this is important! A ray has the following equation: `O + td`. If d is normalized, 
            // t represents the exact range or length along which we trace the ray. This assumption is necessary
            // when we want to set a maximum tracing range.
            randomDir = normalize(randomDir);

            // ===============================================
            // Test ray for intersections within tracing range
            // ===============================================
            RadianceCellScatterPRDHybrid prd{};
            prd.hitFound = 0;

            unsigned int u0, u1, u2, u3;
            u0 = prd.hitFound;

            // Trace ray. In case we find an intersection within the tracing range, we let the indirect light source contribute.
            // In case the ray hits nothing within the tracing range, we approximate "distant lighting" using the most nearby light probe.
            optixTrace(optixLaunchParams.sceneTraversable,
                rayOrigin3f,
                randomDir,
                0.f,            // tmin
                TRACING_RANGE,  // tmax (tracing range)
                0.0f,           // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,     
                0,  // SBT offset
                1,  // SBT stride
                0,  // missSBTIndex
                u0, u1, u2, u3
            );

            prd.resultColor = glm::vec3{ __uint_as_float(u1), __uint_as_float(u2), __uint_as_float(u3) };

            if(u0 == 1)  // TRACED CONTRIBUTION
            {
                // Cosine weighted contribution
                float cosContribution = dot(normalize(randomDir), normalize(uvNormal3f));
                totalRadiance += glm::vec3{ cosContribution * prd.resultColor.x, cosContribution * prd.resultColor.y, cosContribution * prd.resultColor.z };
                ++numSamples;
            }
            else {  // PROBE CONTRIBUTION (How to make the distinction between an actual miss and out of range?)
                // Find "distant projection" along ray direction of point that we are calculating incoming radiance for, 
                // this is necessary to sample an approximated correct direction on the radiance probes.
                double t_min;
                double t_max;
                find_distant_point_along_direction(UVWorldPos, glm::vec3{ randomDir.x, randomDir.y, randomDir.z }, cubeMin, cubeMax, &t_min, &t_max);
                glm::vec3 distantProjectedPoint = UVWorldPos + (glm::vec3{ randomDir.x * t_max,  randomDir.y * t_max,  randomDir.z * t_max });
                //printf("UVWorldPos:%f %f %f , projected: %f %f %f, dir: %f %f %f \n", UVWorldPos.x, UVWorldPos.y, UVWorldPos.z, distantProjectedPoint.x, distantProjectedPoint.y, distantProjectedPoint.z, randomDir.x, randomDir.y, randomDir.z);

                float faceU, faceV;
                int cubeMapFaceIndex;

                // ==================================================================================
                // Sample the probe in the center of this cell
                // ==================================================================================
                glm::vec3 averageProbeContribution = {0.0f, 0.0f, 0.0f};
                int numProbeContributions = 0;
                for (int p = 0; p < 1; p++)
                {
                    if (probeCoords[p].x >= 0.0f && probeCoords[p].x <= 1.0f && probeCoords[p].y >= 0.0f && probeCoords[p].y <= 1.0f && probeCoords[p].z >= 0.0f && probeCoords[p].z <= 1.0f)
                    {
                        glm::vec3 probeSampleDirection = distantProjectedPoint - probeCoords[p];
                        convert_xyz_to_cube_uv(probeSampleDirection.x, probeSampleDirection.y, probeSampleDirection.z, &cubeMapFaceIndex, &faceU, &faceV);

                        int uIndex = optixLaunchParams.cubeMapResolution * faceU;
                        int vIndex = optixLaunchParams.cubeMapResolution * faceV;
                        int uvOffset = vIndex * optixLaunchParams.cubeMapResolution + uIndex;

                        float r = optixLaunchParams.cubeMaps[((probeOffsets[p] + cubeMapFaceIndex * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution)) * 3) + uvOffset * 3 + 0];
                        float g = optixLaunchParams.cubeMaps[((probeOffsets[p] + cubeMapFaceIndex * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution)) * 3) + uvOffset * 3 + 1];
                        float b = optixLaunchParams.cubeMaps[((probeOffsets[p] + cubeMapFaceIndex * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution)) * 3) + uvOffset * 3 + 2];
                        
                        averageProbeContribution += glm::vec3{ r, g, b };
                        numProbeContributions++;
                    }
                }
                averageProbeContribution = glm::vec3{ averageProbeContribution.x / float(numProbeContributions), averageProbeContribution.y / float(numProbeContributions), averageProbeContribution.z / float(numProbeContributions) };

                // Cosine weighted contribution
                float cosContribution = dot(normalize(randomDir), normalize(uvNormal3f));
                if (cosContribution >= 0)
                {
                    totalRadiance += cosContribution * averageProbeContribution;
                    numSamples++;
                }
            }
        }

        const float r = totalRadiance.x / (float(numSamples) * PI);
        const float g = totalRadiance.y / (float(numSamples) * PI);
        const float b = totalRadiance.z / (float(numSamples) * PI);

        optixLaunchParams.currentBounceTexture.colorBuffer[(v * optixLaunchParams.currentBounceTexture.size * 3) + (u * 3) + 0] = r;
        optixLaunchParams.currentBounceTexture.colorBuffer[(v * optixLaunchParams.currentBounceTexture.size * 3) + (u * 3) + 1] = g;
        optixLaunchParams.currentBounceTexture.colorBuffer[(v * optixLaunchParams.currentBounceTexture.size * 3) + (u * 3) + 2] = b;
    }
}