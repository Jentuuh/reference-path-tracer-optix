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

        const int probeResWidth = optixLaunchParams.probeWidthRes;
        const int probeResHeight = optixLaunchParams.probeHeightRes;

        glm::vec3 cubeMin = { 0.0f, 0.0f ,0.0f };
        glm::vec3 cubeMax = { 1.0f, 1.0f ,1.0f };

        glm::vec3 probePos1 = optixLaunchParams.cellCenter;
        glm::vec3 probePos2 = optixLaunchParams.cellCenter + glm::vec3{ 2 * optixLaunchParams.cellSize, 0.0f, 0.0f };

        printf("probe res width: %d probe res height: %d \n", probeResWidth, probeResHeight);
        int probeOffset1 = ((optixLaunchParams.cellCoords.z * probeResWidth * probeResHeight) + (optixLaunchParams.cellCoords.y * probeResWidth) + (optixLaunchParams.cellCoords.x)) * 6 * optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution;
        int probeOffset2 = ((optixLaunchParams.cellCoords.z * probeResWidth * probeResHeight) + (optixLaunchParams.cellCoords.y * probeResWidth) + (optixLaunchParams.cellCoords.x + 2)) * 6 * optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution;

        glm::vec3 UVWorldPos = optixLaunchParams.uvWorldPositions.UVDataBuffer[570 * optixLaunchParams.uvWorldPositions.size + 570].worldPosition;

        glm::vec3 rayDir1 = UVWorldPos - probePos1;
        glm::vec3 rayDir2 = UVWorldPos - probePos2;
        printf("Probe 1 Pos: %f %f %f Probe 1 dir: %f %f %f \n", probePos1.x, probePos1.y, probePos1.z, rayDir1.x, rayDir1.y, rayDir1.z);
        printf("Probe 2 Pos: %f %f %f Probe 2 dir: %f %f %f \n", probePos2.x, probePos2.y, probePos2.z, rayDir2.x, rayDir2.y, rayDir2.z);

        double t_min_1;
        double t_max_1;
        find_distant_point_along_direction(probePos1, rayDir1, cubeMin, cubeMax, &t_min_1, &t_max_1);
        glm::vec3 distantProjectedPoint1 = probePos1 + (glm::vec3{ rayDir1.x * t_max_1,  rayDir1.y * t_max_1,  rayDir1.z * t_max_1 });

        double t_min_2;
        double t_max_2;
        find_distant_point_along_direction(probePos2, rayDir2, cubeMin, cubeMax, &t_min_2, &t_max_2);
        glm::vec3 distantProjectedPoint2 = probePos2 + (glm::vec3{ rayDir2.x * t_max_2,  rayDir2.y * t_max_2,  rayDir2.z * t_max_2 });

        printf("Distant project point 1: %f %f %f \n", distantProjectedPoint1.x, distantProjectedPoint1.y, distantProjectedPoint1.z);
        printf("Distant project point 2: %f %f %f \n", distantProjectedPoint2.x, distantProjectedPoint2.y, distantProjectedPoint2.z);

        printf("UV world pos: %f %f %f \n", UVWorldPos.x, UVWorldPos.y, UVWorldPos.z);

        printf("cubeMapRes: %d \n", optixLaunchParams.cubeMapResolution);

        float faceU1, faceV1, faceU2, faceV2;
        int cubeMapFaceIndex1, cubeMapFaceIndex2;

        convert_xyz_to_cube_uv(rayDir1.x, rayDir1.y, rayDir1.z, &cubeMapFaceIndex1, &faceU1, &faceV1);
        printf("Probe 1: Face index: %f, faceU: %f, faceV: %f \n", cubeMapFaceIndex1, faceU1, faceV1);
        convert_xyz_to_cube_uv(rayDir2.x, rayDir2.y, rayDir2.z, &cubeMapFaceIndex2, &faceU2, &faceV2);
        printf("Probe 2: Face index: %f, faceU: %f, faceV: %f \n", cubeMapFaceIndex2, faceU2, faceV2);

        int uIndex1 = optixLaunchParams.cubeMapResolution * faceU1;
        int vIndex1 = optixLaunchParams.cubeMapResolution * faceV1;
        int uvOffset1 = vIndex1 * optixLaunchParams.cubeMapResolution + uIndex1;
        printf("Probe 1:UV (%d, %d) \n", uIndex1, vIndex1);

        int uIndex2 = optixLaunchParams.cubeMapResolution * faceU2;
        int vIndex2 = optixLaunchParams.cubeMapResolution * faceV2;
        int uvOffset2 = vIndex2 * optixLaunchParams.cubeMapResolution + uIndex2;
        printf("Probe 2:UV (%d, %d) \n", uIndex2, vIndex2);
        
        printf("Probe offset 1: %d\n", probeOffset1);
        printf("Accessed index 1: %d\n", (probeOffset1 + (cubeMapFaceIndex1 * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution))));
        uint32_t incomingRadiance1 = optixLaunchParams.cubeMaps[(probeOffset1 + (cubeMapFaceIndex1 * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution))) + uvOffset1];
        uint32_t r1 = 0x000000ff & (incomingRadiance1);
        uint32_t g1 = (0x0000ff00 & (incomingRadiance1)) >> 8;
        uint32_t b1 = (0x00ff0000 & (incomingRadiance1)) >> 16;

        printf("Probe offset 2: %d\n", probeOffset2);
        printf("Accessed index 2: %d\n", (probeOffset2 + (cubeMapFaceIndex2 * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution))));
        uint32_t incomingRadiance2 = optixLaunchParams.cubeMaps[(probeOffset2 + (cubeMapFaceIndex2 * (optixLaunchParams.cubeMapResolution * optixLaunchParams.cubeMapResolution))) + uvOffset2];
        uint32_t r2 = 0x000000ff & (incomingRadiance2);
        uint32_t g2 = (0x0000ff00 & (incomingRadiance2)) >> 8;
        uint32_t b2 = (0x00ff0000 & (incomingRadiance2)) >> 16;


        printf("Probe read value 1: %d %d %d \n", r1, g1, b1);
        printf("Probe read value 2: %d %d %d \n", r2, g2, b2);
    }
}