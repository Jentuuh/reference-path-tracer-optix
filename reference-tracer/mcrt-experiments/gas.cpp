#include "gas.hpp"

namespace mcrt {
	GAS::GAS(OptixDeviceContext& context, GeometryBufferHandle& geometry, int numBuildInputs, bool disableAnyHit): numBuildInputs{numBuildInputs}
	{
        // ==================================================================
        // Triangle inputs
        // ==================================================================
        std::vector<OptixBuildInput> triangleInputs(numBuildInputs);
        std::vector<CUdeviceptr> d_vertices(numBuildInputs);
        std::vector<CUdeviceptr> d_indices(numBuildInputs);
        std::vector<uint32_t> triangleInputFlags(numBuildInputs);


        for (int meshID = 0; meshID < numBuildInputs; meshID++) {
            // upload the model to the device: the builder
            triangleInputs[meshID] = {};
            triangleInputs[meshID].type
                = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            unsigned flags;
            if (disableAnyHit) {
                flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
            }
            else {
                flags = OPTIX_GEOMETRY_FLAG_NONE;
            }

            // create local variables, because we need a *pointer* to the
            // device pointers
            d_vertices[meshID] = geometry.vertices[meshID].d_pointer();
            d_indices[meshID] = geometry.indices[meshID].d_pointer();

            triangleInputs[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangleInputs[meshID].triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
            triangleInputs[meshID].triangleArray.numVertices = geometry.amountVertices[meshID];
            triangleInputs[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

            triangleInputs[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangleInputs[meshID].triangleArray.indexStrideInBytes = sizeof(glm::ivec3);
            triangleInputs[meshID].triangleArray.numIndexTriplets = geometry.amountIndices[meshID];
            triangleInputs[meshID].triangleArray.indexBuffer = d_indices[meshID];
            triangleInputFlags[meshID] = flags;

            // in this example we have one SBT entry, and no per-primitive
            // materials:
            triangleInputs[meshID].triangleArray.flags = &triangleInputFlags[meshID];
            triangleInputs[meshID].triangleArray.numSbtRecords = 1;
            triangleInputs[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
            triangleInputs[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
            triangleInputs[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
        }

        // ==================================================================
        // Compute memory usage
        // ==================================================================

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
            | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelOptions.motionOptions.numKeys = 1;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage
        (context,
            &accelOptions,
            triangleInputs.data(),
            numBuildInputs,  // num_build_inputs
            &gasBufferSizes
        ));

        // ==================================================================
        // prepare compaction
        // ==================================================================
        CUDABuffer compactedSizeBuffer;
        compactedSizeBuffer.alloc(sizeof(uint64_t));

        OptixAccelEmitDesc emitDesc;
        emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitDesc.result = compactedSizeBuffer.d_pointer();

       // ==================================================================
       // execute build (main stage)
       // ==================================================================

        CUDABuffer tempBuffer;
        tempBuffer.alloc(gasBufferSizes.tempSizeInBytes);

        CUDABuffer outputBuffer;
        outputBuffer.alloc(gasBufferSizes.outputSizeInBytes);

        OPTIX_CHECK(optixAccelBuild(context,
            /* stream */0,
            &accelOptions,
            triangleInputs.data(),
            numBuildInputs,
            tempBuffer.d_pointer(),
            tempBuffer.sizeInBytes,
            outputBuffer.d_pointer(),
            outputBuffer.sizeInBytes,
            &gasTraversableHandle,
            &emitDesc, 1
        ));
        CUDA_SYNC_CHECK();

        // ==================================================================
        // perform compaction
        // ==================================================================
        uint64_t compactedSize;
        compactedSizeBuffer.download(&compactedSize, 1);

        accelerationStructBuffer.alloc(compactedSize);
        OPTIX_CHECK(optixAccelCompact(context,
            /*stream:*/0,
            gasTraversableHandle,
            accelerationStructBuffer.d_pointer(),
            accelerationStructBuffer.sizeInBytes,
            &gasTraversableHandle));
        CUDA_SYNC_CHECK();

        // ==================================================================
        // aaaaaand .... clean up
        // ==================================================================
        outputBuffer.free(); // << the UNcompacted, temporary output buffer
        tempBuffer.free();
        compactedSizeBuffer.free();
	}

}