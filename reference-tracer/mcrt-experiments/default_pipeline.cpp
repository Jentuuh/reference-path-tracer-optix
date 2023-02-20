#include "default_pipeline.hpp"
#include <iostream>

namespace mcrt {
    extern "C" char embedded_ptx_code[];

    // SBT record for a raygen program
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void* data;
    };

    // SBT record for a miss program
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void* data;
    };

    // SBT record for a hitgroup program
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        MeshSBTData data;
    };

	DefaultPipeline::DefaultPipeline(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene): McrtPipeline(context, geometryBuffers, scene){
        init(context, geometryBuffers, scene);
        launchParams.traversable = buildAccelerationStructure(context, geometryBuffers, scene);
        launchParamsBuffer.alloc(sizeof(launchParams));
    }

    void DefaultPipeline::uploadLaunchParams()
    {
        launchParamsBuffer.upload(&launchParams, 1);
    }


	void DefaultPipeline::buildModule(OptixDeviceContext& context)
	{
        moduleCompileOptions.maxRegisterCount = 50;
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        pipelineCompileOptions = {};
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineCompileOptions.usesMotionBlur = false;
        pipelineCompileOptions.numPayloadValues = 3;
        pipelineCompileOptions.numAttributeValues = 2;
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

        // Max # of ray bounces
        pipelineLinkOptions.maxTraceDepth = 2;

        const std::string ptxCode = embedded_ptx_code;

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixModuleCreateFromPTX(context,
            &moduleCompileOptions,
            &pipelineCompileOptions,
            ptxCode.c_str(),
            ptxCode.size(),
            log, &sizeof_log,
            &module
        ));
        if (sizeof_log > 1)
        {
            std::cout << log << std::endl;
        }
	}

	void DefaultPipeline::buildDevicePrograms(OptixDeviceContext& context)
	{
        //---------------------------------------
        //  RAYGEN PROGRAMS
        //---------------------------------------
        raygenPGs.resize(1);

        OptixProgramGroupOptions pgOptionsRaygen = {};
        OptixProgramGroupDesc pgDescRaygen = {};
        pgDescRaygen.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDescRaygen.raygen.module = module;
        pgDescRaygen.raygen.entryFunctionName = "__raygen__renderFrame";

        // OptixProgramGroup raypg;
        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(context,
            &pgDescRaygen,
            1,
            &pgOptionsRaygen,
            log, &sizeof_log,
            &raygenPGs[0]
        ));

        if (sizeof_log > 1)
        {
            std::cout << log << std::endl;
        }

        //---------------------------------------
        //  MISS PROGRAMS
        //---------------------------------------
        missPGs.resize(RAY_TYPE_COUNT);

        OptixProgramGroupOptions pgOptionsMiss = {};
        OptixProgramGroupDesc pgDescMiss = {};
        pgDescMiss.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDescMiss.miss.module = module;

        // ------------------------------------------------------------------
        // radiance rays
        // ------------------------------------------------------------------
        pgDescMiss.miss.entryFunctionName = "__miss__radiance";

        OPTIX_CHECK(optixProgramGroupCreate(context,
            &pgDescMiss,
            1,
            &pgOptionsMiss,
            log, &sizeof_log,
            &missPGs[RADIANCE_RAY_TYPE]
        ));

        if (sizeof_log > 1)
        {
            std::cout << log << std::endl;
        }

        // ------------------------------------------------------------------
        // shadow rays
        // ------------------------------------------------------------------
        pgDescMiss.miss.entryFunctionName = "__miss__shadow";

        OPTIX_CHECK(optixProgramGroupCreate(context,
            &pgDescMiss,
            1,
            &pgOptionsMiss,
            log, &sizeof_log,
            &missPGs[SHADOW_RAY_TYPE]
        ));

        if (sizeof_log > 1)
        {
            std::cout << log << std::endl;
        }

        //---------------------------------------
        //  HITGROUP PROGRAMS
        //---------------------------------------
         // Single hitgroup program for now
        hitgroupPGs.resize(RAY_TYPE_COUNT);

        OptixProgramGroupOptions pgOptionsHitgroup = {};
        OptixProgramGroupDesc    pgDescHitgroup = {};
        pgDescHitgroup.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDescHitgroup.hitgroup.moduleCH = module;
        pgDescHitgroup.hitgroup.moduleAH = module;

        pgDescHitgroup.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        pgDescHitgroup.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

        OPTIX_CHECK(optixProgramGroupCreate(context,
            &pgDescHitgroup,
            1,
            &pgOptionsHitgroup,
            log, &sizeof_log,
            &hitgroupPGs[RADIANCE_RAY_TYPE]
        ));

        if (sizeof_log > 1)
        {
            std::cout << log << std::endl;
        }

        // -------------------------------------------------------
        // shadow rays: technically we don't need this hit group,
        // since we just use the miss shader to check if we were not
        // in shadow
        // -------------------------------------------------------
        pgDescHitgroup.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
        pgDescHitgroup.hitgroup.entryFunctionNameAH = "__anyhit__shadow";

        OPTIX_CHECK(optixProgramGroupCreate(context,
            &pgDescHitgroup,
            1,
            &pgOptionsHitgroup,
            log, &sizeof_log,
            &hitgroupPGs[SHADOW_RAY_TYPE]
        ));

        if (sizeof_log > 1)
        {
            std::cout << log << std::endl;
        }
	}

	void DefaultPipeline::buildSBT(GeometryBufferHandle& geometryBuffers, Scene& scene)
	{
        // ----------------------------------------
        // Build raygen records
        // ----------------------------------------
        std::vector<RaygenRecord> raygenRecords;
        for (int i = 0; i < raygenPGs.size(); i++) {
            RaygenRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
            rec.data = nullptr; /* for now ... */
            raygenRecords.push_back(rec);
        }
        // Upload records to device
        raygenRecordsBuffer.alloc_and_upload(raygenRecords);
        // Maintain a pointer to the device memory
        sbt.raygenRecord = raygenRecordsBuffer.d_pointer();


        // ----------------------------------------
        // Build miss records
        // ----------------------------------------
        std::vector<MissRecord> missRecords;
        for (int i = 0; i < missPGs.size(); i++) {
            MissRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
            rec.data = nullptr; /* for now ... */
            missRecords.push_back(rec);
        }
        // Upload records to device
        missRecordsBuffer.alloc_and_upload(missRecords);
        // Maintain a pointer to the device memory
        sbt.missRecordBase = missRecordsBuffer.d_pointer();

        sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sbt.missRecordCount = (int)missRecords.size();


        // ----------------------------------------
        // Build hitgroup records
        // ----------------------------------------
        int numObjects = scene.numObjects();
        std::vector<HitgroupRecord> hitgroupRecords;
        for (int i = 0; i < numObjects; i++) {
            for (int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++) {

                auto mesh = scene.getGameObjects()[i]->model->mesh;

                int objectType = 0;
                HitgroupRecord rec;
                OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[rayID], &rec));
                rec.data.color = mesh->diffuse;
                if (mesh->diffuseTextureID >= 0) {
                    rec.data.hasTexture = true;
                    rec.data.texture = geometryBuffers.textureObjects[mesh->diffuseTextureID];
                }
                else {
                    rec.data.hasTexture = false;
                }

                rec.data.objectType = 0;
                rec.data.vertex = (glm::vec3*)geometryBuffers.vertices[i].d_pointer();
                rec.data.index = (glm::ivec3*)geometryBuffers.indices[i].d_pointer();
                rec.data.normal = (glm::vec3*)geometryBuffers.normals[i].d_pointer();
                rec.data.texcoord = (glm::vec2*)geometryBuffers.texCoords[i].d_pointer();
                hitgroupRecords.push_back(rec);
            }
        }

        // Record for the radiance grid
        // TODO: THIS NEEDS TO BE MOVED TO ANOTHER PIPELINE LATER!!
     /*   int gridSize = scene.grid.resolution.x * scene.grid.resolution.y * scene.grid.resolution.z;
        for (int i = 0; i < gridSize; i++) {
            int objectType = 0;
            HitgroupRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType], &rec));
            rec.data.objectType = 1;
            rec.data.vertex = (glm::vec3*)geometryBuffers.vertices[numObjects + i].d_pointer();
            rec.data.index = (glm::ivec3*)geometryBuffers.indices[numObjects + i].d_pointer();
            rec.data.color = glm::vec3{ 0.0f, 1.0f, 0.0f };
            hitgroupRecords.push_back(rec);
        }*/

        // Upload records to device
        hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
        // Maintain a pointer to the device memory
        sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
        sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
	}

	void DefaultPipeline::buildPipeline(OptixDeviceContext& context)
	{
        // Gather all program groups
        std::vector<OptixProgramGroup> programGroups;
        for (auto pg : raygenPGs)
            programGroups.push_back(pg);
        for (auto pg : missPGs)
            programGroups.push_back(pg);
        for (auto pg : hitgroupPGs)
            programGroups.push_back(pg);

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(context,
            &pipelineCompileOptions,
            &pipelineLinkOptions,
            programGroups.data(),
            (int)programGroups.size(),
            log, &sizeof_log,
            &pipeline
        ));

        if (sizeof_log > 1)
        {
            std::cout << log << std::endl;
        }

        OPTIX_CHECK(optixPipelineSetStackSize
        (/* [in] The pipeline to configure the stack size for */
            pipeline,
            /* [in] The direct stack size requirement for direct
               callables invoked from IS or AH. */
            2 * 1024,
            /* [in] The direct stack size requirement for direct
               callables invoked from RG, MS, or CH.  */
            2 * 1024,
            /* [in] The continuation stack requirement. */
            2 * 1024,
            /* [in] The maximum depth of a traversable graph
               passed to trace. */
            1))
	}

	OptixTraversableHandle DefaultPipeline::buildAccelerationStructure(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene)
	{
        //int bufferSize = scene.numObjects() + (scene.grid.resolution.x * scene.grid.resolution.y * scene.grid.resolution.z);
        int bufferSize = scene.numObjects();

        OptixTraversableHandle asHandle{ 0 };

        // ==================================================================
        // Triangle inputs
        // ==================================================================
        std::vector<OptixBuildInput> triangleInput(bufferSize);
        std::vector<CUdeviceptr> d_vertices(bufferSize);
        std::vector<CUdeviceptr> d_indices(bufferSize);
        std::vector<uint32_t> triangleInputFlags(bufferSize);

        for (int meshID = 0; meshID < scene.numObjects(); meshID++) {
            // upload the model to the device: the builder
            std::shared_ptr<Model> model = scene.getGameObjects()[meshID]->model;
            std::shared_ptr<TriangleMesh> mesh = model->mesh;

            triangleInput[meshID] = {};
            triangleInput[meshID].type
                = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            // create local variables, because we need a *pointer* to the
            // device pointers
            d_vertices[meshID] = geometryBuffers.vertices[meshID].d_pointer();
            d_indices[meshID] = geometryBuffers.indices[meshID].d_pointer();

            triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
            triangleInput[meshID].triangleArray.numVertices = (int)model->mesh->vertices.size();
            triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

            triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(glm::ivec3);
            triangleInput[meshID].triangleArray.numIndexTriplets = (int)model->mesh->indices.size();
            triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

            triangleInputFlags[meshID] = 0;

            // in this example we have one SBT entry, and no per-primitive
            // materials:
            triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
            triangleInput[meshID].triangleArray.numSbtRecords = 1;
            triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
            triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
            triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
        }

        // We also do this setup for the radiance grid 
        // TODO: THIS NEEDS TO BE MOVED LATER TO ANOTHER PIPELINE!!!
        //for (int i = scene.getGameObjects().size(); i < bufferSize; i++)
        //{
        //    std::vector<glm::vec3> verts = scene.grid.getCell(i - scene.getGameObjects().size()).getVertices();
        //    std::vector<glm::ivec3> inds = scene.grid.getCell(i - scene.getGameObjects().size()).getIndices();

        //    vertexBuffers[i].alloc_and_upload(verts);
        //    indexBuffers[i].alloc_and_upload(inds);

        //    triangleInput[i] = {};
        //    triangleInput[i].type
        //        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        //    // create local variables, because we need a *pointer* to the
        //    // device pointers
        //    d_vertices[i] = vertexBuffers[i].d_pointer();
        //    d_indices[i] = indexBuffers[i].d_pointer();

        //    triangleInput[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        //    triangleInput[i].triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
        //    triangleInput[i].triangleArray.numVertices = (int)verts.size();
        //    triangleInput[i].triangleArray.vertexBuffers = &d_vertices[i];

        //    triangleInput[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        //    triangleInput[i].triangleArray.indexStrideInBytes = sizeof(glm::ivec3);
        //    triangleInput[i].triangleArray.numIndexTriplets = (int)inds.size();
        //    triangleInput[i].triangleArray.indexBuffer = d_indices[i];

        //    triangleInputFlags[i] = 0;

        //    // in this example we have one SBT entry, and no per-primitive
        //    // materials:
        //    triangleInput[i].triangleArray.flags = &triangleInputFlags[i];
        //    triangleInput[i].triangleArray.numSbtRecords = 1;
        //    triangleInput[i].triangleArray.sbtIndexOffsetBuffer = 0;
        //    triangleInput[i].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        //    triangleInput[i].triangleArray.sbtIndexOffsetStrideInBytes = 0;
        //}

        // ==================================================================
        // BLAS setup
        // ==================================================================

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
            | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
            ;
        accelOptions.motionOptions.numKeys = 1;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes blasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage
        (context,
            &accelOptions,
            triangleInput.data(),
            bufferSize,  // num_build_inputs
            &blasBufferSizes
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
        tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

        CUDABuffer outputBuffer;
        outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

        OPTIX_CHECK(optixAccelBuild(context,
            /* stream */0,
            &accelOptions,
            triangleInput.data(),
            bufferSize,
            tempBuffer.d_pointer(),
            tempBuffer.sizeInBytes,

            outputBuffer.d_pointer(),
            outputBuffer.sizeInBytes,

            &asHandle,

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
            asHandle,
            accelerationStructBuffer.d_pointer(),
            accelerationStructBuffer.sizeInBytes,
            &asHandle));
        CUDA_SYNC_CHECK();

        // ==================================================================
        // aaaaaand .... clean up
        // ==================================================================
        outputBuffer.free(); // << the UNcompacted, temporary output buffer
        tempBuffer.free();
        compactedSizeBuffer.free();

        return asHandle;
	}

}