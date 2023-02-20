#include "ias.hpp"

namespace mcrt {
	IAS::IAS(OptixDeviceContext& context, std::vector<glm::mat4> transforms, std::vector<GAS> gases, int numRayTypes, std::vector<int> gasIndices)
	{
		// Initialize OptixInstances
		std::vector<OptixInstance> instances;
		unsigned flags = OPTIX_INSTANCE_FLAG_NONE;

		unsigned int sbtOffset = 0;
		for (int i = 0; i < transforms.size(); i++)
		{
			// If i == 0 the SBT offset needs to be 0
			int prevGasNumBuildInputs = 0;
			if (i > 0) {
				// We take the amount of build inputs of the previous GAS to decide our SBT offset
				prevGasNumBuildInputs = gases[gasIndices[i - 1]].getNumBuildInputs();
			}

			sbtOffset = sbtOffset + numRayTypes * prevGasNumBuildInputs;	// Assuming that each GAS only has 1 SBT record per build input!
			
			OptixInstance instance = {};
			memcpy(instance.transform, glm::value_ptr(transforms[i]), 12 * sizeof(float));
			instance.instanceId = i;
			instance.sbtOffset = sbtOffset;
			instance.visibilityMask = 255;
			instance.flags = flags;
			instance.traversableHandle = gases[gasIndices[i]].traversableHandle();

			instances.push_back(instance);
		}

		// Build the actual IAS
		build(context, instances);
	}


	void IAS::build(OptixDeviceContext& context, const std::vector<OptixInstance>& instances)
	{
		// ==================================================================
		// Instances
		// ==================================================================
		unsigned numInstances = instances.size();
		unsigned numBytes = sizeof(OptixInstance) * numInstances;

		// Upload instances data to device
		d_instances.alloc_and_upload(instances);

		OptixBuildInput buildInput = {};

		buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		buildInput.instanceArray.instances = d_instances.d_pointer();
		buildInput.instanceArray.numInstances = numInstances;
	

		// ==================================================================
		// Compute memory usage
		// ==================================================================
		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
									OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

		OptixAccelBufferSizes iasBufferSizes;

		OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &buildInput, 1, &iasBufferSizes));
		
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
		tempBuffer.alloc(iasBufferSizes.tempSizeInBytes);

		CUDABuffer outputBuffer;
		outputBuffer.alloc(iasBufferSizes.outputSizeInBytes);

		OPTIX_CHECK(optixAccelBuild(context,
			0,                  // CUDA stream
			&accel_options,
			&buildInput,
			1,                  // num build inputs
			tempBuffer.d_pointer(),
			tempBuffer.sizeInBytes,
			outputBuffer.d_pointer(),
			outputBuffer.sizeInBytes,
			&iasTraversableHandle,
			&emitDesc,      
			1                  
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
			iasTraversableHandle,
			accelerationStructBuffer.d_pointer(),
			accelerationStructBuffer.sizeInBytes,
			&iasTraversableHandle));
		CUDA_SYNC_CHECK();

		// ==================================================================
		// aaaaaand .... clean up
		// ==================================================================
		outputBuffer.free(); // << the UNcompacted, temporary output buffer
		tempBuffer.free();
		compactedSizeBuffer.free();
	}

}