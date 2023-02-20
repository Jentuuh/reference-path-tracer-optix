#include "mcrt_pipeline.hpp"
#include <assert.h> 

namespace mcrt {

	McrtPipeline::McrtPipeline(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene){	}

	void McrtPipeline::init(OptixDeviceContext& context, GeometryBufferHandle& geometryBuffers, Scene& scene)
	{
		buildModule(context);
		buildDevicePrograms(context);
		buildPipeline(context);
		buildSBT(geometryBuffers, scene);
	}


	void McrtPipeline::buildGASes(OptixDeviceContext& context, std::vector<GeometryBufferHandle> geometries, std::vector<int> numsBuildInputs, std::vector<bool> disableAnyHit)
	{
		assert(geometries.size() == numsBuildInputs.size() && "buildGASes: The size of `geometries` should be equal to the size of `numsBuildInputs`!");

		for (int i = 0; i < geometries.size(); i++)
		{
			GASes.push_back(GAS{ context, geometries[i], numsBuildInputs[i], disableAnyHit[i]});
		}
	}

	void McrtPipeline::buildIAS(OptixDeviceContext& context, std::vector<glm::mat4> transforms, std::vector<GAS> gases, int numRayTypes, std::vector<int> gasIndices)
	{
		ias = std::make_shared<IAS>( context, transforms, gases, numRayTypes, gasIndices );
	}

	void McrtPipeline::uploadLaunchParams()
	{
	}
}