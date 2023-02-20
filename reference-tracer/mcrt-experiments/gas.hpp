#pragma once
#include "CUDABuffer.hpp"
#include "LaunchParams.hpp"
#include <iostream>

namespace mcrt {
	struct GeometryBufferHandle {
		std::vector<CUDABuffer>& vertices;
		std::vector<CUDABuffer>& indices;
		std::vector<CUDABuffer>& normals;
		std::vector<CUDABuffer>& texCoords;
		std::vector<cudaTextureObject_t>& textureObjects;
		std::vector<int> amountVertices;
		std::vector<int> amountIndices;
	};

	class GAS
	{
	public:
		GAS(OptixDeviceContext& context, GeometryBufferHandle& geometry, int numBuildInputs, bool disableAnyHit);

		OptixTraversableHandle traversableHandle() { return gasTraversableHandle; };
		int getNumBuildInputs() { return numBuildInputs; };
	private:
		int numBuildInputs;
		CUDABuffer accelerationStructBuffer;
		OptixTraversableHandle gasTraversableHandle;
	};
}

